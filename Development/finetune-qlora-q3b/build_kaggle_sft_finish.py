"""Builds kaggle-sft-finish-q3b.ipynb — resume the 3B SFT from checkpoint-600
and finish epochs 2-3 on Kaggle. Mirrors the teammate q7b Kaggle pattern:
device_map='auto' + max_memory (auto 1 or 2 T4), /kaggle/working outputs,
adapter zip for the inference step. Config matches our Colab run so resume is exact.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Finish 3B QLoRA-SFT on Kaggle (resume from checkpoint-600)

Continues our Colab SFT (stopped at step 600 / 1.6 epochs) to the full 3 epochs.
Config is **identical** to the Colab run so the resume is exact (1500 train,
MAX_PIXELS=768, batch 1, grad_accum 4, 3 epochs → 1125 total steps).

**Before running — upload the checkpoint as a Kaggle Dataset:**
1. Zip `ckpt_scaled/` (or just `checkpoint-600`) on your PC.
2. Kaggle → Datasets → New Dataset → upload → name it e.g. `ckpt-scaled-q3b`.
3. In this notebook: Add Input → your dataset. It mounts at `/kaggle/input/ckpt-scaled-q3b/`.
4. Settings → Accelerator → **GPU T4 x2** (or T4). Add your HF token as a Secret named `HF_TOKEN`.

Outputs land in `/kaggle/working/` (downloadable): final adapter + `adapter_sft_final.zip`.""")

md("## 1. Install")
code("""import os, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
!pip install -q -U "transformers>=4.49.0" "peft>=0.10.0" accelerate bitsandbytes qwen-vl-utils
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GB)')""")

md("""## 2. Config + HF login + find checkpoint

Auto-searches `/kaggle/input` for the `checkpoint-600` folder, so the exact
dataset name doesn't matter.""")
code("""import os, glob
from huggingface_hub import login
try:
    from kaggle_secrets import UserSecretsClient
    login(token=UserSecretsClient().get_secret('HF_TOKEN'))
    print('Logged in via Kaggle Secret.')
except Exception as e:
    print('No HF secret (public data still works):', e)

REPO_ID   = 'QCRI/AynVQA-ArabicNLP26'
VLM_MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
MAX_PIXELS = 768 * 28 * 28
N_TRAIN   = 1500          # MUST match the Colab run
EPOCHS    = 3
BATCH, GRAD_ACCUM, LR = 1, 4, 2e-4

OUTPUT_DIR    = '/kaggle/working/checkpoints'
FINAL_ADAPTER = '/kaggle/working/adapter_sft_final'
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(FINAL_ADAPTER, exist_ok=True)

# locate checkpoint-600 anywhere under /kaggle/input
RESUME_CKPT = None
for root, dirs, files in os.walk('/kaggle/input'):
    if os.path.basename(root) == 'checkpoint-600' and 'adapter_config.json' in files:
        RESUME_CKPT = root; break
assert RESUME_CKPT, 'checkpoint-600 not found under /kaggle/input — add your dataset.'
print('Resume from:', RESUME_CKPT)""")

md("## 3. Download train split + images (first 1500, deterministic — matches Colab)")
code("""import json
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

jsonl = hf_hub_download(REPO_ID, filename='task1b/train_en.jsonl', repo_type='dataset')
train = [json.loads(l) for l in open(jsonl, encoding='utf-8') if l.strip()][:N_TRAIN]
print('train items:', len(train))
need = sorted({r['image'] for r in train})
img_path = {}
for rel in tqdm(need, desc='images'):
    img_path[rel] = hf_hub_download(REPO_ID, filename=rel, repo_type='dataset')
print('images:', len(img_path))""")

md("""## 4. Load base 3B (4-bit) across all GPUs + attach LoRA (matches checkpoint)

`device_map='auto'` + `max_memory` balances the model over 1 **or** 2 T4s automatically.""")
code("""import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

N_GPUS = torch.cuda.device_count()
max_mem = {i: '13500MiB' for i in range(N_GPUS)}; max_mem['cpu'] = '8GiB'

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(VLM_MODEL, max_pixels=MAX_PIXELS)
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    VLM_MODEL, torch_dtype=torch.float16, device_map='auto',
    max_memory=max_mem, quantization_config=bnb)
if hasattr(base, 'hf_device_map'):
    from collections import Counter
    print('Layer spread over GPUs:', dict(Counter(base.hf_device_map.values())))

base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none',
                  task_type=TaskType.CAUSAL_LM,
                  target_modules=['q_proj','k_proj','v_proj','o_proj'])
model = get_peft_model(base, lora)
model.print_trainable_parameters()""")

md("## 5. Data collator + training set (identical to Colab)")
code('''import re
from qwen_vl_utils import process_vision_info

JOINT_PROMPT = (
    'You are a visual fact-checker examining an image from the Arab world.\\n'
    'Below are THREE statements about this image. '
    'Exactly ONE statement is grounded in the image (True). '
    'The other two are plausible-sounding hallucinations (False).\\n\\n'
    'Statement 1: {s0}\\nStatement 2: {s1}\\nStatement 3: {s2}\\n\\n'
    'Instructions:\\n- Study the image carefully.\\n'
    '- On the VERY FIRST line write ONLY: "Answer: X" where X is 1, 2, or 3.\\n'
    '- Then explain step by step why that statement is grounded '
    'and why the other two are hallucinations.\\n'
    'Do not write anything before the Answer line.')

def to_example(r):
    gold = r['labels'].index(True)
    return {'messages': [
        {'role': 'user', 'content': [
            {'type': 'image', 'image': img_path[r['image']]},
            {'type': 'text', 'text': JOINT_PROMPT.format(
                s0=r['statements'][0], s1=r['statements'][1], s2=r['statements'][2])}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': f'Answer: {gold + 1}'}]}]}

train_examples = [to_example(r) for r in train]

IMAGE_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
PAD_ID = processor.tokenizer.pad_token_id

def collate_fn(examples):
    texts, images = [], []
    for ex in examples:
        m = ex['messages']
        texts.append(processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False))
        imgs, _ = process_vision_info(m); images.append(imgs)
    batch = processor(text=texts, images=images, return_tensors='pt', padding=True)
    labels = batch['input_ids'].clone()
    labels[labels == PAD_ID] = -100
    labels[labels == IMAGE_TOKEN_ID] = -100
    batch['labels'] = labels
    return batch
print('collator +', len(train_examples), 'train examples ready')''')

md("""## 6. Resume training → finish epochs 2-3
Trainer picks up from step 600 and runs to 1125. New checkpoints saved every 200
steps to `/kaggle/working` (so a Kaggle disconnect can resume from there too).""")
code("""from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    logging_steps=10,
    save_strategy='steps', save_steps=200, save_total_limit=3,
    optim='paged_adamw_8bit',
    remove_unused_columns=False,
    report_to='none',
)
trainer = Trainer(model=model, args=args, train_dataset=train_examples, data_collator=collate_fn)
trainer.train(resume_from_checkpoint=RESUME_CKPT)
print('SFT finished.')""")

md("## 7. Save final adapter + zip (download from Kaggle output)")
code("""import glob, zipfile
model.save_pretrained(FINAL_ADAPTER)
processor.save_pretrained(FINAL_ADAPTER)
zip_path = '/kaggle/working/adapter_sft_final.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fp in glob.glob(f'{FINAL_ADAPTER}/*'):
        zf.write(fp, os.path.basename(fp))
print('Saved:', zip_path)
print('Download it from the Kaggle output panel -> re-upload as a Dataset for the inference/DPO step.')""")

nb['cells'] = cells
nb['metadata'] = {'accelerator': 'GPU', 'colab': {'provenance': []},
                  'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kaggle-sft-finish-q3b.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Wrote', out, 'with', len(cells), 'cells')
