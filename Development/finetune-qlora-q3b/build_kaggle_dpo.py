"""Builds kaggle-dpo-q3b.ipynb — DPO on top of the SFT adapter.
EXPERIMENTAL: multimodal DPO in TRL is newer than SFT and may need debugging.
Same Kaggle conventions as the SFT-finish notebook (device_map auto over 1-2 T4,
/kaggle/working outputs, adapter zip). Starts the policy from the SFT adapter.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# DPO on top of 3B SFT (Kaggle) — EXPERIMENTAL

Teaches the model to *prefer* the correct statement over a hallucinated one, using
our contrastive labels as free preference pairs (chosen = true answer, rejected =
a false answer). Policy starts from the **SFT adapter**; DPO refines it.

⚠️ Multimodal DPO (image in the prompt) via TRL is newer than SFT — expect to
debug the training cell together. Run the **SFT-finish notebook first**; it is the
reliable path. This adds the novel SFT→DPO result on top.

**Before running — upload the SFT adapter as a Kaggle Dataset:**
1. From the SFT-finish run, download `adapter_sft_final.zip`; unzip; upload the
   folder as a Kaggle Dataset (e.g. `sft-adapter-q3b`).
   (For a quick test you can instead point at `checkpoint-600` from the partial SFT.)
2. Add Input → that dataset. Accelerator → **GPU T4 x2** (or T4). Secret `HF_TOKEN`.

Outputs: `/kaggle/working/adapter_dpo_final` + `adapter_dpo_final.zip`.""")

md("## 1. Install")
code("""import os, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
!pip install -q -U "transformers>=4.49.0" "trl>=0.12.0" "peft>=0.10.0" accelerate bitsandbytes qwen-vl-utils datasets
print('GPUs:', torch.cuda.device_count())""")

md("## 2. Config + HF login + find SFT adapter")
code("""import os
from huggingface_hub import login
try:
    from kaggle_secrets import UserSecretsClient
    login(token=UserSecretsClient().get_secret('HF_TOKEN'))
    print('Logged in via Kaggle Secret.')
except Exception as e:
    print('No HF secret (public data still works):', e)

REPO_ID   = 'QCRI/AynVQA-ArabicNLP26'
VLM_MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
MAX_PIXELS = 384 * 28 * 28     # smaller for DPO (two forward passes per step)
SMOKE     = False              # smoke test passed -> full run. Set True again for a quick re-test.
N_DPO     = 200 if SMOKE else 600   # subset of train for DPO pairs (600 = good DPO amount)
FULL_STEPS = 150               # full run: 150 steps x4 = 600 pairs = 1 full epoch (~4 hr)

# find the SFT adapter folder (has adapter_config.json) under /kaggle/input
SFT_ADAPTER = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'adapter_config.json' in files and 'adapter_model.safetensors' in files:
        SFT_ADAPTER = root; break
assert SFT_ADAPTER, 'SFT adapter not found under /kaggle/input — add your dataset.'
print('SFT adapter:', SFT_ADAPTER)""")

md("## 3. Download train + images, build preference pairs")
code("""import json, random
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from PIL import Image
from datasets import Dataset

jsonl = hf_hub_download(REPO_ID, filename='task1b/train_en.jsonl', repo_type='dataset')
train = [json.loads(l) for l in open(jsonl, encoding='utf-8') if l.strip()][:N_DPO]
need = sorted({r['image'] for r in train})
img_path = {}
for rel in tqdm(need, desc='images'):
    img_path[rel] = hf_hub_download(REPO_ID, filename=rel, repo_type='dataset')

PROMPT_TEXT = (
    'You are a visual fact-checker examining an image from the Arab world.\\n'
    'Below are THREE statements about this image. Exactly ONE is grounded (True).\\n'
    'Statement 1: {s0}\\nStatement 2: {s1}\\nStatement 3: {s2}\\n'
    'On the VERY FIRST line write ONLY: "Answer: X" where X is 1, 2, or 3.')

random.seed(42)
rows = []
for r in train:
    gold = r['labels'].index(True)
    wrong = random.choice([i for i in (0,1,2) if i != gold])
    rows.append({
        'images': [Image.open(img_path[r['image']]).convert('RGB')],
        'prompt': [{'role':'user','content':[{'type':'image'},
                    {'type':'text','text': PROMPT_TEXT.format(
                        s0=r['statements'][0], s1=r['statements'][1], s2=r['statements'][2])}]}],
        'chosen':   [{'role':'assistant','content':[{'type':'text','text': f'Answer: {gold+1}'}]}],
        'rejected': [{'role':'assistant','content':[{'type':'text','text': f'Answer: {wrong+1}'}]}],
    })
dpo_ds = Dataset.from_list(rows)
print('DPO pairs:', len(dpo_ds))""")

md("## 4. Load base + SFT adapter as the policy (across 1-2 T4)")
code("""import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

N_GPUS = torch.cuda.device_count()
max_mem = {i: '13500MiB' for i in range(N_GPUS)}; max_mem['cpu'] = '8GiB'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(VLM_MODEL, max_pixels=MAX_PIXELS)
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    VLM_MODEL, torch_dtype=torch.float16, device_map='auto',
    max_memory=max_mem, quantization_config=bnb)
# policy = base + SFT adapter (trainable). DPOTrainer uses adapter-disabled base as reference.
model = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)
model.print_trainable_parameters()""")

md("""## 5. DPO train (EXPERIMENTAL — most likely to need tweaks)

⚠️ **Run this notebook INTERACTIVE, not Save & Run All**, so you see the live tqdm
bar + GPU% (right sidebar). Committed runs hide the progress bar → you fly blind.
`SMOKE=True` (cell 2) caps this at `max_steps=20` (~few min) to prove it moves before
you commit a full run. When the smoke test logs losses + saves, set `SMOKE=False`.""")
code("""from trl import DPOConfig, DPOTrainer

cfg = DPOConfig(
    output_dir='/kaggle/working/dpo_ckpts',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_steps=20 if SMOKE else FULL_STEPS,   # smoke: 20 to confirm it moves; full: FULL_STEPS (~1.5 hr)
    learning_rate=5e-6,
    beta=0.1,
    bf16=True, fp16=False,     # grads are bf16 -> use bf16 (no GradScaler); fp16=True crashes on unscale
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    logging_steps=1,           # loss line EVERY step -> live signal even in committed log
    save_strategy='steps', save_steps=25, save_total_limit=2,  # mid-run saves -> survive disconnect
    optim='paged_adamw_8bit',
    remove_unused_columns=False,
    precompute_ref_log_probs=False,  # no silent full-dataset ref pass before step 1
    dataset_num_proc=2,        # parallelize the (slow, multimodal) tokenization map
    report_to='none',
)   # length args (max_length/max_prompt_length) omitted — TRL version rejects them; defaults used
print('>>> building DPOTrainer (tokenizes dataset — may take a minute on images)...', flush=True)
trainer = DPOTrainer(
    model=model, ref_model=None,        # ref = SFT policy with adapter disabled
    args=cfg, train_dataset=dpo_ds,
    processing_class=processor,
)
print('>>> trainer built. starting trainer.train() now.', flush=True)
# auto-resume: if a checkpoint survived in output dir, continue from it instead of step 0
import glob as _g, os as _o
_cks = _g.glob('/kaggle/working/dpo_ckpts/checkpoint-*')
_resume = max(_cks, key=lambda p: int(p.split('-')[-1])) if _cks else None
print('>>> resume_from_checkpoint =', _resume, flush=True)
trainer.train(resume_from_checkpoint=_resume)
print('DPO finished.  SMOKE =', SMOKE, '(set False in cell 2 for the full run)')""")

md("## 6. Save DPO adapter + zip")
code("""import glob, zipfile, os
OUT = '/kaggle/working/adapter_dpo_final'
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT); processor.save_pretrained(OUT)
zp = '/kaggle/working/adapter_dpo_final.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fp in glob.glob(f'{OUT}/*'):
        zf.write(fp, os.path.basename(fp))
print('Saved:', zp, '-> download, then run the inference notebook to score CI.')""")

nb['cells'] = cells
nb['metadata'] = {'accelerator': 'GPU', 'colab': {'provenance': []},
                  'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kaggle-dpo-q3b.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Wrote', out, 'with', len(cells), 'cells')
