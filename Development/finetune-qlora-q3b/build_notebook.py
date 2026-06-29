"""Builds qlora-3b-colab.ipynb. Run with the repo .venv:
   .venv/Scripts/python.exe Development/finetune-qlora-q3b/build_notebook.py
Keeps the heavy JSON-escaping out of our hands; edit cells here, regenerate.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# QLoRA-SFT · Qwen2.5-VL-3B · Task 1b (fast test)

**Goal:** small-scale fine-tune test on Colab T4. Train on **200** items, evaluate on
**100** held-out dev items. Compare **baseline (no fine-tune)** vs **QLoRA-SFT**.

**Metric:** CI = error rate of picking the True statement (lower = better).

**Resumable:** every heavy step writes to Google Drive and is skipped if its output
already exists. If Colab disconnects, just re-run the cells top-to-bottom — finished
steps load from cache instead of recomputing.

| Step | Cell | Saved to Drive |
|---|---|---|
| Install | 1 | — |
| Mount Drive + paths | 2 | folders created |
| Download data | 3 | `data/`, model cache via `HF_HOME` |
| Load 3B (4-bit) | 4 | (model cached) |
| Helpers (prompt/parser/CI) | 5 | — |
| **Baseline eval** | 6 | `preds/baseline_dev.csv` |
| Build train set | 7 | — |
| **QLoRA train** | 8 | `ckpt/` (resumable), `adapter/` |
| **Fine-tuned eval** | 9 | `preds/ft_dev.csv` |
| Compare | 10 | — |

> Reduced image resolution (`MAX_PIXELS`) is used so 3B trains comfortably on a T4.
> Baseline and fine-tuned use the SAME resolution, so the comparison is fair.""")

md("## 1. GPU check + install")
code("""import torch, subprocess
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU'
print('GPU:', torch.cuda.get_device_name(0))

# Qwen2.5-VL needs transformers>=4.49. Keep installs minimal to avoid conflicts.
!pip install -q -U "transformers>=4.49.0" accelerate peft bitsandbytes qwen-vl-utils datasets
print('Install done. If you see a "restart session" prompt, restart then re-run from cell 2.')""")

md("""## 2. Mount Google Drive + project paths

All artifacts live under one Drive folder so nothing is lost on disconnect.
`HF_HOME` points into Drive too → dataset + model weights download **once**.""")
code("""import os
from google.colab import drive
drive.mount('/content/drive')

PROJ = '/content/drive/MyDrive/ie2026_haldetect'
DATA_DIR    = f'{PROJ}/data'
PREDS_DIR   = f'{PROJ}/preds'
CKPT_DIR    = f'{PROJ}/ckpt'
ADAPTER_DIR = f'{PROJ}/adapter'
HF_CACHE    = f'{PROJ}/hf_cache'
for d in (DATA_DIR, PREDS_DIR, CKPT_DIR, ADAPTER_DIR, HF_CACHE):
    os.makedirs(d, exist_ok=True)

# Persist all HuggingFace downloads (dataset + model) on Drive -> survives reconnect.
os.environ['HF_HOME'] = HF_CACHE
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
print('Project folder:', PROJ)
print('Everything saved here will survive a Colab disconnect.')""")

md("""## 3. Download data (cached)

Pulls `train_en.jsonl` + `dev_en.jsonl` from HuggingFace (public, no token needed),
takes the first **200 train** and **100 dev** items, and downloads their images.
Re-running is cheap: files already on Drive are reused.""")
code("""import json
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

REPO = 'QCRI/AynVQA-ArabicNLP26'
N_TRAIN, N_DEV = 200, 100   # <- fast-test sizes; raise later

def load_split(split):
    p = hf_hub_download(REPO, filename=f'task1b/{split}_en.jsonl', repo_type='dataset')
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

train_all = load_split('train')
dev_all   = load_split('dev')
print(f'Full sizes -> train: {len(train_all)}, dev: {len(dev_all)}')

train = train_all[:N_TRAIN]
dev   = dev_all[:N_DEV]
print(f'Using -> train: {len(train)}, dev: {len(dev)} (first N, deterministic)')

# Download the images these items need (cached on Drive via HF_HOME).
need = sorted({r['image'] for r in train + dev})
img_path = {}
for rel in tqdm(need, desc='images'):
    img_path[rel] = hf_hub_download(REPO, filename=rel, repo_type='dataset')
print(f'Image files ready: {len(img_path)}')

# sanity: each item must have labels (a True index)
assert all('labels' in r for r in train + dev), 'missing labels'
print('Sample item:', {k: train[0][k] for k in ("country","category")},
      '| true idx =', train[0]['labels'].index(True))""")

md("""## 4. Load Qwen2.5-VL-3B in 4-bit (QLoRA base)

4-bit NF4 quantization shrinks the model so it fits a T4 with room for training.""")
code("""import torch
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                          BitsAndBytesConfig)

MODEL_ID   = 'Qwen/Qwen2.5-VL-3B-Instruct'
MAX_PIXELS = 512 * 28 * 28   # reduced res for T4 training speed (same for both evals)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True)

processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map='auto',
    quantization_config=bnb)
print('Model loaded in 4-bit.')""")

md("""## 5. Helpers: prompt, parser, CI scorer (shared by both evals)

Same answer-first joint prompt as the repo's best base (Run 4). Both baseline and
fine-tuned use these identical functions → fair comparison.""")
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

def build_user_messages(image_file, statements):
    return [{'role': 'user', 'content': [
        {'type': 'image', 'image': image_file},
        {'type': 'text', 'text': JOINT_PROMPT.format(
            s0=statements[0], s1=statements[1], s2=statements[2])}]}]

def parse_answer(raw):
    for ln in [l.strip() for l in raw.splitlines() if l.strip()]:
        m = re.search(r'answer\\s*[:\\-]?\\s*([123])', ln, re.IGNORECASE)
        if m: return int(m.group(1)) - 1   # 0-indexed
    for ln in [l.strip() for l in raw.splitlines() if l.strip()]:
        if re.fullmatch(r'[123]', ln): return int(ln) - 1
    return None

@torch.no_grad()
def generate(messages, max_new_tokens=128):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(messages)
    inp = processor(text=[text], images=imgs, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    dec = processor.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    return dec.strip()

def evaluate(items, img_path, csv_out, force=False):
    """Run the model over items, save id/gold/pred/correct, return CI."""
    import csv, os
    if os.path.exists(csv_out) and not force:
        rows = list(csv.DictReader(open(csv_out)))
        ci = 1 - sum(int(r['correct']) for r in rows) / len(rows)
        print(f'Loaded cached {csv_out} -> CI={ci:.4f} ({len(rows)} items)')
        return ci
    rows = []
    for r in tqdm(items, desc='eval'):
        gold = r['labels'].index(True)
        raw = generate(build_user_messages(img_path[r['image']], r['statements']))
        pred = parse_answer(raw)
        if pred is None: pred = 0   # unparsed -> wrong guess
        rows.append({'id': r['id'], 'gold': gold, 'pred': pred,
                     'correct': int(pred == gold)})
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id','gold','pred','correct']); w.writeheader(); w.writerows(rows)
    ci = 1 - sum(x['correct'] for x in rows) / len(rows)
    print(f'Saved {csv_out} -> CI={ci:.4f} ({len(rows)} items)')
    return ci

print('Helpers ready.')''')

md("""## 6. ⭐ Baseline eval (no fine-tune)

Run the plain 3B on the 100 dev items. This is our **"before"** number.
Saved to `preds/baseline_dev.csv` — re-running loads it instead of recomputing.""")
code("""model.eval()
ci_baseline = evaluate(dev, img_path, f'{PREDS_DIR}/baseline_dev.csv')
print(f'\\nBASELINE CI = {ci_baseline:.4f}  (accuracy = {1-ci_baseline:.4f})')""")

md("""## 7. Build the training set (200 items)

Each example = image + the same prompt + the **correct** answer as the target the
model should learn to produce: `Answer: <correct number>`.""")
code("""def to_train_example(r):
    gold = r['labels'].index(True)               # 0-indexed
    msgs = build_user_messages(img_path[r['image']], r['statements'])
    msgs.append({'role': 'assistant', 'content': [
        {'type': 'text', 'text': f'Answer: {gold + 1}'}]})   # 1-indexed target
    return {'messages': msgs}

train_examples = [to_train_example(r) for r in train]
print(f'Built {len(train_examples)} training examples.')
print('Example target:', train_examples[0]['messages'][-1]['content'][0]['text'])""")

md("""## 8. ⭐ QLoRA fine-tune (resumable)

Wraps the 4-bit model with small LoRA adapters and trains with a custom collator
(handles images + masks padding/image tokens from the loss). Checkpoints go to
`ckpt/` — if training is interrupted, re-running resumes from the last checkpoint.""")
code('''import os, glob
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments

IMAGE_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
PAD_ID = processor.tokenizer.pad_token_id

def collate_fn(examples):
    texts, images = [], []
    for ex in examples:
        msgs = ex['messages']
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        imgs, _ = process_vision_info(msgs)
        images.append(imgs)
    batch = processor(text=texts, images=images, return_tensors='pt', padding=True)
    labels = batch['input_ids'].clone()
    labels[labels == PAD_ID] = -100
    labels[labels == IMAGE_TOKEN_ID] = -100         # don't train on image placeholder tokens
    batch['labels'] = labels
    return batch

model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none',
                  task_type='CAUSAL_LM',
                  target_modules=['q_proj','k_proj','v_proj','o_proj'])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=CKPT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    logging_steps=5,
    save_strategy='epoch',
    save_total_limit=2,
    optim='paged_adamw_8bit',
    remove_unused_columns=False,    # keep 'messages' for our collator
    report_to='none',
)

trainer = Trainer(model=model, args=args, train_dataset=train_examples,
                  data_collator=collate_fn)

resume = bool(glob.glob(f'{CKPT_DIR}/checkpoint-*'))
print('Resuming from checkpoint.' if resume else 'Starting fresh training.')
trainer.train(resume_from_checkpoint=resume)

model.save_pretrained(ADAPTER_DIR)
print('Adapter saved to', ADAPTER_DIR)''')

md("""## 9. ⭐ Fine-tuned eval (same 100 dev items)

Switch the model back to inference mode and score the SAME dev items.
Saved to `preds/ft_dev.csv`.""")
code("""model.config.use_cache = True
model.gradient_checkpointing_disable()
model.eval()
ci_ft = evaluate(dev, img_path, f'{PREDS_DIR}/ft_dev.csv', force=True)
print(f'\\nFINE-TUNED CI = {ci_ft:.4f}  (accuracy = {1-ci_ft:.4f})')""")

md("## 10. Compare")
code("""# %-formatting (no nested quotes) so it runs on any Colab Python version.
print('%-14s %9s %9s' % ('', 'CI', 'Acc'))
print('-' * 34)
print('%-14s %9.4f %9.4f' % ('Baseline',  ci_baseline, 1 - ci_baseline))
print('%-14s %9.4f %9.4f' % ('QLoRA-SFT', ci_ft,       1 - ci_ft))
delta = ci_ft - ci_baseline
print('-' * 34)
print('Delta CI: %+.4f  ->' % delta,
      'BETTER (fine-tune helped)' if delta < 0 else 'no improvement')
print('\\nNote: only 100 dev items + reduced resolution. Trend matters more than exact CI.')
print('Next: raise N_TRAIN/N_DEV + MAX_PIXELS, add CoT5 prompt, then replicate on 7B.')""")

nb['cells'] = cells
nb['metadata'] = {'accelerator': 'GPU',
                  'colab': {'provenance': [], 'gpuType': 'T4'},
                  'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qlora-3b-colab.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Wrote', out, 'with', len(cells), 'cells')
