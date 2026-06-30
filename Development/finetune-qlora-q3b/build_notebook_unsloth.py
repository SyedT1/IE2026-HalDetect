"""Builds qlora-3b-unsloth-colab.ipynb (Unsloth-accelerated Option A).
Run with repo .venv:
   .venv/Scripts/python.exe Development/finetune-qlora-q3b/build_notebook_unsloth.py
Speed/robustness vs the plain notebook:
 - Unsloth FastVisionModel (~2x faster, ~50% less VRAM)
 - eval uses max_new_tokens=12 (only the "Answer: X" line) -> ~5x faster eval
 - per-item eval save + 200-step train checkpoints -> resumable on disconnect
 - N_TRAIN=2000 default (fits a free-Colab window; raise to 3000 for full data)
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# QLoRA-SFT (Unsloth) · Qwen2.5-VL-3B · Task 1b

Speed-optimized rebuild after free-Colab GPU limits. Key changes:
- **Unsloth** FastVisionModel — ~2x faster training, ~50% less VRAM.
- **Fast eval** — generates only the `Answer: X` line (12 tokens), ~5x faster.
- **Resumable** — per-item eval save + 200-step train checkpoints on Drive.
- **N_TRAIN=2000** by default (fits a free window; set 3000 for full data).

Run top-to-bottom. On disconnect, re-run from the top — finished steps load from
Drive cache / checkpoints.""")

md("""## 1. Install Unsloth
If Colab asks to **restart session** after this, restart, then run from cell 2.""")
code("""import torch
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU'
print('GPU:', torch.cuda.get_device_name(0))
!pip install -q unsloth
# qwen-vl-utils for the eval path (image loading). Unsloth pulls peft/trl/bitsandbytes.
!pip install -q qwen-vl-utils
print('Install done.')""")

md("""## 2. Mount Drive + paths
Uses *_unsloth folders so earlier runs stay intact.""")
code("""import os
from google.colab import drive
drive.mount('/content/drive')

PROJ = '/content/drive/MyDrive/ie2026_haldetect'
DATA_DIR  = f'{PROJ}/data'
PREDS_DIR = f'{PROJ}/preds_unsloth'
CKPT_DIR  = f'{PROJ}/ckpt_unsloth'
ADAPTER_DIR = f'{PROJ}/adapter_unsloth'
RESULTS_DIR = f'{PROJ}/results'
HF_CACHE  = f'{PROJ}/hf_cache'
for d in (DATA_DIR, PREDS_DIR, CKPT_DIR, ADAPTER_DIR, RESULTS_DIR, HF_CACHE):
    os.makedirs(d, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
print('Project:', PROJ)""")

md("""## 3. Download data (cached)
`N_TRAIN=2000` fits a free window; raise to 3000 later for the full-data paper run.""")
code("""import json
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

REPO = 'QCRI/AynVQA-ArabicNLP26'
N_TRAIN, N_DEV = 2000, 500

def load_split(split):
    p = hf_hub_download(REPO, filename=f'task1b/{split}_en.jsonl', repo_type='dataset')
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

train = load_split('train')[:N_TRAIN]
dev   = load_split('dev')[:N_DEV]
print(f'Using -> train: {len(train)}, dev: {len(dev)}')

need = sorted({r['image'] for r in train + dev})
img_path = {}
for rel in tqdm(need, desc='images'):
    img_path[rel] = hf_hub_download(REPO, filename=rel, repo_type='dataset')
print('Image files ready:', len(img_path))
assert all('labels' in r for r in train + dev)""")

md("""## 4. Load Qwen2.5-VL-3B with Unsloth (4-bit)
Uses Unsloth's pre-quantized checkpoint for a fast load.""")
code("""from unsloth import FastVisionModel
import torch

MAX_PIXELS = 768 * 28 * 28
model, processor = FastVisionModel.from_pretrained(
    'unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit',
    load_in_4bit=True,
    use_gradient_checkpointing='unsloth',
)
# keep image tokens bounded for T4 speed/memory
try:
    processor.image_processor.max_pixels = MAX_PIXELS
    print('max_pixels set to', MAX_PIXELS)
except Exception as e:
    print('could not set max_pixels:', e)
print('Unsloth model loaded.')""")

md("""## 5. Helpers: prompt, parser, fast resumable eval
`evaluate` saves each item immediately and skips already-done ids on re-run.
`generate` produces only the short answer line (fast).""")
code('''import re, csv, os
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
        if m: return int(m.group(1)) - 1
    for ln in [l.strip() for l in raw.splitlines() if l.strip()]:
        if re.fullmatch(r'[123]', ln): return int(ln) - 1
    return None

@torch.no_grad()
def generate(messages, max_new_tokens=12):    # only the "Answer: X" line -> fast
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(messages)
    inp = processor(text=[text], images=imgs, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

def evaluate(items, img_path, csv_out):
    done = set()
    if os.path.exists(csv_out):
        done = {r['id'] for r in csv.DictReader(open(csv_out))}
    todo = [r for r in items if r['id'] not in done]
    print(f'{len(done)} already done, {len(todo)} to go')
    new_file = not os.path.exists(csv_out)
    f = open(csv_out, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=['id','gold','pred','correct'])
    if new_file:
        w.writeheader(); f.flush()
    for r in tqdm(todo, desc='eval'):
        gold = r['labels'].index(True)
        pred = parse_answer(generate(build_user_messages(img_path[r['image']], r['statements'])))
        if pred is None: pred = 0
        w.writerow({'id': r['id'], 'gold': gold, 'pred': pred, 'correct': int(pred == gold)})
        f.flush()
    f.close()
    rows = list(csv.DictReader(open(csv_out)))
    ci = 1 - sum(int(x['correct']) for x in rows) / len(rows)
    print(f'{csv_out} -> CI={ci:.4f} ({len(rows)} items)')
    return ci

print('Helpers ready.')''')

md("""## 6. ⭐ Baseline eval (no fine-tune) — full 500 dev
Fast (12-token gen) + resumable. Saved to Drive.""")
code("""FastVisionModel.for_inference(model)
ci_baseline = evaluate(dev, img_path, f'{PREDS_DIR}/baseline_dev.csv')
print(f'\\nBASELINE CI = {ci_baseline:.4f}  (accuracy = {1-ci_baseline:.4f})')""")

md("""## 7. Add LoRA adapters (Unsloth)""")
code("""model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,     # task is language-decision; keep vision frozen
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8, lora_alpha=16, lora_dropout=0, bias='none', random_state=42,
)
print('LoRA adapters added.')""")

md("""## 8. Build training set (target = `Answer: X`)""")
code("""def to_train_example(r):
    gold = r['labels'].index(True)
    msgs = build_user_messages(img_path[r['image']], r['statements'])
    msgs.append({'role': 'assistant', 'content': [{'type': 'text', 'text': f'Answer: {gold + 1}'}]})
    return {'messages': msgs}

train_examples = [to_train_example(r) for r in train]
print(f'Built {len(train_examples)} training examples.')""")

md("""## 9. ⭐ QLoRA fine-tune (Unsloth + TRL, resumable)""")
code("""import glob
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

cfg = SFTConfig(
    output_dir=CKPT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True, bf16=False,            # T4 has no bf16
    logging_steps=10,
    save_strategy='steps', save_steps=200, save_total_limit=3,
    optim='paged_adamw_8bit',
    report_to='none',
    remove_unused_columns=False,
    dataset_kwargs={'skip_prepare_dataset': True},   # we pass raw messages + vision collator
    max_seq_length=2048,
)
trainer = SFTTrainer(
    model=model, tokenizer=processor,
    data_collator=UnslothVisionDataCollator(model, processor),
    train_dataset=train_examples, args=cfg,
)
resume = bool(glob.glob(f'{CKPT_DIR}/checkpoint-*'))
print('Resuming from checkpoint.' if resume else 'Starting fresh training.')
trainer.train(resume_from_checkpoint=resume)

model.save_pretrained(ADAPTER_DIR)
processor.save_pretrained(ADAPTER_DIR)
print('Adapter saved to', ADAPTER_DIR)""")

md("""## 10. ⭐ Fine-tuned eval — full 500 dev (separate file, resumable)""")
code("""FastVisionModel.for_inference(model)
ci_ft = evaluate(dev, img_path, f'{PREDS_DIR}/ft_dev.csv')
print(f'\\nFINE-TUNED CI = {ci_ft:.4f}  (accuracy = {1-ci_ft:.4f})')""")

md("## 11. Compare")
code("""print('%-14s %9s %9s' % ('', 'CI', 'Acc'))
print('-' * 34)
print('%-14s %9.4f %9.4f' % ('Baseline',  ci_baseline, 1 - ci_baseline))
print('%-14s %9.4f %9.4f' % ('QLoRA-SFT', ci_ft,       1 - ci_ft))
delta = ci_ft - ci_baseline
print('-' * 34)
print('Delta CI: %+.4f  ->' % delta,
      'BETTER (fine-tune helped)' if delta < 0 else 'no improvement')""")

md("""## 12. Save paper results bundle (download this folder)""")
code("""import json, shutil
b = {r['id']: r for r in csv.DictReader(open(f'{PREDS_DIR}/baseline_dev.csv'))}
f = {r['id']: r for r in csv.DictReader(open(f'{PREDS_DIR}/ft_dev.csv'))}
ids = [i for i in b if i in f]
changed = sum(b[i]['pred'] != f[i]['pred'] for i in ids)
fixed   = sum(b[i]['correct']=='0' and f[i]['correct']=='1' for i in ids)
broken  = sum(b[i]['correct']=='1' and f[i]['correct']=='0' for i in ids)
results = {
    'experiment': 'Unsloth QLoRA-SFT (Qwen2.5-VL-3B)',
    'n_train': len(train), 'n_dev': len(dev), 'max_pixels': MAX_PIXELS,
    'epochs': 3, 'lr': 2e-4, 'grad_accum': 8, 'lora_r': 8, 'lora_alpha': 16,
    'ci_baseline': round(ci_baseline,4), 'acc_baseline': round(1-ci_baseline,4),
    'ci_finetuned': round(ci_ft,4), 'acc_finetuned': round(1-ci_ft,4),
    'delta_ci': round(ci_ft-ci_baseline,4),
    'changed': int(changed), 'fixed': int(fixed), 'broken': int(broken),
}
with open(f'{RESULTS_DIR}/unsloth_results.json','w') as fh: json.dump(results, fh, indent=2)
shutil.copy(f'{PREDS_DIR}/baseline_dev.csv', f'{RESULTS_DIR}/unsloth_baseline_dev.csv')
shutil.copy(f'{PREDS_DIR}/ft_dev.csv',       f'{RESULTS_DIR}/unsloth_ft_dev.csv')
print(json.dumps(results, indent=2))
print('\\nDOWNLOAD for the paper ->', RESULTS_DIR)""")

nb['cells'] = cells
nb['metadata'] = {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                  'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qlora-3b-unsloth-colab.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Wrote', out, 'with', len(cells), 'cells')
