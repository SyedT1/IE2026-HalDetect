"""Builds qlora-3b-baseline-colab.ipynb — baseline-only (no fine-tune).
Run with repo .venv:
   .venv/Scripts/python.exe Development/finetune-qlora-q3b/build_notebook_baseline.py
Computes the zero-shot baseline CI on 500 dev, so the two training notebooks can
skip it (run on a 3rd account). MUST match the training notebooks' config:
same model, MAX_PIXELS=768, prompt, parser -> the CI is directly comparable.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Baseline-only · Qwen2.5-VL-3B · Task 1b

Computes the **no-fine-tune baseline CI** on 500 dev. Run this on its own account;
paste the printed CI into the two training notebooks' Compare cell (`BASELINE_CI`).

Config is locked to match the training notebooks (same model, MAX_PIXELS=768,
prompt, parser) so the number is directly comparable. Fast (12-token) + resumable.""")

md("## 1. GPU check + install")
code("""import torch
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU'
print('GPU:', torch.cuda.get_device_name(0))
!pip install -q -U "transformers>=4.49.0" accelerate bitsandbytes qwen-vl-utils
print('Install done.')""")

md("## 2. Mount Drive + paths")
code("""import os
from google.colab import drive
drive.mount('/content/drive')
PROJ = '/content/drive/MyDrive/ie2026_haldetect'
PREDS_DIR = f'{PROJ}/preds_baseline'
RESULTS_DIR = f'{PROJ}/results'
HF_CACHE = f'{PROJ}/hf_cache'
for d in (PREDS_DIR, RESULTS_DIR, HF_CACHE):
    os.makedirs(d, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
print('Project:', PROJ)""")

md("""## 3. Download dev split only (500) + images""")
code("""import json
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
REPO = 'QCRI/AynVQA-ArabicNLP26'
N_DEV = 500

p = hf_hub_download(REPO, filename='task1b/dev_en.jsonl', repo_type='dataset')
dev = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()][:N_DEV]
print('dev items:', len(dev))
need = sorted({r['image'] for r in dev})
img_path = {}
for rel in tqdm(need, desc='images'):
    img_path[rel] = hf_hub_download(REPO, filename=rel, repo_type='dataset')
print('images ready:', len(img_path))""")

md("""## 4. Load Qwen2.5-VL-3B in 4-bit (MAX_PIXELS=768, matches training notebooks)""")
code("""import torch
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                          BitsAndBytesConfig)
MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
MAX_PIXELS = 768 * 28 * 28
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map='auto', quantization_config=bnb).eval()
print('Model loaded.')""")

md("""## 5. Helpers (identical to training notebooks)""")
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
def generate(messages, max_new_tokens=12):
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

md("""## 6. ⭐ Baseline eval + save""")
code("""import shutil, json
ci_baseline = evaluate(dev, img_path, f'{PREDS_DIR}/baseline_dev.csv')
shutil.copy(f'{PREDS_DIR}/baseline_dev.csv', f'{RESULTS_DIR}/baseline_dev.csv')
json.dump({'ci_baseline': round(ci_baseline,4), 'acc_baseline': round(1-ci_baseline,4),
           'n_dev': len(dev), 'max_pixels': 768*28*28},
          open(f'{RESULTS_DIR}/baseline_results.json','w'), indent=2)
print('\\n' + '='*40)
print('BASELINE CI = %.4f   Acc = %.4f' % (ci_baseline, 1-ci_baseline))
print('='*40)
print('\\n>>> PASTE this CI as BASELINE_CI in BOTH training notebooks <<<')
print('Saved:', f'{RESULTS_DIR}/baseline_dev.csv  +  baseline_results.json')""")

nb['cells'] = cells
nb['metadata'] = {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                  'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qlora-3b-baseline-colab.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Wrote', out, 'with', len(cells), 'cells')
