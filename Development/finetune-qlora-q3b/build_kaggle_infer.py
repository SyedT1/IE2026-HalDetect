"""Builds kaggle-infer-q3b.ipynb — load a trained adapter (SFT or DPO) and score
CI on 500 dev. Mirrors the teammate step3 inference notebook. MAX_PIXELS=768 to
compare fairly against our baseline (CI 0.096, same 768px).
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Inference / score a trained adapter (Kaggle) — 3B

Loads the base 3B + a LoRA adapter (SFT-final or DPO-final) and computes CI on the
full 500 dev, so you can compare against the baseline (CI 0.096 @768px).

**Before running:** upload the adapter folder (from `adapter_*_final.zip`) as a
Kaggle Dataset and add it as Input. Accelerator T4 (x2 optional). Secret `HF_TOKEN`.

Output: `/kaggle/working/infer_results.json` + `dev_preds.csv` (download for the paper).""")

md("## 1. Install")
code("""import os, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
!pip install -q -U "transformers>=4.49.0" "peft>=0.10.0" accelerate bitsandbytes qwen-vl-utils
print('GPUs:', torch.cuda.device_count())""")

md("## 2. Config + HF login + find adapter")
code("""import os
from huggingface_hub import login
try:
    from kaggle_secrets import UserSecretsClient
    login(token=UserSecretsClient().get_secret('HF_TOKEN'))
except Exception as e:
    print('No HF secret (public data still works):', e)

REPO_ID   = 'QCRI/AynVQA-ArabicNLP26'
VLM_MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
MAX_PIXELS = 768 * 28 * 28     # match baseline for a fair CI comparison
N_DEV     = 500
BASELINE_CI = 0.096            # our matched baseline

ADAPTER = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'adapter_config.json' in files and 'adapter_model.safetensors' in files:
        ADAPTER = root; break
assert ADAPTER, 'adapter not found under /kaggle/input — add your dataset.'
print('Adapter:', ADAPTER)""")

md("## 3. Download dev + images")
code("""import json
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
jsonl = hf_hub_download(REPO_ID, filename='task1b/dev_en.jsonl', repo_type='dataset')
dev = [json.loads(l) for l in open(jsonl, encoding='utf-8') if l.strip()][:N_DEV]
need = sorted({r['image'] for r in dev})
img_path = {}
for rel in tqdm(need, desc='images'):
    img_path[rel] = hf_hub_download(REPO_ID, filename=rel, repo_type='dataset')
print('dev:', len(dev), 'images:', len(img_path))""")

md("## 4. Load base + adapter")
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
model = PeftModel.from_pretrained(base, ADAPTER).eval()
print('Loaded adapter.')""")

md("## 5. Score CI (fast 12-token, resumable)")
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

def parse_answer(raw):
    for ln in [l.strip() for l in raw.splitlines() if l.strip()]:
        m = re.search(r'answer\\s*[:\\-]?\\s*([123])', ln, re.IGNORECASE)
        if m: return int(m.group(1)) - 1
    return None

@torch.no_grad()
def gen(r):
    msgs = [{'role':'user','content':[{'type':'image','image':img_path[r['image']]},
            {'type':'text','text':JOINT_PROMPT.format(
                s0=r['statements'][0], s1=r['statements'][1], s2=r['statements'][2])}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs,_ = process_vision_info(msgs)
    inp = processor(text=[text], images=imgs, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(**inp, max_new_tokens=12, do_sample=False)
    return processor.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

CSV = '/kaggle/working/dev_preds.csv'
done = {r['id'] for r in csv.DictReader(open(CSV))} if os.path.exists(CSV) else set()
todo = [r for r in dev if r['id'] not in done]
f = open(CSV, 'a', newline=''); w = csv.DictWriter(f, fieldnames=['id','gold','pred','correct'])
if not done: w.writeheader(); f.flush()
from tqdm.auto import tqdm
for r in tqdm(todo, desc='eval'):
    gold = r['labels'].index(True); pred = parse_answer(gen(r))
    if pred is None: pred = 0
    w.writerow({'id':r['id'],'gold':gold,'pred':pred,'correct':int(pred==gold)}); f.flush()
f.close()
rows = list(csv.DictReader(open(CSV)))
ci = 1 - sum(int(x['correct']) for x in rows)/len(rows)
print(f'CI = {ci:.4f}  Acc = {1-ci:.4f}  ({len(rows)} items)')''')

md("## 6. Compare vs baseline + save results")
code("""import json
delta = ci - BASELINE_CI
print('Baseline CI %.4f -> adapter CI %.4f  (delta %+.4f)' % (BASELINE_CI, ci, delta))
print('BETTER' if delta < 0 else 'no improvement')
json.dump({'adapter': ADAPTER, 'ci': round(ci,4), 'acc': round(1-ci,4),
           'baseline_ci': BASELINE_CI, 'delta_ci': round(delta,4), 'n_dev': len(dev),
           'max_pixels': MAX_PIXELS},
          open('/kaggle/working/infer_results.json','w'), indent=2)
print('Saved /kaggle/working/infer_results.json + dev_preds.csv -> download for the paper.')""")

nb['cells'] = cells
nb['metadata'] = {'accelerator': 'GPU', 'colab': {'provenance': []},
                  'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kaggle-infer-q3b.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Wrote', out, 'with', len(cells), 'cells')
