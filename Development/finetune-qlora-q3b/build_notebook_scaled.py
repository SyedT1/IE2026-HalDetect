"""Builds qlora-3b-scaled-colab.ipynb (Option A: full-data SFT).
Run with repo .venv:
   .venv/Scripts/python.exe Development/finetune-qlora-q3b/build_notebook_scaled.py
Differences vs fast-test notebook:
 - N_TRAIN=3000 (full), N_DEV=500
 - MAX_PIXELS=768*28*28 (balance accuracy vs T4 time)
 - finer checkpointing (save_steps=200) for free-tier resume
 - paper results-bundle cell
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# QLoRA-SFT (Option A: full data) · Qwen2.5-VL-3B · Task 1b

Scaled-up run after the fast test (T1a) came back net-zero. Give SFT a **fair shot**:
**3,000 train** items, evaluate on the **full 500 dev**, higher resolution.

**Metric:** CI = error rate of picking the True statement (lower = better).

**Resumable:** all heavy outputs go to Google Drive and are skipped if present.
Training checkpoints every 200 steps → if Colab disconnects, just re-run cells
top-to-bottom and it continues from the last checkpoint.

⏱️ Expect a few hours total on a free T4. The Drive checkpointing is what makes
this survivable across disconnects.""")

md("## 1. GPU check + install")
code("""import torch
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU'
print('GPU:', torch.cuda.get_device_name(0))
!pip install -q -U "transformers>=4.49.0" accelerate peft bitsandbytes qwen-vl-utils datasets
print('Install done. If asked to "restart session", restart then re-run from cell 2.')""")

md("""## 2. Mount Drive + project paths
Same project folder as the fast test, so cached data/model are reused.""")
code("""import os
from google.colab import drive
drive.mount('/content/drive')

PROJ = '/content/drive/MyDrive/ie2026_haldetect'
DATA_DIR, PREDS_DIR = f'{PROJ}/data', f'{PROJ}/preds_scaled'
CKPT_DIR, ADAPTER_DIR = f'{PROJ}/ckpt_scaled', f'{PROJ}/adapter_scaled'
RESULTS_DIR, HF_CACHE = f'{PROJ}/results', f'{PROJ}/hf_cache'
for d in (DATA_DIR, PREDS_DIR, CKPT_DIR, ADAPTER_DIR, RESULTS_DIR, HF_CACHE):
    os.makedirs(d, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
print('Project:', PROJ, '| scaled run uses *_scaled folders (fast-test files untouched)')""")

md("""## 3. Download data (cached)
Full 3,000 train + 500 dev. Set `N_TRAIN=2000` if you want a faster first pass.""")
code("""import json
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

REPO = 'QCRI/AynVQA-ArabicNLP26'
N_TRAIN, N_DEV = 1500, 500     # test run. Raise to 3000 (full) once this works end-to-end.

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
assert all('labels' in r for r in train + dev), 'missing labels'""")

md("""## 4. Load Qwen2.5-VL-3B in 4-bit""")
code("""import torch
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                          BitsAndBytesConfig)

MODEL_ID   = 'Qwen/Qwen2.5-VL-3B-Instruct'
MAX_PIXELS = 768 * 28 * 28    # higher than fast test (512); raise to 1024*28*28 if time allows

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map='auto', quantization_config=bnb)
print('Model loaded in 4-bit. MAX_PIXELS =', MAX_PIXELS)""")

md("""## 5. Helpers: prompt, parser, CI scorer (same as fast test)""")
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
def generate(messages, max_new_tokens=12):    # only the "Answer: X" line -> ~5x faster eval
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(messages)
    inp = processor(text=[text], images=imgs, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

def evaluate(items, img_path, csv_out):
    """Per-item save + skip already-done ids -> fast and disconnect-resumable."""
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
        f.flush()    # save each item immediately
    f.close()
    rows = list(csv.DictReader(open(csv_out)))
    ci = 1 - sum(int(x['correct']) for x in rows) / len(rows)
    print(f'{csv_out} -> CI={ci:.4f} ({len(rows)} items)')
    return ci

print('Helpers ready (fast 12-token + resumable eval).')''')

md("""## 6. Baseline — run separately (not here)
To save GPU, the no-fine-tune baseline is computed once in
**qlora-3b-baseline-colab.ipynb** (on a different account). Paste its CI into the
Compare cell below as `BASELINE_CI`. No baseline GPU run happens in this notebook.""")

md("""## 7. Build the training set (3,000 items)""")
code("""def to_train_example(r):
    gold = r['labels'].index(True)
    msgs = build_user_messages(img_path[r['image']], r['statements'])
    msgs.append({'role': 'assistant', 'content': [{'type': 'text', 'text': f'Answer: {gold + 1}'}]})
    return {'messages': msgs}

train_examples = [to_train_example(r) for r in train]
print(f'Built {len(train_examples)} training examples.')""")

md("""## 8. ⭐ QLoRA fine-tune (resumable, checkpoint every 200 steps)""")
code('''import glob
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
    labels[labels == IMAGE_TOKEN_ID] = -100
    batch['labels'] = labels
    return batch

model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none',
                  task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj'])
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
    logging_steps=10,
    save_strategy='steps',
    save_steps=200,            # finer checkpoints -> less lost on free-tier disconnect
    save_total_limit=3,
    optim='paged_adamw_8bit',
    remove_unused_columns=False,
    report_to='none',
)
trainer = Trainer(model=model, args=args, train_dataset=train_examples, data_collator=collate_fn)
resume = bool(glob.glob(f'{CKPT_DIR}/checkpoint-*'))
print('Resuming from checkpoint.' if resume else 'Starting fresh training.')
trainer.train(resume_from_checkpoint=resume)
model.save_pretrained(ADAPTER_DIR)
print('Adapter saved to', ADAPTER_DIR)''')

md("""## 9. ⭐ Fine-tuned eval — full 500 dev""")
code("""model.config.use_cache = True
model.gradient_checkpointing_disable()
model.eval()
ci_ft = evaluate(dev, img_path, f'{PREDS_DIR}/ft_dev.csv')
print(f'\\nFINE-TUNED CI = {ci_ft:.4f}  (accuracy = {1-ci_ft:.4f})')""")

md("## 10. Compare")
code("""# Baseline is computed in the separate baseline-only notebook. Paste its CI here:
BASELINE_CI = None        # e.g. 0.0800

print('Fine-tuned   CI = %.4f   Acc = %.4f' % (ci_ft, 1 - ci_ft))
if BASELINE_CI is not None:
    d = ci_ft - BASELINE_CI
    print('Baseline     CI = %.4f   Acc = %.4f' % (BASELINE_CI, 1 - BASELINE_CI))
    print('Delta CI: %+.4f  ->' % d,
          'BETTER (fine-tune helped)' if d < 0 else 'no improvement')
else:
    print('Set BASELINE_CI above (from the baseline notebook) to compute the delta.')""")

md("""## 11. Save paper results bundle
Writes a tidy `results/` folder on Drive: metrics JSON + both prediction CSVs.
**Download this folder for the paper.** (The adapter is large — keep it on Drive.)""")
code("""import json, shutil
results = {
    'experiment': 'T1 scaled QLoRA-SFT (Qwen2.5-VL-3B)',
    'model': MODEL_ID, 'n_train': len(train), 'n_dev': len(dev),
    'max_pixels': MAX_PIXELS, 'epochs': 3, 'lr': 2e-4,
    'lora': {'r': 8, 'alpha': 16, 'targets': ['q_proj','k_proj','v_proj','o_proj']},
    'ci_finetuned': round(ci_ft, 4), 'acc_finetuned': round(1-ci_ft, 4),
    'baseline_ci': BASELINE_CI,
    'delta_ci': (round(ci_ft - BASELINE_CI, 4) if BASELINE_CI is not None else None),
}
# optional: if you uploaded the baseline notebook's baseline_dev.csv to this Drive,
# compute the fixed/broken breakdown too.
bcsv = f'{PREDS_DIR}/baseline_dev.csv'
if os.path.exists(bcsv):
    b = {r['id']: r for r in csv.DictReader(open(bcsv))}
    f = {r['id']: r for r in csv.DictReader(open(f'{PREDS_DIR}/ft_dev.csv'))}
    ids = [i for i in b if i in f]
    results['changed'] = int(sum(b[i]['pred'] != f[i]['pred'] for i in ids))
    results['fixed']   = int(sum(b[i]['correct']=='0' and f[i]['correct']=='1' for i in ids))
    results['broken']  = int(sum(b[i]['correct']=='1' and f[i]['correct']=='0' for i in ids))
with open(f'{RESULTS_DIR}/T1_scaled_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
shutil.copy(f'{PREDS_DIR}/ft_dev.csv', f'{RESULTS_DIR}/T1_scaled_ft_dev.csv')
print(json.dumps(results, indent=2))
print('\\nDOWNLOAD for the paper ->', RESULTS_DIR)
print('Adapter stays on Drive:', ADAPTER_DIR)""")

nb['cells'] = cells
nb['metadata'] = {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                  'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qlora-3b-scaled-colab.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Wrote', out, 'with', len(cells), 'cells')
