"""
Generate `dev-eval-4-adapters.ipynb`.

Why this notebook exists
------------------------
The four 7B QLoRA runs (2k / 2.3k / 2.6k / 3k) are each scored by a *single*
Codabench submission on the blind `devtest` split. That gives one number per run
and nothing else -- no per-item record, so no paired test is possible, so there
is no way to tell a real effect from a coin flip. And the differences in question
are 2-3 items out of 500.

This runs all four adapters over the *labelled* `dev` split instead. That buys:

  1. A second, independent measurement of each run. If the devtest ordering
     (2.3k < 2k < 2.6k) does not reproduce on dev, the ordering was noise.
  2. Per-item correctness, which makes a paired McNemar test possible. McNemar is
     the right test here: the same 500 images go to every model, so the runs are
     paired, and an unpaired test throws away most of the power.
  3. Wilson intervals, so "CI 0.028" is reported as the range it actually is.

Everything else is held identical to the original inference notebooks: same CoT5
prompt, same MAX_PIXELS, same greedy decoding, same 4-bit NF4 base. Only the
adapter changes between runs, which is what makes this an ablation.

CAVEAT that must survive into the paper: the 3k adapter was *selected* on dev
(early stopping picked step_600 by best dev CI). Its dev score is therefore
optimistically biased and is NOT comparable to the others. The notebook prints it
but flags it. 2k / 2.3k / 2.6k never saw dev during training and are clean.

Build:  python build_dev_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
NB_PATH = HERE / "dev-eval-4-adapters.ipynb"

RAW = "https://raw.githubusercontent.com/SyedT1/IE2026-HalDetect/main/Development/qwen2p5-3b-7b"

# ---------------------------------------------------------------------------

MD_INTRO = """\
# 7B QLoRA — all four adapters on the **labelled dev split**

Re-scores the four 7B QLoRA fine-tunes (**2k / 2.3k / 2.6k / 3k**) on `dev`
(500 labelled items) so their differences can actually be tested.

**Why.** On `devtest` each run has exactly one number and no per-item record.
The gaps under discussion are 2–3 items out of 500. That is not measurable with
one blind number per run. On `dev` we get per-item correctness → a paired
**McNemar** test → a real answer to *"is this difference an effect or a dice roll?"*

**What is held fixed:** CoT5 prompt, `MAX_PIXELS=1024*28*28`, greedy decoding,
`max_new_tokens=256`, 4-bit NF4 base. Only the adapter changes.

**Setup:** GPU **T4 x2**, Internet **ON**. Nothing to upload — the adapters are
pulled straight from the GitHub repo. Then **Save Version → Save & Run All**.

Runtime ≈ 3–4 h (4 adapters × 500 items).

---
⚠️ **The 3k adapter is dev-selected.** Its training used early stopping on this
same dev split (best dev CI → `step_600`). Its dev number is therefore
optimistically biased and is *not* comparable to the other three. The notebook
prints it, flagged. The 2k / 2.3k / 2.6k adapters never saw dev.
"""

CELL_INSTALL = """\
!pip -q install -U "transformers==4.51.3" accelerate peft bitsandbytes qwen-vl-utils
import transformers, peft
print('transformers', transformers.__version__, '| peft', peft.__version__)
"""

CELL_CONFIG = f"""\
import os, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

REPO_ID        = 'QCRI/AynVQA-ArabicNLP26'
TASK           = 'task1b'
LANG           = 'en'
SPLIT          = 'dev'            # labelled -- this is the whole point
VLM_MODEL      = 'Qwen/Qwen2.5-VL-7B-Instruct'
MAX_PIXELS     = 1024 * 28 * 28   # same as every published inference run
MAX_NEW_TOKENS = 256
QUANTIZE       = True

RAW = '{RAW}'

# name -> (folder in repo, items the run ACTUALLY trained on, dev-contaminated?)
#
# "items actually trained on" = optimizer_steps * grad_accum(4), NOT the folder
# name. Two of these runs never finished an epoch:
#   2.6k ran 510 of the 650 steps a full epoch needs  -> saw 2,040 items
#   3k   early-stopped and shipped step_600 of 750    -> saw 2,400 items
ADAPTERS = [
    ('FT-2k',   'qlora-q7b-2k-image',   2000, False),
    ('FT-2.3k', 'qlora-q7b-2p3k-image', 2348, False),
    ('FT-2.6k', 'qwen-q7b-2p6k-image',  2040, False),
    ('FT-3k',   'qlora-q7b-3k-image',   2400, True),   # <-- selected ON dev
]

# Codabench devtest CI, for reference only. We are re-measuring on dev.
DEVTEST_CI = {{'FT-2k': 0.032, 'FT-2.3k': 0.028, 'FT-2.6k': 0.034, 'FT-3k': None}}

assert torch.cuda.is_available(), 'No GPU -- set Accelerator to T4 x2'
N_GPUS = torch.cuda.device_count()
print(f'GPUs: {{N_GPUS}}')
for i in range(N_GPUS):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {{i}}: {{p.name}} ({{p.total_memory/1024**3:.1f}} GB)')
"""

CELL_FETCH = """\
import zipfile, urllib.request, shutil

# Pull each adapter_final.zip straight from the public GitHub repo. They are
# plain git blobs (~77 MB, no LFS), so the raw URL serves the real file.
ADAPTER_DIRS = {}
for name, folder, _, _ in ADAPTERS:
    dst = f'/kaggle/working/adapters/{name}'
    if os.path.exists(os.path.join(dst, 'adapter_config.json')):
        ADAPTER_DIRS[name] = dst
        print(f'{name:<9} cached')
        continue
    os.makedirs(dst, exist_ok=True)
    url = f'{RAW}/{folder}/adapter_final.zip'
    tmp = f'/kaggle/working/{name}.zip'
    urllib.request.urlretrieve(url, tmp)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(dst)
    os.remove(tmp)

    # Some zips nest everything one level down; flatten if so.
    if not os.path.exists(os.path.join(dst, 'adapter_config.json')):
        subs = [d for d in os.listdir(dst)
                if os.path.isdir(os.path.join(dst, d))]
        for s in subs:
            inner = os.path.join(dst, s)
            if os.path.exists(os.path.join(inner, 'adapter_config.json')):
                for f in os.listdir(inner):
                    shutil.move(os.path.join(inner, f), os.path.join(dst, f))
                break

    assert os.path.exists(os.path.join(dst, 'adapter_config.json')), \\
        f'{name}: adapter_config.json not found after extract'
    ADAPTER_DIRS[name] = dst
    sz = sum(os.path.getsize(os.path.join(dst, f))
             for f in os.listdir(dst)) / 1024**2
    print(f'{name:<9} ok  ({sz:.0f} MB)')

print('\\nAll adapters ready.')
"""

CELL_DATA = """\
import json
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

jsonl = hf_hub_download(
    REPO_ID, filename=f'{TASK}/{SPLIT}_{LANG}.jsonl', repo_type='dataset')
records = [json.loads(l) for l in open(jsonl, encoding='utf-8') if l.strip()]
assert 'labels' in records[0], f'{SPLIT} has no labels -- this notebook needs them'
print(f'{SPLIT}: {len(records)} labelled records')

needed, paths = sorted({r['image'] for r in records}), {}
for rel in tqdm(needed, desc='images'):
    try:
        paths[rel] = hf_hub_download(REPO_ID, filename=rel, repo_type='dataset')
    except Exception as e:
        print(f'Failed: {rel}: {e}')
print(f'Downloaded {len(paths)}/{len(needed)} images.')

records = [r for r in records if r['image'] in paths]
GOLD = {r['id']: r['labels'].index(True) for r in records}
print(f'Usable items: {len(records)}')
"""

CELL_MODEL = """\
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
) if QUANTIZE else None

max_mem = {i: '13500MiB' for i in range(N_GPUS)}
max_mem['cpu'] = '4GiB'

print('Loading base model (once -- adapters are hot-swapped on top)...')
processor  = AutoProcessor.from_pretrained(VLM_MODEL, max_pixels=MAX_PIXELS)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    VLM_MODEL, torch_dtype=dtype,
    device_map='auto', max_memory=max_mem,
    quantization_config=bnb_config)

# Load adapter #1 to create the PeftModel, then attach the rest as named
# adapters on the SAME base. Loading the 7B base four times would waste ~30 min
# and risk OOM; swapping adapters is instant and keeps the base bit-identical
# across runs, which is exactly the control we want.
names = [n for n, _, _, _ in ADAPTERS]
model = PeftModel.from_pretrained(
    base_model, ADAPTER_DIRS[names[0]], adapter_name=names[0])
for n in names[1:]:
    model.load_adapter(ADAPTER_DIRS[n], adapter_name=n)
model.eval()
print('Adapters loaded:', list(model.peft_config.keys()))

for i in range(N_GPUS):
    a = torch.cuda.memory_allocated(i) / 1024**3
    t = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {a:.1f}/{t:.1f} GB')
"""

CELL_PROMPT = """\
import re

# CoT5 attribute-checklist prompt -- character-for-character the prompt every
# published 7B QLoRA inference run used. Held fixed across all four adapters.
PROMPT = (
    'You are a visual fact-checker examining an image from the Arab world.\\n'
    'Below are THREE statements about this image. '
    'Exactly ONE statement is grounded in the image (True). '
    'The other two are plausible-sounding hallucinations (False).\\n\\n'
    'Statement 1: {s0}\\n'
    'Statement 2: {s1}\\n'
    'Statement 3: {s2}\\n\\n'
    'Instructions:\\n'
    '- On the VERY FIRST line write ONLY: "Answer: X" where X is 1, 2, or 3.\\n'
    '- For each statement evaluate:\\n'
    '    (a) Colour/texture evidence for or against\\n'
    '    (b) Shape/form evidence for or against\\n'
    '    (c) Contextual evidence for or against\\n'
    '- Then state your conclusion.\\n'
    'Do not write anything before the Answer line.'
)


@torch.no_grad()
def vlm_call(image_path, text):
    conv = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image_path},
        {'type': 'text',  'text':  text}]}]
    prompt_str = processor.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conv)
    inputs = processor(
        text=[prompt_str], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors='pt').to(model.device)
    gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    out = processor.decode(
        gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    del inputs, gen
    torch.cuda.empty_cache()
    return out


def parse_answer(raw):
    \"\"\"Answer-first: trust line 1, then fall back to scanning the whole reply.\"\"\"
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for line in lines[:1] + lines:
        m = re.search(r'answer\\s*[:\\-]?\\s*([123])', line, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None
"""

CELL_RUN = """\
import csv, time

ALL = {}          # adapter name -> {id: predicted index}
RAWS = {}         # adapter name -> {id: raw reply}

for name, folder, seen, contaminated in ADAPTERS:
    model.set_adapter(name)          # hot-swap; base weights untouched
    t0 = time.time()
    preds, raws, unparsed = {}, {}, 0

    for r in tqdm(records, desc=name):
        raw = vlm_call(paths[r['image']], PROMPT.format(
            s0=r['statements'][0], s1=r['statements'][1], s2=r['statements'][2]))
        chosen = parse_answer(raw)
        if chosen is None:
            unparsed += 1
            chosen = 1               # arbitrary; counted and reported below
        preds[r['id']] = chosen - 1
        raws[r['id']]  = raw

    ALL[name], RAWS[name] = preds, raws
    n_ok = sum(preds[i] == GOLD[i] for i in preds)
    ci   = 1 - n_ok / len(preds)
    mins = (time.time() - t0) / 60
    flag = '  [DEV-SELECTED -- biased, see note]' if contaminated else ''
    print(f'{name:<9} dev CI={ci:.4f}  ({len(preds)-n_ok}/{len(preds)} wrong)  '
          f'unparsed={unparsed}  {mins:.0f} min{flag}')

    # Per-item record. This is the artifact that makes McNemar possible; the
    # Codabench submissions do not contain it.
    with open(f'dev_predictions_{name}.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'gold_idx', 'pred_idx', 'correct', 'raw'])
        for i in preds:
            w.writerow([i, GOLD[i], preds[i],
                        int(preds[i] == GOLD[i]), raws[i]])

print('\\nAll adapters scored.')
"""

CELL_STATS = """\
import math, itertools
from scipy.stats import binomtest

N = len(records)


def wilson(k, n, z=1.96):
    \"\"\"95% Wilson interval for an error rate. Normal approximation breaks down
    at p~0.03 with n=500; Wilson does not.\"\"\"
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


print(f'DEV SPLIT, n={N}.  CI == error rate, so 1 item = {1/N:.4f} CI.\\n')
print(f'{"run":<9} {"seen":>6} {"wrong":>6} {"dev CI":>8} '
      f'{"95% CI (Wilson)":>22} {"devtest CI":>11}')
print('-' * 70)
for name, _, seen, contaminated in ADAPTERS:
    wrong = sum(ALL[name][i] != GOLD[i] for i in ALL[name])
    lo, hi = wilson(wrong, N)
    dt = DEVTEST_CI[name]
    dts = f'{dt:.3f}' if dt is not None else 'unsubmitted'
    flag = '  <- dev-selected, biased' if contaminated else ''
    print(f'{name:<9} {seen:>6} {wrong:>6} {wrong/N:>8.4f} '
          f'{f"[{lo:.4f}, {hi:.4f}]":>22} {dts:>11}{flag}')

print('\\nIf these intervals overlap, the runs are not distinguishable at n=500.\\n')

# ---- paired McNemar -------------------------------------------------------
# The same 500 images go to every model, so the runs are PAIRED. McNemar looks
# only at the items where the two models disagree: b = A right / B wrong,
# c = A wrong / B right. Items both get right (or both wrong) carry no
# information about which is better and are correctly ignored.
print('PAIRED McNEMAR (exact, two-sided)\\n')
print(f'{"comparison":<24} {"A>B":>4} {"B>A":>4} {"p":>8}   verdict')
print('-' * 66)
for (a, _, _, _), (b, _, _, _) in itertools.combinations(ADAPTERS, 2):
    ca = {i: ALL[a][i] == GOLD[i] for i in GOLD}
    cb = {i: ALL[b][i] == GOLD[i] for i in GOLD}
    only_a = sum(ca[i] and not cb[i] for i in GOLD)   # A right, B wrong
    only_b = sum(cb[i] and not ca[i] for i in GOLD)   # B right, A wrong
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    verdict = ('SIGNIFICANT' if p < 0.05
               else 'not significant -- consistent with noise')
    print(f'{a+" vs "+b:<24} {only_a:>4} {only_b:>4} {p:>8.3f}   {verdict}')

print('\\n' + '=' * 66)
print('READ THIS:')
print('  A "not significant" row means the two runs are indistinguishable on')
print('  500 items. It does NOT prove they are equal -- it proves the data')
print('  cannot tell them apart, so no ordering between them may be claimed.')
print('  Report those runs as a tier, not a ranking.')
print('=' * 66)
"""

CELL_PACK = """\
import zipfile, glob

out = 'dev_eval_4_adapters.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob('dev_predictions_*.csv')):
        z.write(f)
print(f'Wrote {out}')
print()
print('Download from the Output panel and commit into:')
print('  Development/qwen2p5-3b-7b/dev-eval-adapters/')
print()
print('Then run  python significance.py  locally to regenerate the tables.')
"""

CELLS = [
    ("md", MD_INTRO),
    ("md", "## 1. Install"),
    ("code", CELL_INSTALL),
    ("md", "## 2. Config\n\nNote `seen` — the number of items each run *actually* "
           "trained on, which is not what its folder name says."),
    ("code", CELL_CONFIG),
    ("md", "## 3. Fetch the four adapters from GitHub\n\nNo Kaggle Dataset upload "
           "needed — they are public blobs in the repo."),
    ("code", CELL_FETCH),
    ("md", "## 4. Labelled dev split"),
    ("code", CELL_DATA),
    ("md", "## 5. Base model + all four adapters\n\nBase is loaded **once** and the "
           "adapters are hot-swapped on top, so the base is bit-identical across "
           "runs. That is the control."),
    ("code", CELL_MODEL),
    ("md", "## 6. CoT5 prompt (identical to every published FT inference run)"),
    ("code", CELL_PROMPT),
    ("md", "## 7. Score all four on dev\n\n≈45 min per adapter."),
    ("code", CELL_RUN),
    ("md", "## 8. Wilson intervals + paired McNemar\n\nThe part that decides whether "
           "the data-volume 'curve' is real."),
    ("code", CELL_STATS),
    ("md", "## 9. Package outputs"),
    ("code", CELL_PACK),
]


def build() -> None:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "code":
            # Refuse to emit a notebook with a syntax error. Jupyter line magics
            # (!pip, %cd) are not Python, so they are skipped rather than compiled.
            if not any(l.lstrip().startswith(("!", "%")) for l in lines):
                compile(src, "<cell>", "exec")
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": lines})
        else:
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": lines})

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"wrote {NB_PATH}  ({len(cells)} cells, {n_code} code cells compiled OK)")


if __name__ == "__main__":
    build()
