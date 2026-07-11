"""
Generate the InternVL notebooks for ImageEval 2026 Task 1b.

These mirror the Qwen2.5-VL notebooks in ../qwen2p5-3b-7b one-for-one so the two
model families are directly comparable in the paper. Everything that could move a
number -- the official `evaluate_tf` parser, the prompts, the scoring block, the
submission zip layout -- is held byte-identical to the Qwen runs. Only the model
loading and image preprocessing change, because InternVL uses `model.chat()` with
448px dynamic tiling instead of Qwen's processor + `max_pixels`.

The parser is not retyped here: it is lifted verbatim out of the Qwen baseline
notebook at build time, so it cannot drift.

Usage:
    python build_internvl_notebooks.py            # build the 2 smoke-test notebooks
    python build_internvl_notebooks.py --all      # build all 11
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
QWEN = HERE.parent / "qwen2p5-3b-7b"

# Both models are from the InternVL2 generation, deliberately. InternVL3.5-8B was the
# other candidate for the large slot, but it postdates Qwen2.5-VL by a wide margin: a win
# there could not be attributed to model family rather than to a more recent training run.
# InternVL2 is Qwen2.5-VL's contemporary, so the comparison is clean.
MODEL_2B = "OpenGVLab/InternVL2-2B"
MODEL_8B = "OpenGVLab/InternVL2-8B"


# ── notebook plumbing ────────────────────────────────────────────────────────

def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def steal_parser_cell() -> str:
    """Lift `evaluate_tf` verbatim from the Qwen baseline notebook.

    Copying rather than retyping guarantees the InternVL runs are scored by exactly
    the same parser as the Qwen runs -- and as the Codabench scorer.
    """
    nb = json.loads((QWEN / "baseline" / "qwen2p5vl-baseline.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and "def evaluate_tf(" in src:
            return src
    raise SystemExit("could not find evaluate_tf in the Qwen baseline notebook")


PARSER_SRC = steal_parser_cell()


# ── shared cells ─────────────────────────────────────────────────────────────

CELL_INSTALL = """
# ── Kaggle setup ──────────────────────────────────────────────────────────────
# transformers is PINNED to 4.49.0, and the pin is load-bearing. Both InternVL2 models
# vendor their own InternLM2ForCausalLM via trust_remote_code, and that class never
# declared GenerationMixin. From v4.50 PreTrainedModel stopped inheriting the mixin, so
# on any newer transformers the language model simply has no .generate() and model.chat()
# dies with:
#     AttributeError: 'InternLM2ForCausalLM' object has no attribute 'generate'
# It fails at the first inference call — i.e. AFTER the model has loaded cleanly.
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

!pip install -q "transformers==__PIN__" accelerate bitsandbytes timm einops \\
               sentencepiece "huggingface_hub" 2>&1 | tail -3
import transformers
print(f"Dependencies installed. transformers {transformers.__version__}")
"""

# Both models are InternVL2-generation, so both sit below the v4.50 boundary.
TRANSFORMERS_PIN = {MODEL_2B: "4.49.0", MODEL_8B: "4.49.0"}


def cell_install(model: str) -> str:
    return CELL_INSTALL.replace("__PIN__", TRANSFORMERS_PIN[model])

CELL_LOGIN = """
# The AynVQA dataset is public, so anonymous download works and this cell is optional.
# If you hit a rate limit, add a HF READ token as a Kaggle Secret named HF_TOKEN.
# Never hardcode a token in the notebook.
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    login(token=UserSecretsClient().get_secret("HF_TOKEN"))
    print("Logged in to Hugging Face via Kaggle Secret.")
except Exception as e:
    print(f"No HF_TOKEN secret ({type(e).__name__}) — continuing anonymously (dataset is public).")
"""

CELL_DOWNLOAD = """
import json
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

jsonl = hf_hub_download(REPO_ID, filename=f"{TASK}/{SPLIT}_{LANG}.jsonl", repo_type="dataset")
records = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
if MAX_ITEMS:
    records = records[:MAX_ITEMS]
print(len(records), "items;  labelled:", "labels" in records[0])

# Sequential download — no ThreadPoolExecutor (avoids Kaggle kernel crashes)
needed = sorted({r["image"] for r in records})
paths = {}
for rel in tqdm(needed, desc="images"):
    try:
        paths[rel] = hf_hub_download(REPO_ID, filename=rel, repo_type="dataset")
    except Exception as e:
        print(f"Failed to download {rel}: {e}")
print(f"Downloaded {len(paths)}/{len(needed)} images.")
"""

# The one genuinely new piece of code: InternVL's 448px dynamic tiling + .chat() loader.
CELL_MODEL = """
import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

assert torch.cuda.is_available(), "No GPU — Kaggle: Settings → Accelerator → GPU T4."
print("GPU:", torch.cuda.get_device_name(0))

# T4 (Turing, sm75) has no bf16 and no FlashAttention-2. The InternVL model card
# says bfloat16 + use_flash_attn=True; both of those crash here. fp16 + eager instead.
DTYPE = torch.float16
USE_FLASH_ATTN = False

# ── InternVL image preprocessing: 448px dynamic tiling ────────────────────────
# InternVL has no `max_pixels` knob. It splits the image into up to MAX_TILES
# 448x448 tiles (plus a thumbnail), so MAX_TILES is the resolution lever —
# it is the InternVL analogue of Qwen's MAX_PIXELS.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff, best_ratio = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff, best_ratio = ratio_diff, ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image(image_file, input_size=448, max_num=MAX_TILES):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


# ── Load model ────────────────────────────────────────────────────────────────
# Same NF4 double-quant recipe the Qwen 7B runs used, so quantisation is not a
# confound when comparing the two families. device_map={'': 0} pins everything to
# one GPU: 8B at 4-bit is ~5.5 GB and fits a single T4, so no sharding is needed
# and every tensor lands on the same device as the pixel values.
quant = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True,
) if QUANTIZE else None

load_kwargs = dict(
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    use_flash_attn=USE_FLASH_ATTN,
    trust_remote_code=True,
)
if QUANTIZE:
    model = AutoModel.from_pretrained(
        VLM_MODEL, quantization_config=quant, device_map={"": 0}, **load_kwargs).eval()
else:
    model = AutoModel.from_pretrained(VLM_MODEL, **load_kwargs).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL, trust_remote_code=True, use_fast=False)
DEVICE = torch.device("cuda:0")

# Guard for the v4.50 GenerationMixin split. The pin above should already avoid this,
# but if transformers ever resolves to >= 4.50 for a model whose remote code predates
# it, the language model silently loses .generate() and model.chat() dies deep inside
# the vendored code. Re-attach the mixin rather than fail 40 minutes into a run.
from transformers.generation import GenerationMixin

lm_cls = type(model.language_model)
if not issubclass(lm_cls, GenerationMixin):
    lm_cls.__bases__ = (GenerationMixin,) + lm_cls.__bases__
    print(f"Patched {lm_cls.__name__} to inherit GenerationMixin "
          f"(transformers {__import__('transformers').__version__} >= 4.50).")

print(f"Model loaded: {VLM_MODEL} | dtype={DTYPE} | quantized={QUANTIZE} | MAX_TILES={MAX_TILES}")
"""

CELL_VLM_CALL = """
import re

@torch.no_grad()
def vlm_call(image_path, text, max_new_tokens=MAX_NEW_TOKENS):
    \"\"\"One InternVL forward pass. Returns the decoded string.

    InternVL takes the image as a pixel_values tensor and requires the '<image>'
    placeholder at the top of the question. Greedy decoding (do_sample=False) to
    match the Qwen runs.
    \"\"\"
    pixel_values = load_image(image_path).to(DTYPE).to(DEVICE)
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    out = model.chat(tokenizer, pixel_values, "<image>\\n" + text, generation_config)
    del pixel_values
    torch.cuda.empty_cache()          # release per-item activations -> avoids slow OOM
    return (out or "").strip()
"""

CELL_CSV = """
import csv
OUT_CSV = f"predictions_{RUN_ID}_{LANG}.csv"
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "statement_index", "raw_prediction", "prediction_parsed"])
    w.writerows(rows)
print("Wrote", OUT_CSV, ":", len(rows), "rows")
"""

# Scoring block: identical arithmetic to the Qwen notebooks (CI / combined / CFHR / Q+ / Q-),
# with the matching Qwen run printed alongside so the comparison is on-screen.
CELL_SCORE = """
gold = {r["id"]: r["labels"].index(True) for r in records if "labels" in r}
if gold:
    by_item = {}
    for iid, si, _raw, parsed in rows:
        by_item.setdefault(iid, {})[si] = parsed

    total = q_plus = q_minus = q_minus_total = combined = 0
    n_partial = n_consistent = 0      # Contrastive Instability
    cfhr_num = cfhr_den = 0           # CFHR
    for iid, true_idx in gold.items():
        total += 1
        q_minus_total += 2
        pr = by_item.get(iid, {})
        labels = {i: evaluate_tf(pr.get(i, "")).pred for i in range(3)}
        ok_t = labels.get(true_idx) == "true"
        ok_f = [labels.get(i) == "false" for i in range(3) if i != true_idx]
        if ok_t:
            q_plus += 1
        q_minus += sum(ok_f)
        all_ok = ok_t and all(ok_f)
        any_ok = ok_t or any(ok_f)
        if all_ok:
            combined += 1
        if any_ok:                    # CI = 1 - consistent / partial
            n_partial += 1
            if all_ok:
                n_consistent += 1
        if ok_t:                      # CFHR = P(miss any Q- | Q+ correct)
            cfhr_den += 1
            if not all(ok_f):
                cfhr_num += 1

    ci           = 1 - n_consistent / n_partial if n_partial else 0.0
    cfhr         = cfhr_num / cfhr_den if cfhr_den else 0.0
    combined_acc = combined / total
    q_plus_acc   = q_plus / total
    q_minus_acc  = q_minus / q_minus_total

    print(f"Run: {RUN_ID}   split: {SPLIT}  ({total} items)")
    print()
    print(f"{'Metric':<34} {'InternVL':>10}   {QWEN_REF_NAME:>14}")
    print("-" * 62)
    print(f"{'Contrastive Instability (CI) v':<34} {ci:>10.4f}   {QWEN_REF['CI']:>14.4f}")
    print(f"{'Combined Accuracy ^':<34} {combined_acc:>10.4f}   {QWEN_REF['Comb']:>14.4f}")
    print(f"{'CFHR v':<34} {cfhr:>10.4f}   {QWEN_REF['CFHR']:>14.4f}")
    print(f"{'Q+ Accuracy ^':<34} {q_plus_acc:>10.4f}   {QWEN_REF['Q+']:>14.4f}")
    print(f"{'Q- Accuracy ^':<34} {q_minus_acc:>10.4f}   {QWEN_REF['Q-']:>14.4f}")
    print()
    delta = ci - QWEN_REF["CI"]
    print(f"CI delta vs {QWEN_REF_NAME}: {delta:+.4f}  "
          f"({'InternVL better' if delta < 0 else 'Qwen better' if delta > 0 else 'tie'})")
else:
    print(f"'{SPLIT}' is blind (no labels) — submit the zip to Codabench for the score.")
"""

CELL_ZIP = """
import zipfile
with open("prediction.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "statement_index", "prediction"])
    for iid, si, _raw, parsed in rows:
        w.writerow([iid, si, parsed])
zip_name = f"prediction_{LANG}.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("prediction.csv", "prediction.csv")
print("Wrote", zip_name, " -> submit to Codabench 17051 (task1b English)")
print("Then download it from the Kaggle output and commit it next to this notebook.")
"""


# ── per-run cells ────────────────────────────────────────────────────────────

def cell_config(run_id, model, quantize, max_tiles, max_new_tokens, ref_name, ref, est) -> str:
    return '''
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_ID   = "QCRI/AynVQA-ArabicNLP26"
TASK      = "task1b"
LANG      = "en"
SPLIT     = "devtest"   # "devtest"/"test" -> blind (submit) | "dev" -> labelled (scored locally)
MAX_ITEMS = None        # e.g. 5 for a smoke test; None = whole split

RUN_ID    = "{run_id}"

VLM_MODEL = "{model}"
QUANTIZE  = {quantize}       # NF4 4-bit — same recipe as the Qwen 7B runs

# InternVL's resolution lever: up to MAX_TILES 448x448 tiles + a thumbnail.
MAX_TILES      = {max_tiles}
MAX_NEW_TOKENS = {max_new_tokens}

# The matching Qwen run, printed next to our score in the scoring cell.
QWEN_REF_NAME = "{ref_name}"
QWEN_REF      = {ref!r}

print(f"Run: {{RUN_ID}}")
print(f"config: {{TASK}}_{{LANG}}/{{SPLIT}} | {{VLM_MODEL}} | quantized={{QUANTIZE}}")
print(f"MAX_TILES={{MAX_TILES}} | MAX_NEW_TOKENS={{MAX_NEW_TOKENS}}")
print("Comparing against Qwen {ref_name} (CI {ci:.3f}). Estimated inference: {est}")
'''.format(run_id=run_id, model=model, quantize=quantize, max_tiles=max_tiles,
           max_new_tokens=max_new_tokens, ref_name=ref_name, ref=ref, ci=ref["CI"], est=est)


# Baseline: one True/False judgement per statement. Same prompt as Qwen Run 1.
CELL_BASELINE_PROMPT = '''
PROMPT = (
    "You are checking a statement against an image for visual hallucination. "
    "Look only at what the image actually shows.\\n\\n"
    "Statement: \\"{s}\\"\\n\\n"
    "If the image clearly supports the statement, answer True. If the statement describes "
    "something that is not in the image or is contradicted by it (a hallucination), answer "
    "False. Answer with only one word: True or False."
)

print("PROMPT PREVIEW:")
print("-" * 60)
print(PROMPT.format(s="<statement>"))
print("-" * 60)
'''

CELL_BASELINE_RUN = '''
DEFAULT_LABEL = "false"   # fallback when evaluate_tf cannot read a verdict

rows = []   # (id, statement_index, raw_prediction, prediction_parsed)
n_fallback = 0
for r in tqdm(records, desc=f"[{RUN_ID}] infer"):
    for si, stmt in enumerate(r["statements"]):
        raw = vlm_call(paths[r["image"]], PROMPT.format(s=stmt))
        parsed = evaluate_tf(raw).pred
        n_fallback += (parsed is None)
        rows.append((r["id"], si, raw, parsed or DEFAULT_LABEL))

print(f"done: {len(rows)} judgements  |  unparseable -> defaulted to {DEFAULT_LABEL!r}: {n_fallback}")
'''

# Joint runs: one forward pass per item, model names the grounded statement.
CELL_JOINT_INFER = '''
def parse_joint_answer(raw):
    """Parse 'Answer: X' (X in 1..3). Returns 1-indexed int, or None."""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for line in ANSWER_SCAN_ORDER(lines):
        m = re.search(r"answer\\s*[:\\-]?\\s*([123])", line, re.IGNORECASE)
        if m:
            return int(m.group(1))
    for line in lines:                      # any standalone digit on its own line
        if re.fullmatch(r"[123]", line):
            return int(line)
    return None


FALLBACK_PROMPT = (
    "You are a visual fact-checker.\\n"
    "Decide if the following statement about the image is True or False.\\n"
    "Context: exactly one of three statements about this image is True.\\n\\n"
    "Statement: \\"{s}\\"\\n\\n"
    "Write your answer on the FIRST line as exactly one word: True or False.\\n"
    "Then briefly explain your reasoning."
)


def predict_item(image_path, statements):
    """Joint prediction for one item. Returns (labels, raw, mode, chosen_1indexed)."""
    raw = vlm_call(image_path, PROMPT.format(s0=statements[0], s1=statements[1], s2=statements[2]))
    chosen = parse_joint_answer(raw)
    if chosen is not None:
        labels = ["false", "false", "false"]
        labels[chosen - 1] = "true"
        return labels, raw, "joint", chosen

    # Fallback: judge each statement on its own. Only reached when the model never
    # emitted an 'Answer: X' line — expected to be rare (<5% on the Qwen runs).
    labels, fb_raws = [], [raw]
    for stmt in statements:
        fb = vlm_call(image_path, FALLBACK_PROMPT.format(s=stmt), max_new_tokens=128)
        fb_raws.append(fb)
        first = fb.splitlines()[0] if fb.strip() else ""
        pred = evaluate_tf(first).pred or evaluate_tf(fb).pred or "false"
        labels.append(pred)
    if labels.count("true") != 1:           # enforce exactly-one-True
        labels = ["true", "false", "false"]
    return labels, " ||| ".join(fb_raws), "fallback", labels.index("true") + 1


print("Inference functions defined.")
print("PROMPT PREVIEW:")
print("-" * 60)
print(PROMPT.format(s0="<statement 1>", s1="<statement 2>", s2="<statement 3>"))
print("-" * 60)
'''

CELL_JOINT_RUN = '''
rows = []       # (id, statement_index, raw, prediction)
results = []    # per-item detail, for the error analysis below
n_joint = n_fallback = 0

for r in tqdm(records, desc=f"[{RUN_ID}] joint infer"):
    labels, raw, mode, chosen = predict_item(paths[r["image"]], r["statements"])
    n_joint    += (mode == "joint")
    n_fallback += (mode == "fallback")
    gold_idx = r["labels"].index(True) if "labels" in r else None
    results.append({
        "id": r["id"], "country": r.get("country", "?"), "category": r.get("category", "?"),
        "gold_idx": gold_idx, "pred_idx": chosen - 1,
        "correct": (chosen - 1 == gold_idx) if gold_idx is not None else None,
        "mode": mode,
    })
    for si, lbl in enumerate(labels):
        rows.append((r["id"], si, raw if si == 0 else "", lbl))

print(f"Done: {len(records)} items | joint: {n_joint} | fallback: {n_fallback}")
print(f"Fallback rate: {n_fallback / len(records) * 100:.1f}%  (Qwen runs: <5%)")
'''


# ── run definitions ──────────────────────────────────────────────────────────
# Each InternVL run is pinned to the Qwen run it replicates. `ref` values are the
# devtest numbers from the repo README, so the notebook prints the delta in place.

ANSWER_FIRST_HEAD = '''
# Answer-FIRST joint prompt: the model commits to "Answer: X" on line 1, THEN
# justifies. On Qwen this beat reason-first by a wide margin (Run 2 -> Run 4).
ANSWER_SCAN_ORDER = lambda lines: lines[:1] + lines   # first line first, then the rest
'''

REASON_FIRST_HEAD = '''
# Reason-FIRST joint prompt: the model reasons, THEN writes "Answer: X" on the
# LAST line. Scanning starts from the bottom.
ANSWER_SCAN_ORDER = lambda lines: list(reversed(lines))
'''

JOINT_REASON_FIRST_PROMPT = '''
PROMPT = (
    "You are a visual fact-checker examining an image from the Arab world.\\n"
    "Below are THREE statements about this image. "
    "Exactly ONE statement is grounded in the image (True). "
    "The other two are plausible-sounding hallucinations (False).\\n\\n"
    "Statement 1: {s0}\\n"
    "Statement 2: {s1}\\n"
    "Statement 3: {s2}\\n\\n"
    "Instructions:\\n"
    "- Study the image carefully.\\n"
    "- Reason step by step about each statement.\\n"
    "- On the very last line write ONLY: \\"Answer: X\\" "
    "where X is 1, 2, or 3 (the grounded statement).\\n"
    "Do not write anything after the Answer line."
)
'''

JOINT_ANSWER_FIRST_PROMPT = '''
PROMPT = (
    "You are a visual fact-checker examining an image from the Arab world.\\n"
    "Below are THREE statements about this image. "
    "Exactly ONE statement is grounded in the image (True). "
    "The other two are plausible-sounding hallucinations (False).\\n\\n"
    "Statement 1: {s0}\\n"
    "Statement 2: {s1}\\n"
    "Statement 3: {s2}\\n\\n"
    "Instructions:\\n"
    "- Study the image carefully.\\n"
    "- On the VERY FIRST line write ONLY: \\"Answer: X\\" where X is 1, 2, or 3.\\n"
    "- Then explain step by step why that statement is grounded "
    "and why the other two are hallucinations.\\n"
    "Do not write anything before the Answer line."
)
'''

COT_HEAD = (
    'PROMPT = (\n'
    '    "You are a visual fact-checker examining an image from the Arab world.\\n"\n'
    '    "Below are THREE statements. Exactly ONE is grounded in the image (True). "\n'
    '    "The other two are hallucinations (False).\\n\\n"\n'
    '    "Statement 1: {s0}\\n"\n'
    '    "Statement 2: {s1}\\n"\n'
    '    "Statement 3: {s2}\\n\\n"\n'
    '    "Instructions:\\n"\n'
    '    "- On the VERY FIRST line write ONLY: \\"Answer: X\\" where X is 1, 2, or 3.\\n"\n'
)
COT_TAIL = '    "Do not write anything before the Answer line."\n)\n'

# The six CoT bodies, verbatim from the Qwen CoT notebooks.
COT_BODIES = {
    "evidence-first": (
        '    "- On the second line write ONE sentence describing only what you literally see "\n'
        '    "in the image (objects, materials, colours, actions) — do NOT use the statement text.\\n"\n'
        '    "- Then explain why that statement is grounded and the others are not.\\n"\n'
    ),
    "elimination": (
        '    "- Then work through elimination: which statement can you rule out FIRST "\n'
        '    "and why? Which statement can you rule out SECOND and why? "\n'
        '    "The remaining statement is the grounded one — confirm it.\\n"\n'
    ),
    "confidence-ranked": (
        '    "- Then rank ALL THREE statements by how strongly the image supports them:\\n"\n'
        '    "  Most grounded: statement X — [specific visual evidence]\\n"\n'
        '    "  Less grounded: statement Y — [why the evidence is weak or absent]\\n"\n'
        '    "  Least grounded: statement Z — [why it is contradicted]\\n"\n'
    ),
    "devils-advocate": (
        '    "- Then for each statement you did NOT choose, write:\\n"\n'
        '    "  \\"Why statement Y might seem correct: [best argument for it]\\n"\n'
        '    "   Why it is actually wrong: [specific visual evidence that contradicts it]\\"\\n"\n'
        '    "- Finally confirm why your chosen statement IS grounded.\\n"\n'
    ),
    "attribute-checklist": (
        '    "- For each statement evaluate these visual attributes from the image:\\n"\n'
        '    "  (a) Colour/texture evidence for or against\\n"\n'
        '    "  (b) Shape/form evidence for or against\\n"\n'
        '    "  (c) Context/setting evidence for or against\\n"\n'
        '    "- Then state which statement has the strongest combined evidence.\\n"\n'
    ),
    "socratic": (
        '    "- Then answer these questions in order:\\n"\n'
        '    "  Q1: What is the single most distinctive visual feature in this image?\\n"\n'
        '    "  Q2: Does that feature directly support statement 1, 2, or 3?\\n"\n'
        '    "  Q3: What would the image need to show for each of the OTHER two statements "\n'
        '    "to be true? Is that present?\\n"\n'
        '    "  Q4: Therefore, which statement is grounded?\\n"\n'
    ),
}

COT_META = {   # folder -> (RUN_ID, Qwen CoT name, Qwen devtest CI)
    "evidence-first":      ("CoT1_evidence_first",      "CoT1", 0.044),
    "elimination":         ("CoT2_elimination",         "CoT2", 0.056),
    "confidence-ranked":   ("CoT3_confidence_ranked",   "CoT3", 0.046),
    "devils-advocate":     ("CoT4_devils_advocate",     "CoT4", 0.048),
    "attribute-checklist": ("CoT5_attribute_checklist", "CoT5", 0.042),
    "socratic":            ("CoT6_socratic",            "CoT6", 0.046),
}


def ref(ci):
    """Qwen reference metrics. On this task the three statements are decided jointly,
    so Combined = Q+ = 1-CI, Q- = 1-CI/2, and CFHR is 0 by construction."""
    return {"CI": ci, "Comb": 1 - ci, "CFHR": 0.0, "Q+": 1 - ci, "Q-": 1 - ci / 2}


def build_baseline() -> tuple[str, dict]:
    cells = [
        md("# InternVL2-2B — Task 1b baseline (per-statement True/False)\n\n"
           "Replicates **Qwen Run 1** (`../qwen2p5-3b-7b/baseline/`) on InternVL2-2B.\n"
           "Each statement is judged independently: 500 items x 3 statements = **1,500 forward passes**.\n\n"
           "Qwen Run 1 reference: **CI 0.257**, combined acc 0.740.\n\n"
           "> **Kaggle:** Settings -> Accelerator -> **GPU T4**. Then Run All.\n"
           "> Smoke test first: set `MAX_ITEMS = 5` and `SPLIT = 'dev'` in the config cell."),
        md("## 1. Install dependencies"), code(cell_install(MODEL_2B)),
        md("## 2. Configuration"),
        code(cell_config("run1_baseline_internvl2b", MODEL_2B, False, 6, 10,
                         "Run 1 (Qwen-3B)", ref(0.257), "~50 min on T4 (1,500 passes)")),
        md("## 3. Hugging Face login (optional)"), code(CELL_LOGIN),
        md("## 4. Download the split + images"), code(CELL_DOWNLOAD),
        md("## 5. Load InternVL2-2B"), code(CELL_MODEL),
        md("## 6. Official True/False parser\n\n"
           "Verbatim copy of `backbone.evaluate_tf` — the same parser the Codabench 1b scorer "
           "and every Qwen notebook use, so the scores are directly comparable."),
        code(PARSER_SRC),
        md("## 7. Prompt + inference helper"), code(CELL_VLM_CALL), code(CELL_BASELINE_PROMPT),
        md("## 8. Run inference\n\n"
           "Unparseable replies fall back to `false`. Since 2 of every 3 statements really are "
           "false, that is the correct prior, not a freebie — the fallback count is printed."),
        code(CELL_BASELINE_RUN),
        md("## 9. Write predictions CSV"), code(CELL_CSV),
        md("## 10. Score (Contrastive Instability)"), code(CELL_SCORE),
        md("## 11. Codabench submission zip"), code(CELL_ZIP),
    ]
    return "baseline/internvl2b-baseline.ipynb", notebook(cells)


def build_joint(folder, fname, run_id, model, quantize, answer_first, qwen_ref_name, qwen_ci, title, blurb):
    prompt_cell = (ANSWER_FIRST_HEAD + JOINT_ANSWER_FIRST_PROMPT) if answer_first \
        else (REASON_FIRST_HEAD + JOINT_REASON_FIRST_PROMPT)
    cells = [
        md(f"# {title}\n\n{blurb}\n\n"
           f"Qwen reference: **{qwen_ref_name}, CI {qwen_ci:.3f}**.\n\n"
           "> **Kaggle:** Settings -> Accelerator -> **GPU T4**. Then Run All.\n"
           "> Smoke test first: set `MAX_ITEMS = 5` and `SPLIT = 'dev'` in the config cell."),
        md("## 1. Install dependencies"), code(cell_install(model)),
        md("## 2. Configuration"),
        code(cell_config(run_id, model, quantize, 12, 256, qwen_ref_name, ref(qwen_ci),
                         "~40 min on T4" if quantize else "~25 min on T4")),
        md("## 3. Hugging Face login (optional)"), code(CELL_LOGIN),
        md("## 4. Download the split + images"), code(CELL_DOWNLOAD),
        md(f"## 5. Load {model.split('/')[-1]}"), code(CELL_MODEL),
        md("## 6. Official True/False parser"), code(PARSER_SRC),
        md("## 7. Prompt + inference"), code(CELL_VLM_CALL), code(prompt_cell), code(CELL_JOINT_INFER),
        md("## 8. Run inference"), code(CELL_JOINT_RUN),
        md("## 9. Write predictions CSV"), code(CELL_CSV),
        md("## 10. Score (Contrastive Instability)"), code(CELL_SCORE),
        md("## 11. Codabench submission zip"), code(CELL_ZIP),
    ]
    return f"{folder}/{fname}", notebook(cells)


def build_cot(variant):
    run_id, qwen_name, qwen_ci = COT_META[variant]
    prompt_cell = ANSWER_FIRST_HEAD + "\n" + COT_HEAD + COT_BODIES[variant] + COT_TAIL
    cells = [
        md(f"# InternVL2-8B — {variant.replace('-', ' ')} CoT ({run_id})\n\n"
           f"Replicates Qwen **{qwen_name}** (`../../qwen2p5-3b-7b/all-COT-variations-q7b/{variant}/`).\n"
           "Answer-first joint prompt + this CoT structure. Single forward pass per item.\n\n"
           f"Qwen {qwen_name} reference: **CI {qwen_ci:.3f}**  (Qwen best zero-shot: CoT5, CI 0.042).\n\n"
           "> **Kaggle:** Settings -> Accelerator -> **GPU T4**. Then Run All.\n"
           "> Smoke test first: set `MAX_ITEMS = 5` and `SPLIT = 'dev'` in the config cell."),
        md("## 1. Install dependencies"), code(cell_install(MODEL_8B)),
        md("## 2. Configuration"),
        code(cell_config(run_id, MODEL_8B, True, 12, 384, qwen_name, ref(qwen_ci), "~45 min on T4")),
        md("## 3. Hugging Face login (optional)"), code(CELL_LOGIN),
        md("## 4. Download the split + images"), code(CELL_DOWNLOAD),
        md("## 5. Load InternVL2-8B"), code(CELL_MODEL),
        md("## 6. Official True/False parser"), code(PARSER_SRC),
        md("## 7. Prompt + inference"), code(CELL_VLM_CALL), code(prompt_cell), code(CELL_JOINT_INFER),
        md("## 8. Run inference"), code(CELL_JOINT_RUN),
        md("## 9. Write predictions CSV"), code(CELL_CSV),
        md("## 10. Score (Contrastive Instability)"), code(CELL_SCORE),
        md("## 11. Codabench submission zip"), code(CELL_ZIP),
    ]
    return f"all-COT-variations-i8b/{variant}/cot-{variant}.ipynb", notebook(cells)


SMOKE = ["baseline", "answer-first-joint-i8b"]


def all_builds() -> dict[str, tuple[str, dict]]:
    b = {}
    b["baseline"] = build_baseline()
    b["joint-3-i2b"] = build_joint(
        "joint-3-i2b", "joint-3-stat-internvl2b.ipynb", "run3_joint_internvl2b", MODEL_2B, False,
        answer_first=False, qwen_ref_name="Run 3 (Qwen-3B)", qwen_ci=0.142,
        title="InternVL2-2B — joint 3-statement prompt, reason-first",
        blurb="Replicates **Qwen Run 3** (`../qwen2p5-3b-7b/joint-3-q3b/`). All three statements in "
              "one prompt; the model reasons first and names the grounded statement on the last line.")
    b["joint-3-i8b"] = build_joint(
        "joint-3-i8b", "joint-3-stat-internvl8b.ipynb", "run2_joint_internvl8b", MODEL_8B, True,
        answer_first=False, qwen_ref_name="Run 2 (Qwen-7B)", qwen_ci=0.092,
        title="InternVL2-8B — joint 3-statement prompt, reason-first",
        blurb="Replicates **Qwen Run 2** (`../qwen2p5-3b-7b/joint-3-q7b/`). Same joint reason-first "
              "prompt as the 2B run, so this isolates model scale.")
    b["answer-first-joint-i2b"] = build_joint(
        "answer-first-joint-i2b", "answer-first-internvl2b.ipynb", "run5_answer_first_internvl2b",
        MODEL_2B, False, answer_first=True, qwen_ref_name="Run 5 (Qwen-3B)", qwen_ci=0.082,
        title="InternVL2-2B — joint 3-statement prompt, answer-first",
        blurb="Replicates **Qwen Run 5** (`../qwen2p5-3b-7b/answer-first-joint-q3b/`). Identical to the "
              "reason-first 2B run except the answer comes on line 1 — this isolates prompt order.")
    b["answer-first-joint-i8b"] = build_joint(
        "answer-first-joint-i8b", "answer-first-internvl8b.ipynb", "run4_answer_first_internvl8b",
        MODEL_8B, True, answer_first=True, qwen_ref_name="Run 4 (Qwen-7B)", qwen_ci=0.050,
        title="InternVL2-8B — joint 3-statement prompt, answer-first",
        blurb="Replicates **Qwen Run 4** (`../qwen2p5-3b-7b/answer-first-joint-q7b/`), the strongest "
              "plain (non-CoT) Qwen system and the base that all six CoT variants build on.")
    for variant in COT_META:
        b[f"cot-{variant}"] = build_cot(variant)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="build all 11 notebooks (default: smoke-test 2)")
    ap.add_argument("--only", nargs="+", metavar="KEY", help="build just these runs, e.g. --only baseline joint-3-i2b")
    args = ap.parse_args()

    builds = all_builds()
    if args.only:
        unknown = set(args.only) - set(builds)
        if unknown:
            raise SystemExit(f"unknown run(s): {sorted(unknown)}\nknown: {sorted(builds)}")
        wanted = {k: builds[k] for k in args.only}
    elif args.all:
        wanted = builds
    else:
        wanted = {k: v for k, v in builds.items() if k in SMOKE}

    for key, (relpath, nb) in wanted.items():
        out = HERE / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

        # Syntax-check every code cell so a broken notebook never reaches Kaggle.
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                src = "".join(cell["source"])
                if src.lstrip().startswith("!") or "\n!" in src:
                    continue                       # shell magics are not valid Python
                try:
                    compile(src, f"{relpath}[cell {i}]", "exec")
                except SyntaxError as e:
                    print(f"  SYNTAX ERROR in {relpath} cell {i}: {e}", file=sys.stderr)
                    raise
        print(f"wrote {relpath}")

    print(f"\n{len(wanted)} notebook(s) built, all code cells compile.")


if __name__ == "__main__":
    main()
