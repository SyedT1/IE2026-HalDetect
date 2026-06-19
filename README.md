# IE2026-HalDetect

Hallucination detection for **ImageEval 2026 — Ayn-VQA Task 1b (English)**

Given an image and three statements, predict which one is **True** (grounded in the image)
and which two are **False** (hallucinated). Exactly one statement per image is correct.

---

## Results

### Dev Phase (devtest, 500 items)

| Run | Method | CI ↓ | Combined Acc ↑ | CFHR ↓ | Q+ Acc ↑ | Q- Acc ↑ |
|-----|--------|:---:|:---:|:---:|:---:|:---:|
| Run 1 (baseline) | Qwen2.5-VL-3B, per-statement, greedy, max_new_tokens=10 | 0.257 | 0.740 | — | 0.912 | 0.888 |
| Run 3 | Qwen2.5-VL-3B, joint 3-statement prompt | 0.142 | 0.858 | 0.000 | 0.858 | 0.929 |
| **Run 2 (ours, best)** | **Qwen2.5-VL-7B, joint 3-statement prompt** | **0.092** | **0.908** | **0.000** | **0.908** | **0.954** |

**Run 2 reduces Contrastive Instability by 64.2% vs the baseline (0.257 → 0.092).**
**Run 3 shows that joint prompting alone (without the 7B model) accounts for a significant portion of the gain, reducing CI by 44.7% vs the baseline (0.257 → 0.142).**

---

## Method

### Run 1 — Baseline

**Model:** `Qwen2.5-VL-3B-Instruct`, `QUANTIZE=False`, `max_new_tokens=10`

Each statement is judged independently with a simple one-shot prompt and greedy decoding:

```
You are checking a statement against an image for visual hallucination.
Look only at what the image actually shows.

Statement: "{s}"

If the image clearly supports the statement, answer True. If the statement describes
something that is not in the image or is contradicted by it (a hallucination), answer
False. Answer with only one word: True or False.
```

500 items × 3 statements = 1,500 forward passes per run.

---

### Run 3 — Joint 3-Statement Prompt, Small Model

**Model:** `Qwen2.5-VL-3B-Instruct`, joint 3-statement prompt

Same joint prompting strategy as Run 2 (see below), but using the 3B model instead of 7B.
This run isolates the contribution of the joint prompting approach independently of model scale.

---

### Run 2 — Joint 3-Statement Prompt (ours, best)

**Model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4, `MAX_PIXELS=1024×28×28`, `max_new_tokens=256`

**Core idea:** instead of judging each statement in isolation, show all three to the model
simultaneously and ask it to identify which single one is grounded. This directly exploits
the task constraint (exactly one is True) and forces contrastive reasoning across statements.

```
You are a visual fact-checker examining an image from the Arab world.
Below are THREE statements about this image. Exactly ONE statement is
grounded in the image (True). The other two are plausible-sounding
hallucinations (False).

Statement 1: {s0}
Statement 2: {s1}
Statement 3: {s2}

Instructions:
- Study the image carefully.
- Reason step by step about each statement.
- On the very last line write ONLY: "Answer: X" where X is 1, 2, or 3.
```

The model responds with chain-of-thought reasoning then a final `Answer: X` line.
The chosen index is marked True; the other two are automatically False — the task
constraint is **enforced by design**, not by post-processing.

If the model does not follow the `Answer: X` format, individual CoT passes run as
fallback for that item only (< 5% of items in practice).

**Speed:** 500 × 1 pass = 500 forward passes → ~40 minutes on T4.

---

## Why Joint Prompting Works

| | Run 1 (baseline) | Run 3 | Run 2 (ours) |
|---|---|---|---|
| Model | Qwen2.5-VL-3B | Qwen2.5-VL-3B | Qwen2.5-VL-7B |
| Passes per item | 3 | 1 | 1 |
| Sees other statements | ❌ | ✅ | ✅ |
| Task constraint enforced | ❌ post-hoc | ✅ by design | ✅ by design |
| Can compare statements | ❌ | ✅ | ✅ |
| Reasoning | greedy, 10 tokens | chain-of-thought | chain-of-thought, 256 tokens |
| CI | 0.257 | 0.142 | **0.092** |
| CFHR | — | **0.000** | **0.000** |

Run 3 vs Run 1 isolates the effect of **joint prompting** (same 3B model, −44.7% CI).
Run 2 vs Run 3 isolates the effect of **model scale** (same joint prompt, −35.2% CI).
A CFHR of 0.000 means every item where the model correctly identified the True statement
also had both False statements correct — near-perfect internal consistency.

---

## Repository Structure

```
IE2026-HalDetect/
├── README.md
└── Development/
    └── baseline
    ---- Joint      # Joint 3-statement prompt notebook (Run 2, best)
```

---

## Dataset

**QCRI/AynVQA-ArabicNLP26** — config `task1b_en`

```python
from datasets import load_dataset
ds = load_dataset("QCRI/AynVQA-ArabicNLP26", "task1b_en", split="devtest")
```

| Split | Items | Labels |
|-------|------:|:------:|
| train | 3,000 | ✅ |
| dev | 500 | ✅ |
| devtest | 500 | ❌ blind |
| test | 1,000 | ❌ blind, final ranking |

---

## Reproduction

1. Upload `baseline/SyedT1.ipynb` to Kaggle
2. Enable **T4 GPU** under Settings → Accelerator
3. Add your HuggingFace token under **Add-ons → Secrets → HF_TOKEN**
4. Click **Run All** — completes in ~40 minutes
5. Download `prediction_en.zip` and submit to [Codabench](https://www.codabench.org)

---

## Shared Task

- **Task:** ImageEval 2026 — Task 1b Hallucination Detection (English)
- **Workshop:** ArabicNLP 2026
- **Task website:** https://imageeval2026.github.io/
- **Leaderboard metric:** Contrastive Instability (lower is better)