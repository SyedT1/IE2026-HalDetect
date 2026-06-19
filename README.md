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
| Run 3 | Qwen2.5-VL-3B, joint 3-statement prompt, reason→answer | 0.142 | 0.858 | 0.000 | 0.858 | 0.929 |
| Run 2 | Qwen2.5-VL-7B, joint 3-statement prompt, reason→answer | 0.092 | 0.908 | 0.000 | 0.908 | 0.954 |
| **Run 4 (best)** | **Qwen2.5-VL-7B, joint 3-statement prompt, answer→reason** | **0.050** | **0.950** | **0.000** | **0.950** | **0.975** |

**Run 4 reduces Contrastive Instability by 80.5% vs the baseline (0.257 → 0.050).**

The gains decompose cleanly across three orthogonal factors:

| Factor | Runs compared | CI reduction |
|--------|:---:|:---:|
| Joint prompting (same 3B model) | Run 1 → Run 3 | −44.7% |
| Model scale 3B → 7B (same joint prompt) | Run 3 → Run 2 | −35.2% |
| Answer-first vs reason-first (same 7B model) | Run 2 → Run 4 | −45.7% |

---

## Method

### Run 1 — Baseline

**Model:** `Qwen2.5-VL-3B-Instruct`, `QUANTIZE=False`, `max_new_tokens=10`

Each statement is judged independently with a simple zero-shot prompt and greedy decoding:

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

**Model:** `Qwen2.5-VL-3B-Instruct`, joint 3-statement prompt, reason→answer

Same joint prompting strategy as Run 2 (see below), but using the 3B model instead of 7B.
Isolates the contribution of joint prompting independently of model scale.

---

### Run 2 — Joint 3-Statement Prompt, Large Model

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

The model reasons step-by-step then commits to `Answer: X` on the final line.

---

### Run 4 — Answer-First Joint Prompt (best)

**Model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4, `MAX_PIXELS=1024×28×28`, `max_new_tokens=256`

**One change from Run 2:** the model commits to `Answer: X` on the **first** line, then
justifies. In Run 2 the answer came last (reason→answer). Here it comes first (answer→reason).

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
- On the VERY FIRST line write ONLY: "Answer: X" where X is 1, 2, or 3.
- Then explain step by step why that statement is grounded
  and why the other two are hallucinations.
Do not write anything before the Answer line.
```

**Why this works:** the M²CQA paper (arXiv:2602.05437, QCRI/HBKU) found that
reason-first prompting consistently increases counterfactual hallucination acceptance
on Arab cultural imagery, while answering before justifying improves robustness.
Run 4 confirms this finding on AynVQA: a single prompt inversion reduces CI by a
further 45.7% on top of Run 2.

The task constraint (exactly one True) is still **enforced by design** — the chosen index
is marked True and the other two are automatically False, regardless of prompt order.

**Speed:** 500 × 1 pass = 500 forward passes → ~40 minutes on T4.

---

## System Comparison

| | Run 1 | Run 3 | Run 2 | Run 4 |
|---|:---:|:---:|:---:|:---:|
| Model | Qwen2.5-VL-3B | Qwen2.5-VL-3B | Qwen2.5-VL-7B | Qwen2.5-VL-7B |
| Passes per item | 3 | 1 | 1 | 1 |
| Sees other statements | ❌ | ✅ | ✅ | ✅ |
| Task constraint enforced | ❌ post-hoc | ✅ by design | ✅ by design | ✅ by design |
| Can compare statements | ❌ | ✅ | ✅ | ✅ |
| Answer position | — | last | last | **first** |
| Reasoning | greedy, 10 tok | CoT, 256 tok | CoT, 256 tok | CoT, 256 tok |
| CI ↓ | 0.257 | 0.142 | 0.092 | **0.050** |
| Combined Acc ↑ | 0.740 | 0.858 | 0.908 | **0.950** |
| CFHR ↓ | — | **0.000** | **0.000** | **0.000** |
| Q+ Acc ↑ | 0.912 | 0.858 | 0.908 | **0.950** |
| Q− Acc ↑ | 0.888 | 0.929 | 0.954 | **0.975** |

---

## Repository Structure

```
IE2026-HalDetect/
├── README.md
└── Development/
    ├── baseline/                   # Run 1: per-statement baseline (Qwen2.5-VL-3B)
    ├── joint-3-q7b/                # Run 2: joint prompt, reason→answer (Qwen2.5-VL-7B)
    └── answer-first-q7b/           # Run 4: joint prompt, answer→reason (Qwen2.5-VL-7B)
```

Run 3 shares the notebook from `joint-3-q7b/` with `VLM_MODEL` switched to the 3B variant.

---

## Dataset

**QCRI/AynVQA-ArabicNLP26** — config `task1b_en`

```python
from datasets import load_dataset
ds = load_dataset("QCRI/AynVQA-ArabicNLP26", "task1b_en", split="devtest")
```

| Split | Items | Labels | Use |
|-------|------:|:---:|---|
| train | 3,000 | ✅ | training and fine-tuning |
| dev | 500 | ✅ | local validation |
| devtest | 500 | ❌ blind | dev-phase leaderboard |
| test | 1,000 | ❌ blind | final ranking |

---

## Reproduction

| Run | Notebook | GPU | Time |
|-----|----------|:---:|:---:|
| Run 1 | `baseline/SyedT1.ipynb` | T4 | ~15 min |
| Run 2 | `joint-3-q7b/joint-3-stat-qwen2p5vl7b.ipynb` | T4 | ~40 min |
| Run 3 | same as Run 2, set `VLM_MODEL` to `Qwen/Qwen2.5-VL-3B-Instruct` | T4 | ~20 min |
| Run 4 | `answer-first-q7b/run4-answer-first-qwen2p5vl7b.ipynb` | T4 | ~40 min |

1. Upload the notebook to Kaggle
2. Enable **T4 GPU** under Settings → Accelerator
3. Add your HuggingFace token under **Add-ons → Secrets → HF_TOKEN**
4. Set `SPLIT = 'dev'` to score locally; `SPLIT = 'devtest'` to produce a submission
5. Click **Run All**, download `prediction_run4_en.zip`, submit to Codabench

---

## Shared Task

- **Task:** ImageEval 2026 — Task 1b Hallucination Detection (English)
- **Workshop:** ArabicNLP 2026
- **Task website:** https://imageeval2026.github.io/
- **Leaderboard:** https://www.codabench.org/competitions/17051
- **Leaderboard metric:** Contrastive Instability (lower is better)
- **Dataset licence:** CC BY-NC 4.0