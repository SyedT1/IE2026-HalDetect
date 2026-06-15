# IE2026-HalDetect

Hallucination detection for **ImageEval 2026 — Ayn-VQA Task 1b (English)**

Given an image and three statements, predict which one is **True** (grounded in the image)
and which two are **False** (hallucinated). Exactly one statement per image is correct.

---

## Results

### Dev Phase (devtest, 500 items)

| Run | Method | Contrastive Instability ↓ | Combined Acc ↑ | CFHR ↓ | Q+ Acc ↑ | Q- Acc ↑ |
|-----|--------|:---:|:---:|:---:|:---:|:---:|
| Baseline (official) | Qwen2.5-VL-3B, independent per-statement | 0.313 | — | — | — | — |
| Run 1 | Qwen2.5-VL-7B, CoT, majority voting (N=3) | 0.257 | 0.740 | — | 0.912 | 0.888 |
| **Run 2 (best)** | **Qwen2.5-VL-7B, joint 3-statement prompt** | **0.092** | **0.908** | **0.000** | **0.908** | **0.954** |

**Run 2 beats the official baseline by 70.6% on Contrastive Instability (0.313 → 0.092).**

---

## Method

### Key Insight: Joint 3-Statement Prompting

The core improvement is shifting from **independent per-statement classification** to
**joint contrastive reasoning** over all three statements at once.

**Previous approach (Run 1):**
- 3 separate forward passes per item (one per statement)
- Model judged each statement in isolation with no knowledge of the others
- 500 items × 3 statements × 3 votes = 4,500 forward passes (~22 hours on T4)

**Current approach (Run 2):**
- 1 forward pass per item — all three statements shown simultaneously
- Model reasons comparatively and selects which single statement is grounded
- 500 items × 1 pass = 500 forward passes (~40 minutes on T4)
- Task constraint (exactly one True) is **enforced by design**, not post-processing

### Prompt Design

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

The model responds with chain-of-thought reasoning, then a final `Answer: X` line.
The chosen index is marked True; the other two are automatically marked False.

### Fallback

If the model does not follow the `Answer: X` format, individual chain-of-thought
passes run for each statement, followed by constraint enforcement (exactly one True).
In practice this fallback triggers for fewer than 5% of items.

### Model

- **Qwen/Qwen2.5-VL-7B-Instruct** (Qwen2.5-VL family)
- 4-bit NF4 quantization via `bitsandbytes` (fits Kaggle T4 16 GB)
- `MAX_PIXELS = 1024 × 28 × 28` (high resolution for fine-grained cultural details)
- `max_new_tokens = 256` (enough for step-by-step reasoning over 3 statements)

---

## Repository Structure

```
IE2026-HalDetect/
├── README.md
├── Development/
│   └── baseline
    --- joint 3 prompt          # Joint 3-statement prompt notebook (best run)
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
| devtest | 500 | ❌ (blind) |
| test | 1,000 | ❌ (blind, final ranking) |

---

## Reproduction

1. Upload `baseline/SyedT1.ipynb` to Kaggle
2. Enable **T4 GPU** accelerator
3. Add your HuggingFace token under **Add-ons → Secrets → HF_TOKEN**
4. Click **Run All** — completes in ~40 minutes
5. Download `prediction_en.zip` and submit to [Codabench](https://www.codabench.org)

---

## Shared Task

- **Task:** ImageEval 2026 — Task 1b Hallucination Detection (English)
- **Workshop:** ArabicNLP 2026
- **Task website:** https://imageeval2026.github.io/
- **Codabench:** task1b_en competition
- **Ranking metric:** Contrastive Instability (lower is better)
