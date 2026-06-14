# IE2026-HalDetect

Hallucination detection for ImageEval 2026 — Ayn-VQA Task 1b (English)

Given an image and three statements, predict which one is true (grounded) and which two are false (hallucinated). Exactly one statement per image is correct.

## Dev Phase Results

| Metric | Score |
|--------|-------|
| Contrastive Instability | **0.257** |
| Combined Accuracy | 0.74 |
| Q+ Accuracy | 0.912 |
| Q- Accuracy | 0.888 |

Beats baseline Qwen2.5-VL-3B (CI: 0.313 → 0.257)

## Dataset

`QCRI/AynVQA-ArabicNLP26` — config `task1b_en`

```python
from datasets import load_dataset
ds = load_dataset("QCRI/AynVQA-ArabicNLP26", "task1b_en", split="devtest")
