# IE2026-HalDetect

Hallucination detection for **ImageEval 2026 — Ayn-VQA Task 1b (English)**

Given an image and three statements, predict which one is **True** (grounded in the image)
and which two are **False** (hallucinated). Exactly one statement per image is correct.

---

## Results

### Dev Phase (devtest, 500 items)

| Run | Method | CI ↓ | Combined Acc ↑ | CFHR ↓ | Q+ Acc ↑ | Q− Acc ↑ |
|-----|--------|:---:|:---:|:---:|:---:|:---:|
| Run 1 (baseline) | Qwen2.5-VL-3B, per-statement, greedy, max_new_tokens=10 | 0.257 | 0.740 | — | 0.912 | 0.888 |
| Run 3 | Qwen2.5-VL-3B, joint 3-statement prompt, reason→answer | 0.142 | 0.858 | 0.000 | 0.858 | 0.929 |
| Run 2 | Qwen2.5-VL-7B, joint 3-statement prompt, reason→answer | 0.092 | 0.908 | 0.000 | 0.908 | 0.954 |
| Run 5 | Qwen2.5-VL-3B, joint 3-statement prompt, answer→reason | 0.082 | 0.918 | 0.000 | 0.918 | 0.959 |
| Run 4 | Qwen2.5-VL-7B, joint 3-statement prompt, answer→reason | 0.050 | 0.950 | 0.000 | 0.950 | 0.975 |
| **Run ADE (best)** | **Run 4 + Latin square permutation ensemble (A+D+E)** | **0.042** | **0.958** | **0.000** | **0.958** | **0.979** |

**Run ADE reduces Contrastive Instability by 83.7% vs the baseline (0.257 → 0.042).**

The gains decompose cleanly across four orthogonal factors:

| Factor | Runs compared | CI reduction |
|--------|:---:|:---:|
| Joint prompting (same 3B model) | Run 1 → Run 3 | −44.7% |
| Model scale 3B → 7B (same joint prompt) | Run 3 → Run 2 | −35.2% |
| Answer-first vs reason-first (same 7B model) | Run 2 → Run 4 | −45.7% |
| Answer-first vs reason-first (same 3B model) | Run 3 → Run 5 | −42.3% |
| Permutation ensemble A+D+E | Run 4 → Run ADE | −16.0% |

> **Key finding (Run 5):** answer-first prompting on the 3B model (CI 0.082) outperforms
> reason-first on the 7B model (CI 0.092), confirming that prompt order is a stronger
> lever than model scale alone.

> **Key finding (Run ADE):** a Latin square permutation ensemble over three statement
> orderings eliminates the dev/devtest generalisation gap (both splits reach CI 0.042),
> suggesting the remaining gap in Run 4 was partly attributable to position bias.

---

## Error Analysis (dev split, 500 items)

Run 4 on the labelled dev split (CI 0.042, 21 failures) reveals a homogeneous error
pattern: **all 21 failures are type A — culturally plausible distractors**. Zero fallback
failures, zero same-category confusion failures.

Error sub-patterns identified through manual inspection:

| Sub-pattern | Examples | Share |
|---|---|:---:|
| Visual texture/material ambiguity | silk vs cotton scarf; masala chai vs saffron tea | ~30% |
| Intent/purpose inference | lanterns: decorative vs festival; green box: maintenance vs spiritual | ~40% |
| Fine-grained geometric/factual detail | Kuwaiti flag trapezoid vs rectangle; bathhouse fountain vs pool | ~20% |
| Event/activity classification | sports awards ceremony vs business meeting; croquet vs deck chairs | ~10% |

Hardest countries (CI > overall): Bahrain (0.103), Kuwait (0.103), Tunisia (0.069),
Palestine (0.069), UAE (0.067), Syria (0.067).

Hardest categories: Sports & Recreation (0.082), Food & Cooking (0.063),
Religion & Spirituality (0.061).

Ordering sensitivity (100 dev items × 6 permutations): 10% of items are
order-sensitive (correct in some orderings, wrong in others); position bias range = 0.055
(position 1: 95.5%, position 2: 92.5%, position 3: 90.0%). The Latin square ensemble
cancels this bias and closes the dev/devtest gap.

---

## Method

### Run 1 — Baseline

**Model:** `Qwen2.5-VL-3B-Instruct`, `QUANTIZE=False`, `max_new_tokens=10`

Each statement is judged independently with a simple zero-shot prompt and greedy decoding.
The model scores each statement in isolation and the predicted label is taken as the
index with the highest True probability:

$$\hat{y}_i = \underset{j \in \{1,2,3\}}{\arg\max} \ P_\theta\!\left(\texttt{"True"} \mid \mathcal{I}_i,\ s_i^{(j)}\right)$$

where $\mathcal{I}_i$ is the image and $s_i^{(j)}$ is the $j$-th statement scored alone.
Because each statement is seen in isolation the model has no access to the one-true
constraint and cannot perform contrastive reasoning across the three candidates.

500 items × 3 statements = **1,500 forward passes**.

```
You are checking a statement against an image for visual hallucination.
Look only at what the image actually shows.

Statement: "{s}"

If the image clearly supports the statement, answer True. If the statement describes
something that is not in the image or is contradicted by it (a hallucination), answer
False. Answer with only one word: True or False.
```

---

### Run 3 — Joint 3-Statement Prompt, Small Model

**Model:** `Qwen2.5-VL-3B-Instruct`, joint 3-statement prompt, reason→answer

Same joint prompting strategy as Run 2 (see below), but using the 3B model instead of 7B.
Isolates the contribution of joint prompting independently of model scale.

All three statements are presented together so the model performs a single constrained
selection. It generates a chain-of-thought $r_i$ and commits to the answer on the final
line (reason → answer):

$$\text{output}_i = \langle\, r_i,\ \texttt{Answer: }\hat{y}_i \,\rangle$$

$$\hat{y}_i = f_\theta\!\left(\mathcal{I}_i,\ s_i^{(1)}, s_i^{(2)}, s_i^{(3)}\right) \in \{1,2,3\}$$

500 items × 1 joint prompt = **500 forward passes**.

---

### Run 2 — Joint 3-Statement Prompt, Large Model

**Model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4, `MAX_PIXELS=1024×28×28`, `max_new_tokens=256`

**Core idea:** instead of judging each statement in isolation, show all three to the model
simultaneously and ask it to identify which single one is grounded. This directly exploits
the task constraint (exactly one is True) and forces contrastive reasoning across statements.

$$\hat{y}_i = \underset{j \in \{1,2,3\}}{\arg\max} \
    P_\theta\!\left(j \mid \mathcal{I}_i,\ s_i^{(1)}, s_i^{(2)}, s_i^{(3)},\ \text{``exactly one is True''}\right)$$

$$\text{output}_i = \langle\, r_i,\ \texttt{Answer: }\hat{y}_i \,\rangle$$

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

---

### Run 5 — Answer-First Joint Prompt, Small Model

**Model:** `Qwen2.5-VL-3B-Instruct`, joint 3-statement prompt, answer→reason

Same answer-first strategy as Run 4 (see below), but using the 3B model instead of 7B.
Isolates the contribution of answer-first prompting independently of model scale.
Achieves CI 0.082 — lower than reason-first on the 7B model (CI 0.092).

$$\text{output}_i = \langle\, \texttt{Answer: }\hat{y}_i,\ r_i \,\rangle
\quad \text{(answer → reason)}$$

versus Run 3:

$$\text{output}_i = \langle\, r_i,\ \texttt{Answer: }\hat{y}_i \,\rangle
\quad \text{(reason → answer)}$$

500 items × 1 joint prompt = **500 forward passes** (~20 min on T4).

---

### Run 4 — Answer-First Joint Prompt, Large Model

**Model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4, `MAX_PIXELS=1024×28×28`, `max_new_tokens=256`

**One change from Run 2:** the model commits to `Answer: X` on the **first** line, then
justifies. In Run 2 the answer came last (reason→answer). Here it comes first (answer→reason).

$$\text{output}_i = \langle\, \texttt{Answer: }\hat{y}_i,\ r_i \,\rangle
\quad \text{(answer → reason)}$$

**Why this works:** the M²CQA paper (arXiv:2602.05437, QCRI/HBKU) found that
reason-first prompting consistently increases counterfactual hallucination acceptance
on Arab cultural imagery, while answering before justifying improves robustness.
Run 4 confirms this finding on AynVQA: a single prompt inversion reduces CI by a
further 45.7% on top of Run 2. Run 5 shows the same effect holds on the 3B model.

The task constraint (exactly one True) is **enforced by design** — the chosen index
is marked True and the other two are automatically False.

**Speed:** 500 × 1 pass = 500 forward passes → ~40 minutes on T4.

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

---

### Run ADE — Latin Square Permutation Ensemble (best)

**Base system:** Run 4 (answer-first, Qwen2.5-VL-7B)

**Motivation:** ordering sensitivity analysis (100 dev items × 6 permutations) found
that 10% of items are order-sensitive with a position bias range of 0.055 — the model
favours statements at position 1 (95.5% accuracy) over position 3 (90.0%).

**Method:** run inference three times per item using a Latin square of statement orderings
that places each statement at each position exactly once, then take majority vote:

| Permutation | Statement order | Run |
|---|---|---|
| A | [1, 2, 3] | Run 4 |
| D | [2, 3, 1] | Run 5a |
| E | [3, 1, 2] | Run 5b |

For each item, the statement receiving 2 or 3 votes wins. In the rare case of a 3-way
tie, Run 4 (permutation A) is used as tiebreaker. The majority vote cancels position
bias because each statement appears at each position exactly once across the three runs.

**Result:** CI drops from 0.050 (Run 4 devtest) to **0.042**, eliminating the
dev/devtest generalisation gap entirely (dev CI was already 0.042 with Run 4 alone).

**Speed:** 3 × 500 passes ≈ 15 hours total on T4 (Run 4 already done; 5a and 5b
each ~5 hours, run in parallel on two sessions).

---

## System Comparison

| | Run 1 | Run 3 | Run 2 | Run 5 | Run 4 | Run ADE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Model | 3B | 3B | 7B | 3B | 7B | 7B ×3 |
| Passes per item | 3 | 1 | 1 | 1 | 1 | 3 |
| Sees other statements | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Task constraint enforced | ❌ post-hoc | ✅ by design | ✅ by design | ✅ by design | ✅ by design | ✅ by design |
| Answer position | — | last | last | **first** | **first** | **first** |
| Reasoning | greedy, 10 tok | CoT, 256 tok | CoT, 256 tok | CoT, 256 tok | CoT, 256 tok | CoT, 256 tok |
| Permutation ensemble | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ A+D+E** |
| CI ↓ | 0.257 | 0.142 | 0.092 | 0.082 | 0.050 | **0.042** |
| Combined Acc ↑ | 0.740 | 0.858 | 0.908 | 0.918 | 0.950 | **0.958** |
| CFHR ↓ | — | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |
| Q+ Acc ↑ | 0.912 | 0.858 | 0.908 | 0.918 | 0.950 | **0.958** |
| Q− Acc ↑ | 0.888 | 0.929 | 0.954 | 0.959 | 0.975 | **0.979** |

---

## Repository Structure

```
IE2026-HalDetect/
├── README.md
└── Development/
    ├── baseline/                   # Run 1: per-statement baseline (Qwen2.5-VL-3B)
    ├── joint-3-q7b/                # Run 2: joint prompt, reason→answer (Qwen2.5-VL-7B)
    ├── answer-first-q3b/           # Run 5: joint prompt, answer→reason (Qwen2.5-VL-3B)
    ├── answer-first-q7b/           # Run 4: joint prompt, answer→reason (Qwen2.5-VL-7B)
    └── ensemble-ADE/               # Run ADE: Latin square permutation ensemble + majority vote
```

Run 3 shares the notebook from `joint-3-q7b/` with `VLM_MODEL` switched to the 3B variant.
Run 5 shares the notebook from `answer-first-q7b/` with `VLM_MODEL` switched to the 3B variant.

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
| Run 5 | same as Run 4, set `VLM_MODEL` to `Qwen/Qwen2.5-VL-3B-Instruct` | T4 | ~20 min |
| Run 4 | `answer-first-q7b/run4-answer-first-qwen2p5vl7b.ipynb` | T4 | ~40 min |
| Run 5a | `ensemble-ADE/run5a-perm-D-qwen2p5vl7b.ipynb` | T4 | ~5 hrs |
| Run 5b | `ensemble-ADE/run5b-perm-E-qwen2p5vl7b.ipynb` | T4 | ~5 hrs |
| Run ADE | `ensemble-ADE/majority-vote-combiner-ADE.ipynb` | None | <1 min |

1. Upload the notebook to Kaggle
2. Enable **T4 GPU** under Settings → Accelerator (not needed for combiner)
3. Add your HuggingFace token under **Add-ons → Secrets → HF_TOKEN**
4. Set `SPLIT = 'dev'` to score locally; `SPLIT = 'devtest'` to produce a submission
5. Click **Run All**, download the predictions zip, submit to Codabench

Run 5a and 5b can be run in parallel on two separate Kaggle sessions.
The combiner requires the three CSVs from Run 4, 5a, and 5b as inputs.

---

## Shared Task

- **Task:** ImageEval 2026 — Task 1b Hallucination Detection (English)
- **Workshop:** ArabicNLP 2026
- **Task website:** https://imageeval2026.github.io/
- **Leaderboard:** https://www.codabench.org/competitions/17051
- **Leaderboard metric:** Contrastive Instability (lower is better)
- **Dataset licence:** CC BY-NC 4.0
