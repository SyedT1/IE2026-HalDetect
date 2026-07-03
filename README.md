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
| CoT2 | Run 4 + elimination-based CoT | 0.056 | 0.944 | 0.000 | 0.944 | 0.972 |
| Res1280 | Run 4 + MAX_PIXELS=1280×28×28 | 0.054 | 0.946 | 0.000 | 0.946 | 0.973 |
| Run 4 | Qwen2.5-VL-7B, joint 3-statement prompt, answer→reason | 0.050 | 0.950 | 0.000 | 0.950 | 0.975 |
| CoT4 | Run 4 + devil's advocate CoT | 0.048 | 0.952 | 0.000 | 0.952 | 0.976 |
| CoT3 | Run 4 + confidence-ranked CoT | 0.046 | 0.954 | 0.000 | 0.954 | 0.977 |
| CoT6 | Run 4 + Socratic CoT | 0.046 | 0.954 | 0.000 | 0.954 | 0.977 |
| CoT1 | Run 4 + evidence-first CoT | 0.044 | 0.956 | 0.000 | 0.956 | 0.978 |
| CoT5 | Run 4 + attribute checklist CoT | 0.042 | 0.958 | 0.000 | 0.958 | 0.979 |
| Run ADE | Run 4 + Latin square permutation ensemble (A+D+E) | 0.042 | 0.958 | 0.000 | 0.958 | 0.979 |
| **RunFT (best)** | **QLoRA fine-tuned Qwen2.5-VL-7B + CoT5 prompt** | **0.032** | **0.968** | **0.000** | **0.968** | **0.984** |

**RunFT reduces Contrastive Instability by 87.5% vs the baseline (0.257 → 0.032).**
**RunFT reduces CI by a further 23.8% below the best zero-shot system (0.042 → 0.032).**

> **Key finding (RunFT):** QLoRA fine-tuning on 2,000 training items for 500 optimizer
> steps with frozen vision encoder achieves CI 0.032 — surpassing all zero-shot and
> prompting-based systems by a clear margin. This confirms that the 21 failures of the
> best zero-shot system (CoT5/ADE) were learnable from labelled examples and were not
> irreducible at 7B scale — they required task-specific adaptation of the LLM layers.

The gains decompose cleanly across orthogonal factors:

| Factor | Runs compared | CI reduction |
|--------|:---:|:---:|
| Joint prompting (same 3B model) | Run 1 → Run 3 | −44.7% |
| Model scale 3B → 7B (same joint prompt) | Run 3 → Run 2 | −35.2% |
| Answer-first vs reason-first (same 7B model) | Run 2 → Run 4 | −45.7% |
| Answer-first vs reason-first (same 3B model) | Run 3 → Run 5 | −42.3% |
| Devil's advocate CoT | Run 4 → CoT4 | −4.0% |
| Confidence-ranked CoT | Run 4 → CoT3 | −8.0% |
| Socratic CoT | Run 4 → CoT6 | −8.0% |
| Evidence-first CoT | Run 4 → CoT1 | −12.0% |
| Attribute checklist CoT | Run 4 → CoT5 | −16.0% |
| Permutation ensemble A+D+E | Run 4 → Run ADE | −16.0% |
| **QLoRA fine-tuning (frozen vision encoder)** | **CoT5 → RunFT** | **−23.8%** |

> **Key finding (Run 5):** answer-first prompting on the 3B model (CI 0.082) outperforms
> reason-first on the 7B model (CI 0.092), confirming that prompt order is a stronger
> lever than model scale alone.

> **Key finding (Run ADE):** a Latin square permutation ensemble over three statement
> orderings eliminates the dev/devtest generalisation gap (both splits reach CI 0.042),
> suggesting the remaining gap in Run 4 was partly attributable to position bias.

---

## Negative Results

The following methods were evaluated and did not improve over Run 4 (CI 0.050):

| Method | CI ↓ | Notes |
|--------|:---:|---|
| DoLa (layer 20, α=0.5) | 0.132 | Token-by-token contrastive decoding disrupts answer-first format compliance |
| Caption-then-verify cascade | 0.084 | Caption stage loses fine-grained visual detail needed for texture/intent errors |
| CoT2 — elimination | 0.056 | Falsification framing hurts; model rebuttals anchor to distractor vocabulary |
| Res1280 — higher resolution | 0.054 | MAX_PIXELS=1280×28×28 hurts vs 1024×28×28; extra visual tokens diffuse attention |
| Cultural grounding hint (Run A6) | — | Devtest submission zeroed due to wrong split; dev results inconclusive |

These results establish that CI=0.042 is the ceiling for **training-free** methods at 7B
scale under zero-shot inference. Fine-tuning (RunFT, CI=0.032) breaks through this ceiling,
confirming the remaining zero-shot failures were learnable rather than irreducible.

---

## CoT Ablation Summary

All CoT variants use Qwen2.5-VL-7B, answer-first format, single forward pass, 500 items.

| CoT variant | Strategy | CI ↓ | vs Run 4 |
|---|---|:---:|:---:|
| CoT2 — elimination | Rule out false statements before naming the true one | 0.056 | +12.0% ✗ |
| Run 4 — free-form | No structured instructions | 0.050 | baseline |
| CoT4 — devil's advocate | Steelman each distractor then rebut | 0.048 | −4.0% |
| CoT3 — confidence-ranked | Rank all three by visual evidence strength | 0.046 | −8.0% |
| CoT6 — Socratic | Answer structured sub-questions before concluding | 0.046 | −8.0% |
| CoT1 — evidence-first | Neutral image description before reasoning | 0.044 | −12.0% |
| **CoT5 — attribute checklist** | **Evaluate colour/texture/form/context per statement** | **0.042** | **−16.0%** |

**Pattern:** structured CoTs that force per-attribute visual grounding (CoT5, CoT1)
outperform those that operate on statements as holistic units (CoT4, CoT3, CoT6).
Elimination framing (CoT2) actively hurts by anchoring rebuttal reasoning to
distractor vocabulary, confirming that the failure mode is language-prior driven
rather than task-framing driven.

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

$$\hat{y}_i = \underset{j \in \{1,2,3\}}{\arg\max} \ P_\theta\!\left(\texttt{"True"} \mid \mathcal{I}_i,\ s_i^{(j)}\right)$$

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

$$\text{output}_i = \langle\, r_i,\ \texttt{Answer: }\hat{y}_i \,\rangle$$

500 items × 1 joint prompt = **500 forward passes**.

---

### Run 2 — Joint 3-Statement Prompt, Large Model

**Model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4, `MAX_PIXELS=1024×28×28`, `max_new_tokens=256`

$$\hat{y}_i = \underset{j \in \{1,2,3\}}{\arg\max} \
    P_\theta\!\left(j \mid \mathcal{I}_i,\ s_i^{(1)}, s_i^{(2)}, s_i^{(3)},\ \text{``exactly one is True''}\right)$$

$$\text{output}_i = \langle\, r_i,\ \texttt{Answer: }\hat{y}_i \,\rangle$$

```
You are a visual fact-checker examining an image from the Arab world.
Below are THREE statements about this image. Exactly ONE statement is
grounded in the image (True). The other two are plausible-sounding
hallucinations (False).

Statement 1: {s0}  Statement 2: {s1}  Statement 3: {s2}

Instructions:
- Study the image carefully.
- Reason step by step about each statement.
- On the very last line write ONLY: "Answer: X" where X is 1, 2, or 3.
```

---

### Run 5 — Answer-First Joint Prompt, Small Model

**Model:** `Qwen2.5-VL-3B-Instruct`, joint 3-statement prompt, answer→reason

$$\text{output}_i = \langle\, \texttt{Answer: }\hat{y}_i,\ r_i \,\rangle$$

500 items × 1 joint prompt = **500 forward passes** (~20 min on T4).

---

### Run 4 — Answer-First Joint Prompt, Large Model

**Model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4, `MAX_PIXELS=1024×28×28`, `max_new_tokens=256`

$$\text{output}_i = \langle\, \texttt{Answer: }\hat{y}_i,\ r_i \,\rangle
\quad \text{(answer → reason)}$$

**Why this works:** the M²CQA paper (arXiv:2602.05437, QCRI/HBKU) found that
reason-first prompting consistently increases counterfactual hallucination acceptance
on Arab cultural imagery, while answering before justifying improves robustness.

```
Instructions:
- Study the image carefully.
- On the VERY FIRST line write ONLY: "Answer: X" where X is 1, 2, or 3.
- Then explain step by step why that statement is grounded
  and why the other two are hallucinations.
Do not write anything before the Answer line.
```

**Speed:** ~40 minutes on T4.

---

### CoT Variants (CoT1–CoT6)

All share the Run 4 base prompt. Only the reasoning instructions after `Answer: X` differ.
All use `Qwen2.5-VL-7B-Instruct`, 4-bit NF4, `MAX_PIXELS=1024×28×28`, `max_new_tokens=384`.

**CoT2 — Elimination (CI 0.056, negative result)**
```
- Then: which statement can you rule out FIRST and why?
  Which SECOND and why? The remaining statement is grounded.
```

**CoT4 — Devil's Advocate (CI 0.048)**
```
- Then for each rejected statement: "Why it might seem correct: [argument]
  Why it is wrong: [visual evidence]"
- Finally confirm why your chosen statement IS grounded.
```

**CoT3 — Confidence-Ranked (CI 0.046)**
```
- Then rank ALL THREE statements:
  Most grounded: X — [visual evidence]
  Less grounded: Y — [why evidence is weak]
  Least grounded: Z — [why contradicted]
```

**CoT6 — Socratic (CI 0.046)**
```
- Then answer: Q1: Most distinctive visual feature?
  Q2: Does it support statement 1, 2, or 3?
  Q3: What would the image need to show for the others to be true? Is that present?
  Q4: Therefore, which statement is grounded?
```

**CoT1 — Evidence-First (CI 0.044)**
```
- On the second line write ONE sentence describing only what you literally
  see (objects, materials, colours, actions) — do NOT use the statement text.
- Then explain why that statement is grounded and the others are not.
```

**CoT5 — Attribute Checklist (CI 0.042, best zero-shot single-pass)**
```
- For each statement evaluate:
    (a) Colour/texture evidence for or against
    (b) Shape/form evidence
    (c) Contextual evidence
  Then state your conclusion.
```

---

### Res1280 — Higher Resolution (CI 0.054, negative result)

**Model:** Run 4 with `MAX_PIXELS=1280×28×28` instead of `1024×28×28`.

Increasing resolution hurts: CI rises from 0.050 to 0.054. The extra visual tokens
generated by higher resolution diffuse the model's attention across a larger input
without improving perception of the fine-grained details responsible for failure cases.
`MAX_PIXELS=1024×28×28` is the optimal setting for zero-shot inference.

---

### Run ADE — Latin Square Permutation Ensemble

**Base system:** Run 4 (answer-first, Qwen2.5-VL-7B)

**Motivation:** ordering sensitivity analysis (100 dev items × 6 permutations) found
10% of items are order-sensitive; position bias range = 0.055 (position 1: 95.5%,
position 3: 90.0%).

| Permutation | Statement order | Run |
|---|---|---|
| A | [1, 2, 3] | Run 4 |
| D | [2, 3, 1] | Run 5a |
| E | [3, 1, 2] | Run 5b |

Majority vote (2-of-3 wins; 3-way tie → Run 4). CI drops from 0.050 to **0.042**,
closing the dev/devtest gap entirely.

**Speed:** 3 × 500 passes ≈ 15 hours total on T4 (5a and 5b run in parallel).

---

### RunFT — QLoRA Fine-tuned Qwen2.5-VL-7B (best)

**Base model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4 QLoRA
**Training data:** 2,000 items subsampled from `train_en.jsonl` (3,000 total available)
**Training:** 500 optimizer steps, frozen vision encoder, LoRA rank 8 on LLM layers only
**Inference prompt:** CoT5 attribute checklist (same as best zero-shot system)

**Architecture:** QLoRA with frozen vision encoder — the ViT tower runs forward (image
features needed) but backward gradients are blocked at the frozen boundary. Only the
LLM attention and MLP layers receive LoRA adapters (~20M trainable parameters out of 8.3B
total, 0.24%). This design:
- Makes training feasible on dual T4 (16GB each) by eliminating vision backward cost
- Avoids corrupting the pre-trained visual representations
- Focuses adaptation on the cultural reasoning and contrastive statement selection task

**Training dynamics:** loss decreased from 0.396 → 0.041 over 320 logged steps,
indicating strong convergence on the 2,000-item subset.

**Result:** CI **0.032**, Combined Acc **0.968**, CFHR **0.000**, Q+ **0.968**, Q− **0.984**.
This is a 23.8% further reduction below the best zero-shot system (CoT5/ADE, CI=0.042),
confirming that the 21 remaining zero-shot failures were learnable from labelled examples
rather than irreducible at 7B scale.

**Speed:** inference ~40 minutes on T4 (same as zero-shot, base model + adapter load).

---

## System Comparison

| | R1 | R3 | R2 | R5 | CoT2 | Res1280 | R4 | CoT4 | CoT3 | CoT6 | CoT1 | CoT5 | ADE | **FT** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Model | 3B | 3B | 7B | 3B | 7B | 7B | 7B | 7B | 7B | 7B | 7B | 7B | 7B×3 | **7B+LoRA** |
| Passes | 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 3 | **1** |
| Joint | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **✅** |
| Ans first | — | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **✅** |
| CoT style | — | free | free | free | elim | free | free | devil | rank | socratic | evid | attr | free | **attr** |
| Fine-tuned | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Ensemble | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| CI ↓ | 0.257 | 0.142 | 0.092 | 0.082 | 0.056 | 0.054 | 0.050 | 0.048 | 0.046 | 0.046 | 0.044 | 0.042 | 0.042 | **0.032** |
| Comb ↑ | 0.740 | 0.858 | 0.908 | 0.918 | 0.944 | 0.946 | 0.950 | 0.952 | 0.954 | 0.954 | 0.956 | 0.958 | 0.958 | **0.968** |
| CFHR ↓ | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| Q+ ↑ | 0.912 | 0.858 | 0.908 | 0.918 | 0.944 | 0.946 | 0.950 | 0.952 | 0.954 | 0.954 | 0.956 | 0.958 | 0.958 | **0.968** |
| Q− ↑ | 0.888 | 0.929 | 0.954 | 0.959 | 0.972 | 0.973 | 0.975 | 0.976 | 0.977 | 0.977 | 0.978 | 0.979 | 0.979 | **0.984** |

---

## 3B QLoRA Fine-tuning Track (dev split)

An **efficiency-focused** track: can a small **Qwen2.5-VL-3B** recover most of the 7B
fine-tuning gain? RunFT above used 7B (CI 0.032). This track fine-tunes the **3B** model
with QLoRA and measures on the labelled **dev** split (500 items). All numbers are our own
matched baseline vs fine-tuned on the **same** dev items, so the deltas are apples-to-apples.

> Note: metrics here are **dev split**, CI + Combined Accuracy only (the CFHR / Q± split
> metrics are not recomputed for this track). Our 3B zero-shot baseline is measured at
> `MAX_PIXELS=768×28×28` and so reads 0.096, higher than the devtest Run 5 (0.082); the
> delta vs our own matched baseline is the fair comparison.

| Run | Method | CI ↓ | Comb Acc ↑ | vs our baseline |
|-----|--------|:---:|:---:|:---:|
| Baseline-3B | Qwen2.5-VL-3B, answer-first joint prompt, zero-shot, @768px | 0.096 | 0.904 | — |
| **SFT-3B** | **+ QLoRA-SFT, 1,500 train, 3 epochs, LoRA r=8 (q/k/v/o), @768px** | **0.058** | **0.942** | **−0.038 (−40% rel err)** |
| DPO-3B | + DPO on SFT (600 contrastive pairs, 150 steps, β=0.1, lr 5e-6) | 0.058 | 0.942 | 0.000 (flat) |

**Key finding (SFT-3B):** QLoRA-SFT cuts 3B error by **40% relative** (48→29 wrong of 500),
bringing a 3B model (CI 0.058) to within striking distance of the best 7B *zero-shot* system
(CoT5, CI 0.042) at a fraction of the inference cost. The 21→29 comparison is cross-split, but
the trajectory shows most of the 7B fine-tuning benefit is reachable at 3B scale.

**Key finding (DPO-3B, negative):** Direct Preference Optimization on top of the SFT model,
using the contrastive labels as free preference pairs (chosen = true statement, rejected =
a false one), produced **zero change** — CI 0.058, and **0 / 500 predictions differed** from
SFT. The gentle DPO update (β=0.1, lr 5e-6, 150 steps) shifted logits but not enough to flip
any greedy argmax; the SFT objective already saturates the contrastive signal. This is a clean
negative result: **preference tuning adds nothing once task-SFT is strong** on this data shape.

**Architecture:** 4-bit NF4 QLoRA, LoRA rank 8 on the LLM attention projections only
(~3.7M trainable params, 0.10% of 3.76B). Vision tower frozen. Training resumed across two
Kaggle T4 sessions via checkpointing (`device_map='auto'` shards the model over 2×T4).

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
    ├── cot-variants/               # CoT1–CoT6: structured CoT ablations
    ├── ensemble-ADE/               # Run ADE: Latin square permutation ensemble + majority vote
    ├── finetune/                   # RunFT: 7B QLoRA training + inference notebooks
    └── finetune-qlora-q3b/         # 3B QLoRA-SFT + DPO track (dev split); builders,
                                    #   Kaggle notebooks, results/ (preds, metrics, adapters)
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
| Run 3 | same as Run 2, `VLM_MODEL = Qwen/Qwen2.5-VL-3B-Instruct` | T4 | ~20 min |
| Run 5 | same as Run 4, `VLM_MODEL = Qwen/Qwen2.5-VL-3B-Instruct` | T4 | ~20 min |
| Run 4 | `answer-first-q7b/run4-answer-first-qwen2p5vl7b.ipynb` | T4 | ~40 min |
| CoT1–6 | `cot-variants/CoT[1-6]_*.ipynb` | T4 | ~40 min each |
| Run 5a | `ensemble-ADE/run5a-perm-D-qwen2p5vl7b.ipynb` | T4 | ~5 hrs |
| Run 5b | `ensemble-ADE/run5b-perm-E-qwen2p5vl7b.ipynb` | T4 | ~5 hrs |
| Run ADE | `ensemble-ADE/majority-vote-combiner-ADE.ipynb` | None | <1 min |
| RunFT train | `finetune/finetune-qlora-train-v6-selfcontained.ipynb` | T4×2 | ~10 hrs |
| RunFT infer | `finetune/qlora-infer-final.ipynb` | T4 | ~40 min |
| SFT-3B train | `finetune-qlora-q3b/kaggle-sft-finish-q3b.ipynb` | T4×2 | ~3 hrs |
| DPO-3B train | `finetune-qlora-q3b/kaggle-dpo-q3b.ipynb` | T4 | ~4 hrs |
| SFT/DPO-3B infer | `finetune-qlora-q3b/kaggle-infer-q3b.ipynb` | T4 | ~15 min |

1. Upload notebook to Kaggle → enable T4 GPU → add HF_TOKEN secret
2. Set `SPLIT = 'dev'` to score locally; `SPLIT = 'devtest'` for Codabench submission
3. Run All → download predictions zip → submit to Codabench 17051

Run 5a and 5b can be run in parallel on two separate Kaggle sessions.

---

## Shared Task

- **Task:** ImageEval 2026 — Task 1b Hallucination Detection (English)
- **Workshop:** ArabicNLP 2026
- **Task website:** https://imageeval2026.github.io/
- **Leaderboard:** https://www.codabench.org/competitions/17051
- **Leaderboard metric:** Contrastive Instability (lower is better)
- **Dataset licence:** CC BY-NC 4.0
