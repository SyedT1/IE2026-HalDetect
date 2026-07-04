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
| RunFT-3k | QLoRA Qwen2.5-VL-7B, 3000 images, ~600 steps | 0.036 | 0.964 | 0.000 | 0.964 | 0.982 |
| **RunFT (best)** | **QLoRA Qwen2.5-VL-7B, 2000 images, 500 steps** | **0.032** | **0.968** | **0.000** | **0.968** | **0.984** |

**RunFT reduces Contrastive Instability by 87.5% vs the baseline (0.257 → 0.032).**
**RunFT reduces CI by a further 23.8% below the best zero-shot system (0.042 → 0.032).**

> **Key finding (RunFT vs RunFT-3k):** training on 2,000 items for 500 steps (CI=0.032)
> outperforms training on 3,000 items for ~600 steps (CI=0.036). The 3,000-item run
> did not fully converge before the Kaggle 12h session limit cut it off at step ~600
> (loss still at 0.22 vs ~0.04 for the 2,000-item run at step 500). The 2,000-item
> run trained to completion within a single session and is the stronger system.

> **Key finding (RunFT):** QLoRA fine-tuning on 2,000 training items for 500 optimizer
> steps with frozen vision encoder achieves CI 0.032 — surpassing all zero-shot and
> prompting-based systems by a clear margin. This confirms that the 21 failures of the
> best zero-shot system (CoT5/ADE) were learnable from labelled examples and were not
> irreducible at 7B scale.

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
| **QLoRA fine-tuning, 2k items, 500 steps** | **CoT5 → RunFT** | **−23.8%** |

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
| RunFT-3k — 3000 items, ~600 steps | 0.036 | Incomplete convergence due to 12h Kaggle session limit; 2000-item run converged fully |
| Cultural grounding hint (Run A6) | — | Devtest submission zeroed due to wrong split; dev results inconclusive |

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
```

**CoT3 — Confidence-Ranked (CI 0.046)**
```
- Then rank ALL THREE: Most grounded / Less grounded / Least grounded
  with specific visual evidence for each.
```

**CoT6 — Socratic (CI 0.046)**
```
- Q1: Most distinctive visual feature?
  Q2: Does it support statement 1, 2, or 3?
  Q3: What would image need to show for others to be true?
  Q4: Therefore, which is grounded?
```

**CoT1 — Evidence-First (CI 0.044)**
```
- One sentence describing only what you literally see
  (do NOT use statement text) then explain your answer.
```

**CoT5 — Attribute Checklist (CI 0.042, best zero-shot)**
```
- For each statement evaluate:
    (a) Colour/texture evidence  (b) Shape/form  (c) Contextual evidence
  Then state your conclusion.
```

---

### Run ADE — Latin Square Permutation Ensemble

**Base:** Run 4 (answer-first, Qwen2.5-VL-7B)

| Permutation | Statement order | Run |
|---|---|---|
| A | [1, 2, 3] | Run 4 |
| D | [2, 3, 1] | Run 5a |
| E | [3, 1, 2] | Run 5b |

Majority vote (2-of-3 wins; 3-way tie → Run 4). CI 0.050 → **0.042**.
Speed: ~15 hours total on T4 (5a and 5b run in parallel).

---

### RunFT — QLoRA Fine-tuned Qwen2.5-VL-7B (best)

**Base model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4 QLoRA
**Training data:** 2,000 items from `train_en.jsonl` (3,000 available; dev not used)
**Training:** 500 optimizer steps, frozen vision encoder, LoRA rank 8
**Prompt:** CoT5 attribute checklist (same as best zero-shot system)
**Result:** CI **0.032**, Combined Acc **0.968**, CFHR **0.000**

**Architecture:** frozen vision encoder eliminates ~70% of backward-pass cost,
making training feasible on dual T4. Only LLM attention and MLP layers receive
LoRA adapters (~20M trainable of 8.3B total, 0.24%).

**Training dynamics:** loss 0.396 → 0.041 over 500 steps — full convergence
within a single 12h Kaggle session.

---

### RunFT-3k — QLoRA 3000 images (incomplete convergence)

**Base model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4 QLoRA
**Training data:** 3,000 items from `train_en.jsonl` (full training split)
**Training:** ~600 steps across three Kaggle sessions (step_100→300→600)
**Result:** CI **0.036** — worse than RunFT (CI 0.032)

Training ran across three sequential sessions due to Kaggle's 12h limit
(step1: 0→300, step2: 300→600, step3: 600→750 target but timed out again).
Loss was still at 0.217 at step 600 — not fully converged. The 2,000-item
run converged to loss 0.041 within a single session and is the stronger system.

---

## System Comparison

| | R1 | R3 | R2 | R5 | R4 | CoT5 | ADE | FT-3k | **FT** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Model | 3B | 3B | 7B | 3B | 7B | 7B | 7B×3 | 7B+LoRA | **7B+LoRA** |
| Train items | — | — | — | — | — | — | — | 3000 | **2000** |
| Train steps | — | — | — | — | — | — | — | ~600 | **500** |
| Converged | — | — | — | — | — | — | — | ❌ | **✅** |
| Fine-tuned | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | **✅** |
| CI ↓ | 0.257 | 0.142 | 0.092 | 0.082 | 0.050 | 0.042 | 0.042 | 0.036 | **0.032** |
| Comb ↑ | 0.740 | 0.858 | 0.908 | 0.918 | 0.950 | 0.958 | 0.958 | 0.964 | **0.968** |
| CFHR ↓ | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| Q+ ↑ | 0.912 | 0.858 | 0.908 | 0.918 | 0.950 | 0.958 | 0.958 | 0.964 | **0.968** |
| Q− ↑ | 0.888 | 0.929 | 0.954 | 0.959 | 0.975 | 0.979 | 0.979 | 0.982 | **0.984** |

---

## Repository Structure

```
IE2026-HalDetect/
├── README.md
└── Development/
    ├── baseline/                   # Run 1
    ├── joint-3-q7b/                # Run 2
    ├── answer-first-q3b/           # Run 5
    ├── answer-first-q7b/           # Run 4
    ├── cot-variants/               # CoT1–CoT6
    ├── ensemble-ADE/               # Run ADE
    └── finetune/
        ├── step1-qlora-q7b-3k-uptostep300.ipynb
        ├── step2-continue-from-300ckpt-qlora-q7b-3k.ipynb
        ├── step3-continue-from-600ckpt-qlora-q7b-3k.ipynb
        └── qlora-infer-final.ipynb
```

---

## Dataset

**QCRI/AynVQA-ArabicNLP26** — config `task1b_en`

| Split | Items | Labels | Use |
|-------|------:|:---:|---|
| train | 3,000 | ✅ | fine-tuning only (dev not used for training) |
| dev | 500 | ✅ | local validation |
| devtest | 500 | ❌ blind | dev-phase leaderboard |
| test | 1,000 | ❌ blind | final ranking |

---

## Reproduction

| Run | Notebook | GPU | Time |
|-----|----------|:---:|:---:|
| Run 1 | `baseline/SyedT1.ipynb` | T4 | ~15 min |
| Run 2 | `joint-3-q7b/joint-3-stat-qwen2p5vl7b.ipynb` | T4 | ~40 min |
| Run 3 | same as Run 2, `VLM_MODEL=Qwen2.5-VL-3B-Instruct` | T4 | ~20 min |
| Run 5 | same as Run 4, `VLM_MODEL=Qwen2.5-VL-3B-Instruct` | T4 | ~20 min |
| Run 4 | `answer-first-q7b/run4-answer-first-qwen2p5vl7b.ipynb` | T4 | ~40 min |
| CoT1–6 | `cot-variants/CoT[1-6]_*.ipynb` | T4 | ~40 min each |
| Run 5a | `ensemble-ADE/run5a-perm-D-qwen2p5vl7b.ipynb` | T4 | ~5 hrs |
| Run 5b | `ensemble-ADE/run5b-perm-E-qwen2p5vl7b.ipynb` | T4 | ~5 hrs |
| Run ADE | `ensemble-ADE/majority-vote-combiner-ADE.ipynb` | None | <1 min |
| RunFT train | `finetune/finetune-qlora-3k-v7.ipynb` (2k subset) | T4×2 | ~5 hrs |
| RunFT infer | `finetune/qlora-infer-final.ipynb` | T4 | ~40 min |

---

## Shared Task

- **Task:** ImageEval 2026 — Task 1b Hallucination Detection (English)
- **Workshop:** ArabicNLP 2026
- **Task website:** https://imageeval2026.github.io/
- **Leaderboard:** https://www.codabench.org/competitions/17051
- **Leaderboard metric:** Contrastive Instability (lower is better)
- **Dataset licence:** CC BY-NC 4.0
