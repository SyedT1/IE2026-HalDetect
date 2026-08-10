# IE2026-HalDetect

Hallucination detection for **ImageEval 2026 — Ayn-VQA Task 1b (English)**

Given an image and three statements, predict which one is **True** (grounded in the image)
and which two are **False** (hallucinated). Exactly one statement per image is correct.

---

## Results

### Test Phase (1,000 images)

| Submission | Method | CI ↓ | Artifact |
|------------|--------|:----:|----------|
| CoT2-elimination | Qwen2.5-VL-7B, elimination CoT | 0.0730 | `Test/qwen2p5-3b-7b/All COT variations/cot-elimination/` |
| CoT6-socratic | Qwen2.5-VL-7B, Socratic CoT | 0.0670 | `Test/qwen2p5-3b-7b/All COT variations/cot-socratic/` |
| CoT4-devils-advocate | Qwen2.5-VL-7B, devil's-advocate CoT | 0.0660 | `Test/qwen2p5-3b-7b/All COT variations/cot-devils-advocate/` |
| CoT1-evidence-first | Qwen2.5-VL-7B, evidence-first CoT | 0.0630 | `Test/qwen2p5-3b-7b/All COT variations/cot-evidence-first/` |
| CoT5-attribute-checklist | Qwen2.5-VL-7B, attribute-checklist CoT | 0.0620 | `Test/qwen2p5-3b-7b/All COT variations/cot-attribute-checklist/` |
| CoT3-confidence-ranked | Qwen2.5-VL-7B, confidence-ranked CoT | 0.0620 | `Test/qwen2p5-3b-7b/All COT variations/cot-confidence-ranked/` |
| QLoRA-Q7B-2k-image | QLoRA fine-tuned Qwen2.5-VL-7B, 2,000 training items | 0.0490 | `Test/qwen2p5-3b-7b/qlora-q7b-2k-image/` |
| QLoRA-Q7B-2.3k-image | QLoRA fine-tuned Qwen2.5-VL-7B, 2,300 training items | 0.0390 | `Test/qwen2p5-3b-7b/qlora-q7b-2p3k-image/` |
| **QLoRA-Q7B-2.6k-image (best)** | **QLoRA fine-tuned Qwen2.5-VL-7B, 2,600 training items** | **0.0350** | `Test/qwen2p5-3b-7b/qlora-q7b-2p6k-image/` |
| QLoRA-Q7B-3k-image (legacy) | Resumed step-600 adapter from the nominal 3,000-item experiment | 0.0400 | `Test/qwen2p5-3b-7b/qlora-q7b-3k-image/` |

All test-phase submissions used the same inference image budget:
`MAX_PIXELS = 1024 × 28 × 28`.

### Dev Phase (devtest, 500 items)

| Run | Method | CI ↓ | Combined Acc ↑ | CFHR ↓ | Q+ Acc ↑ | Q− Acc ↑ |
|-----|--------|:---:|:---:|:---:|:---:|:---:|
| Intern-R3 | InternVL2-2B, joint 3-statement prompt, reason→answer | 0.298 | 0.702 | 0.000 | 0.702 | 0.851 |
| Run 1 (baseline) | Qwen2.5-VL-3B, per-statement, greedy, max_new_tokens=10 | 0.257 | 0.740 | — | 0.912 | 0.888 |
| Intern-R5 | InternVL2-2B, joint 3-statement prompt, answer→reason | 0.232 | 0.768 | 0.000 | 0.768 | 0.884 |
| Run 3 | Qwen2.5-VL-3B, joint 3-statement prompt, reason→answer | 0.142 | 0.858 | 0.000 | 0.858 | 0.929 |
| Intern-R2 | InternVL2-8B, joint 3-statement prompt, reason→answer | 0.098 | 0.902 | 0.000 | 0.902 | 0.951 |
| Run 2 | Qwen2.5-VL-7B, joint 3-statement prompt, reason→answer | 0.092 | 0.908 | 0.000 | 0.908 | 0.954 |
| Intern-R4 | InternVL2-8B, joint 3-statement prompt, answer→reason (free-form) | 0.084 | 0.916 | 0.000 | 0.916 | 0.958 |
| Intern-CoT5 | InternVL2-8B, Intern-R4 + attribute checklist CoT | 0.084 | 0.916 | 0.000 | 0.916 | 0.958 |
| Run 5 | Qwen2.5-VL-3B, joint 3-statement prompt, answer→reason | 0.082 | 0.918 | 0.000 | 0.918 | 0.959 |
| CoT2 | Run 4 + elimination-based CoT | 0.056 | 0.944 | 0.000 | 0.944 | 0.972 |
| Res1280 | Run 4 + MAX_PIXELS=1280×28×28 | 0.054 | 0.946 | 0.000 | 0.946 | 0.973 |
| Run 4 | Qwen2.5-VL-7B, joint 3-statement prompt, answer→reason | 0.050 | 0.950 | 0.000 | 0.950 | 0.975 |
| RunFT-3B | QLoRA fine-tuned Qwen2.5-VL-3B + CoT5 prompt, 2,000 items | 0.050 | 0.950 | 0.000 | 0.950 | 0.975 |
| CoT4 | Run 4 + devil's advocate CoT | 0.048 | 0.952 | 0.000 | 0.952 | 0.976 |
| CoT3 | Run 4 + confidence-ranked CoT | 0.046 | 0.954 | 0.000 | 0.954 | 0.977 |
| CoT6 | Run 4 + Socratic CoT | 0.046 | 0.954 | 0.000 | 0.954 | 0.977 |
| CoT1 | Run 4 + evidence-first CoT | 0.044 | 0.956 | 0.000 | 0.956 | 0.978 |
| RunFT-3B (3k) | QLoRA fine-tuned Qwen2.5-VL-3B (3,000 items) + CoT5 prompt | 0.044 | 0.956 | 0.000 | 0.956 | 0.978 |
| CoT5 | Run 4 + attribute checklist CoT | 0.042 | 0.958 | 0.000 | 0.958 | 0.979 |
| Run ADE | Run 4 + Latin square permutation ensemble (A+D+E) | 0.042 | 0.958 | 0.000 | 0.958 | 0.979 |
| RunFT (3k legacy) | Resumed step-600 QLoRA adapter from the nominal 3,000-item experiment | 0.0360 | — | — | — | — |
| RunFT (2.6k) | QLoRA fine-tuned Qwen2.5-VL-7B + CoT5 prompt, 2,600 items | 0.034 | 0.966 | 0.000 | 0.966 | 0.983 |
| RunFT | QLoRA fine-tuned Qwen2.5-VL-7B + CoT5 prompt, 2,000 items | 0.032 | 0.968 | 0.000 | 0.968 | 0.984 |
| **RunFT (2.3k) (best)** | **QLoRA fine-tuned Qwen2.5-VL-7B + CoT5 prompt, 2,300 items** | **0.028** | **0.972** | **0.000** | **0.972** | **0.986** |

#### Development-phase inference image budgets

The audit covered all 59 notebooks under `Development/`; 52 contain a generation path,
including training notebooks with internal checkpoint validation. The notebook that
performs final inference was used for each experiment folder. When a folder has no
separate inference notebook, the sole or combined experiment notebook was inspected.
`MAX_PIXELS` is a maximum Qwen visual-input budget, not a forced square resize; the
processor preserves the image aspect ratio.

| Inference budget | Development runs / notebooks |
|---|---|
| `MAX_PIXELS = 512 × 28 × 28` | **Run 1** (`baseline/qwen2p5vl-baseline.ipynb`). The older combined 3B QLoRA train/eval notebook `finetune-qlora-q3b/qlora-3b-colab.ipynb` also evaluates at 512, but it is not a scored row in the main devtest table. |
| `MAX_PIXELS = 768 × 28 × 28` | The matched **Baseline-3B / SFT-3B / DPO-3B** dev-split track under `finetune-qlora-q3b/`, including `qlora-3b-baseline-colab.ipynb` and the shared `kaggle-infer-q3b.ipynb`. The combined scaled and Unsloth 3B notebooks also evaluate at 768. |
| `MAX_PIXELS = 1024 × 28 × 28` | **Runs 2–5**; **CoT1–CoT6**; all three **Run ADE** permutation inference passes; **RunFT-3B** 2k and 3k; all final 7B QLoRA inference notebooks (**RunFT** 2k, 2.3k, 2.6k, and 3k legacy). The prepared HP2–HP6 notebooks also use 1024, including the unscored Qwen3-VL-8B HP6 notebook. |
| `MAX_PIXELS = 1280 × 28 × 28` | **Res1280 / HP1** and the prepared, unscored combined **HP7** notebook. |
| Not applicable | The `permutation-ensembling-q7b/permutation-ensembling/` notebook only combines existing CSV predictions and does not load or process images. |

InternVL2 does not expose Qwen's `MAX_PIXELS` option. Its corresponding resolution
control is the number of 448×448 image tiles (plus a thumbnail):

| InternVL image budget | Development runs / notebooks |
|---|---|
| `MAX_TILES = 6` | The standalone InternVL2-2B per-statement baseline notebook. |
| `MAX_TILES = 12` | **Intern-R2, Intern-R3, Intern-R4, Intern-R5, Intern-CoT5**, and all remaining prepared InternVL2-8B CoT notebooks. Thus every scored InternVL run in the table above uses 12 tiles. |

Training-time image budgets are separate from the final-inference groups above. The main
Qwen 3B/7B QLoRA training notebooks use `MAX_PIXELS = 256 × 28 × 28`, while their
standalone final inference notebooks use 1024. In the efficiency-focused 3B track, SFT
training uses 768 and DPO training uses 384, while the matched reported inference for
both uses 768.

**RunFT (2.3k) reduces Contrastive Instability by 89.1% vs the baseline (0.257 → 0.028).**
**RunFT (2.3k) reduces CI by a further 33.3% below the best zero-shot system (0.042 → 0.028).**
**RunFT (2.3k) reduces CI by 12.5% below the original 2,000-item fine-tune (RunFT, 0.032 → 0.028).**

> **Key finding (RunFT):** QLoRA fine-tuning on 2,000 training items for 500 optimizer
> steps with frozen vision encoder achieves CI 0.032 — surpassing all zero-shot and
> prompting-based systems by a clear margin. This confirms that the 21 failures of the
> best zero-shot system (CoT5/ADE) were learnable from labelled examples and were not
> irreducible at 7B scale — they required task-specific adaptation of the LLM layers.
> (Superseded by RunFT (2.3k) below, CI 0.028.)

> **Key finding (RunFT (2.3k)):** increasing the 7B QLoRA fine-tuning set from 2,000 to
> 2,300 items — same frozen-vision-encoder, LoRA-rank-8 recipe, same CoT5 inference
> prompt — drops CI further from 0.032 to **0.028**, a 12.5% relative reduction, making
> this the new best system overall and the first to clear both the zero-shot ceiling
> (CoT5/ADE, 0.042) and the original 2,000-item fine-tune by a wide margin. This shows
> training-set size has not saturated at 7B scale either: the same data-volume lever that
> helped 3B (RunFT-3B → RunFT-3B (3k), −12.0% from +1,000 items) also helps 7B, and at a
> proportionally smaller data increase (+300 items). A later nominal 3,000-item legacy
> run scored CI 0.036, but its resumed sessions did not preserve dataloader position;
> therefore a clean one-epoch evaluation over all 3,000 items remains untested.

> **Key finding (RunFT (2.6k)):** increasing the 7B QLoRA fine-tuning set further from
> 2,300 to **2,600** items — same frozen-vision-encoder, LoRA-rank-8 recipe, same CoT5
> inference prompt — does **not** continue the improvement: CI rises from 0.028 to
> **0.034** (Combined Acc 0.966, Q+ 0.966, Q− 0.983, CFHR 0.000; duration not recorded).
> That is a 21.4% relative *regression* vs RunFT (2.3k) and is also slightly worse than
> the original 2,000-item RunFT (0.032). RunFT (2.6k) still beats every zero-shot system
> (CoT5/ADE, 0.042) by 19.0% relative, so fine-tuning remains clearly beneficial, but the
> 7B data-volume curve is **non-monotonic**: 2,000 → 2,300 helped, while 2,300 → 2,600
> hurt. The sweet spot among the three tested subsample sizes is therefore **2,300 items**,
> not "more is better." The nominal 3,000-item legacy run (CI 0.036) does not resolve
> the curve because its resumed training did not cover a clean 3,000-item epoch.

> **Key finding (RunFT-3B):** QLoRA fine-tuning the smaller Qwen2.5-VL-3B model on the
> same 2,000-item subset lands at CI 0.050 — exactly matching zero-shot Run 4 (7B,
> answer-first, no fine-tuning) and falling short of both the best zero-shot prompting
> system (CoT5, CI 0.042) and the 7B fine-tuned system (RunFT, CI 0.032). Fine-tuning
> narrows the gap created by smaller model scale but does not close it: at 3B, QLoRA
> adaptation only buys back what the extra ~4B parameters of the 7B model already
> provided zero-shot — it does not reach the ceiling that fine-tuning unlocks at 7B.

> **Key finding (RunFT-3B (3k)):** retraining the identical 3B QLoRA recipe on the full
> 3,000-item train set (instead of the 2,000-item subsample) drops CI from 0.050 to
> **0.044** — a 12.0% relative reduction from training data alone, with model, LoRA rank,
> and inference prompt held fixed. This ties CoT1 (evidence-first zero-shot CoT) and
> surpasses CoT4, CoT3, and CoT6, though it still falls short of the best zero-shot
> ceiling (CoT5/ADE, 0.042) and the 7B fine-tune (RunFT, 0.032). Training-set size is a
> meaningful lever at 3B scale, but — like prompting — it cannot fully substitute for the
> base model capacity that the 7B backbone provides.

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
| QLoRA fine-tuning at 3B scale (frozen vision encoder) | Run 5 (3B, ans-first) → RunFT-3B | −39.0% |
| Training-set size 2,000 → 3,000 items (3B QLoRA, same recipe) | RunFT-3B → RunFT-3B (3k) | −12.0% |
| QLoRA fine-tuning at 7B scale (frozen vision encoder) | CoT5 → RunFT | −23.8% |
| **Training-set size 2,000 → 2,300 items (7B QLoRA, same recipe)** | **RunFT → RunFT (2.3k)** | **−12.5%** |
| Training-set size 2,300 → 2,600 items (7B QLoRA, same recipe) | RunFT (2.3k) → RunFT (2.6k) | **+21.4%** ✗ |
| Answer-first vs reason-first (InternVL2-2B) | Intern-R3 → Intern-R5 | −22.1% |
| Answer-first vs reason-first (InternVL2-8B) | Intern-R2 → Intern-R4 | −14.3% |
| Model scale InternVL2-2B → 8B (answer-first) | Intern-R5 → Intern-R4 | −63.8% |
| Model scale InternVL2-2B → 8B (reason-first) | Intern-R3 → Intern-R2 | −67.1% |
| Intern-R4 + attribute checklist CoT (InternVL2-8B) | Intern-R4 → Intern-CoT5 | **0.0%** ✗ |
| Qwen vs InternVL, large answer-first | Intern-R4 → Run 4 | −40.5% (Qwen lower CI) |
| Qwen vs InternVL, small answer-first | Intern-R5 → Run 5 | −64.7% (Qwen lower CI) |
| Qwen vs InternVL, large + CoT5 | Intern-CoT5 → CoT5 | −50.0% (Qwen lower CI) |

> **Key finding (Run 5):** answer-first prompting on the 3B model (CI 0.082) outperforms
> reason-first on the 7B model (CI 0.092), confirming that prompt order is a stronger
> lever than model scale alone.

> **Key finding (Run ADE):** a Latin square permutation ensemble over three statement
> orderings eliminates the dev/devtest generalisation gap (both splits reach CI 0.042),
> suggesting the remaining gap in Run 4 was partly attributable to position bias.

> **Key finding (InternVL cross-family):** re-running the joint-prompt matrix on
> InternVL2-2B / InternVL2-8B (contemporaries of Qwen2.5-VL-3B / 7B) shows the same
> qualitative levers transfer, but absolute error is higher on every matched cell.
> Answer-first still beats reason-first at both scales (2B: 0.298 → 0.232, −22.1%;
> 8B: 0.098 → 0.084, −14.3%), and scaling 2B → 8B remains the dominant gain
> (answer-first: 0.232 → 0.084, −63.8%). Best InternVL zero-shot so far is
> Intern-R4 / Intern-CoT5 (both CI **0.084**) — roughly 1.7× the error of Qwen Run 4
> (0.050) under the same prompt and split. The gap is a capability difference, not a
> porting artefact: CFHR is 0.000 on all scored InternVL runs, so format compliance holds.

> **Key finding (Intern-CoT5 = Intern-R4 + attribute checklist CoT, null):** the same
> scaffold that defines Qwen CoT5 (*Run 4 + attribute checklist CoT*, 0.050 → 0.042,
> −16.0%) yields **zero** CI reduction when ported as *Intern-R4 + attribute checklist
> CoT* on InternVL2-8B. Intern-CoT5 lands at CI **0.084**, identical to free-form
> Intern-R4 (Comb 0.916, Q+ 0.916, Q− 0.958, CFHR 0.000). Against Qwen CoT5 (0.042) that
> is a **2.00×** error ratio, wider than the free-form Intern-R4 vs Run 4 gap (1.68×).
> Structured CoT that buys a clean gain on Qwen does not transfer; InternVL's prompting
> ceiling under this recipe appears to already be at free-form answer-first.

---

## Negative Results

The following methods were evaluated and either failed to beat Run 4 (CI 0.050),
failed to beat a stronger peer under the same recipe, or were otherwise inconclusive:

| Method | CI ↓ | Notes |
|--------|:---:|---|
| DoLa (layer 20, α=0.5) | 0.132 | Token-by-token contrastive decoding disrupts answer-first format compliance |
| Caption-then-verify cascade | 0.084 | Caption stage loses fine-grained visual detail needed for texture/intent errors |
| Intern-CoT5 — Intern-R4 + attribute checklist CoT (InternVL2-8B) | 0.084 | Exact parallel of Qwen CoT5; ties free-form Intern-R4 exactly; CoT5's −16% Qwen gain does not transfer |
| CoT2 — elimination | 0.056 | Falsification framing hurts; model rebuttals anchor to distractor vocabulary |
| Res1280 — higher resolution | 0.054 | MAX_PIXELS=1280×28×28 hurts vs 1024×28×28; extra visual tokens diffuse attention |
| RunFT-3B — QLoRA fine-tuning at 3B scale, 2,000 items | 0.050 | Ties Run 4 zero-shot but does not beat best zero-shot (CoT5, 0.042) or 7B fine-tune (RunFT, 0.032); model capacity, not adaptation, is the binding constraint at 3B |
| RunFT (3k legacy) — resumed step-600 QLoRA adapter from the nominal 3,000-item 7B experiment | 0.0360 | Worse than RunFT 2k (0.032), RunFT 2.3k (0.028), and RunFT 2.6k (0.034); resume sessions restarted shuffled dataloaders, so this is not a clean full-3k comparison |
| RunFT (2.6k) — QLoRA at 7B, 2,600-item subsample | 0.034 | Beats zero-shot (CoT5 0.042) but *worse* than RunFT 2k (0.032) and RunFT 2.3k (0.028); 7B data-volume curve is non-monotonic — more training items is not always better |
| Cultural grounding hint (Run A6) | — | Devtest submission zeroed due to wrong split; dev results inconclusive |

These results establish that CI=0.042 is the ceiling for **training-free** methods at 7B
scale under zero-shot inference. Fine-tuning breaks through this ceiling at every data
cleanly tested size — RunFT (2,000 items, CI 0.032), RunFT (2.6k) (2,600 items,
CI 0.034), and RunFT (2.3k) (2,300 items, CI 0.028, current best system overall) — confirming the
remaining zero-shot failures were learnable rather than irreducible. RunFT-3B (2,000
items, CI 0.050) shows the same recipe at 3B scale is insufficient to reach the
zero-shot ceiling; more training data helps at 3B too (RunFT-3B (3k), 3,000 items,
CI 0.044), but still falls well short of either 7B fine-tune. Data volume is a real
lever at **both** scales, but it is **not monotonic** at 7B: 2,000 → 2,300 improved CI
by 12.5% relative, while 2,300 → 2,600 *regressed* by 21.4% relative (CI 0.028 → 0.034),
landing slightly worse than the original 2,000-item run. The 3B fine-tune improved
12.0% from +1,000 items (2,000 → 3,000). The 3B-vs-7B capacity gap remains the larger
effect: even RunFT-3B (3k)'s best result (0.044) is worse than RunFT's original,
smaller-data 7B run (0.032), and worse than the non-monotonic 2.6k 7B run (0.034).

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

$$\hat{y}_i = \arg\max_{j \in \{1,2,3\}} P_\theta(\text{"True"} \mid \mathcal{I}_i, s_i^{(j)})$$

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

$$\hat{y}_i = \arg\max_{j \in \{1,2,3\}} P_{\theta_{3B}}(j \mid \mathcal{I}_i, s_i^{(1)}, s_i^{(2)}, s_i^{(3)}, \text{"exactly one is True"})$$

$$\text{output}_i = (r_i, \text{"Answer: "}\hat{y}_i)$$

500 items × 1 joint prompt = **500 forward passes**.

---

### Run 2 — Joint 3-Statement Prompt, Large Model

**Model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4, `MAX_PIXELS=1024×28×28`, `max_new_tokens=256`

$$\hat{y}_i = \arg\max_{j \in \{1,2,3\}} P_\theta(j \mid \mathcal{I}_i, s_i^{(1)}, s_i^{(2)}, s_i^{(3)}, \text{"exactly one is True"})$$

$$\text{output}_i = (r_i, \text{"Answer: "}\hat{y}_i)$$

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

$$\hat{y}_i = \arg\max_{j \in \{1,2,3\}} P_{\theta_{3B}}(j \mid \mathcal{I}_i, s_i^{(1)}, s_i^{(2)}, s_i^{(3)})$$

$$\text{output}_i = (\text{"Answer: "}\hat{y}_i,\ r_i) \quad \text{(answer → reason)}$$

500 items × 1 joint prompt = **500 forward passes** (~20 min on T4).

---

### Run 4 — Answer-First Joint Prompt, Large Model

**Model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4, `MAX_PIXELS=1024×28×28`, `max_new_tokens=256`

$$\hat{y}_i = \arg\max_{j \in \{1,2,3\}} P_\theta(j \mid \mathcal{I}_i, s_i^{(1)}, s_i^{(2)}, s_i^{(3)})$$

$$\text{output}_i = (\text{"Answer: "}\hat{y}_i,\ r_i) \quad \text{(answer → reason)}$$

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

Every CoT variant is Run 4's decision rule conditioned on an additional reasoning-scaffold
instruction $c^{(k)}$ appended to the base prompt, where $k$ indexes the CoT variant:

$$\hat{y}_i^{(k)} = \arg\max_{j \in \{1,2,3\}} P_\theta(j \mid \mathcal{I}_i, s_i^{(1)}, s_i^{(2)}, s_i^{(3)}, c^{(k)})$$

$$\text{output}_i = (\text{"Answer: "}\hat{y}_i^{(k)},\ r_i^{(k)})$$

Only $c^{(k)}$ (the scaffold text) changes across CoT1–CoT6; the base image/statement
conditioning, model weights $\theta$, and answer-first output order are held fixed, so any
CI difference between variants is attributable purely to the reasoning scaffold.

**CoT2 — Elimination (CI 0.056, negative result)**

$$c^{(\text{CoT2})} = \text{"rule out statements sequentially; the survivor is grounded"}$$

```
- Then: which statement can you rule out FIRST and why?
  Which SECOND and why? The remaining statement is grounded.
```

**CoT4 — Devil's Advocate (CI 0.048)**

$$c^{(\text{CoT4})} = \text{"steelman each rejected statement, then rebut with visual evidence"}$$

```
- Then for each rejected statement: "Why it might seem correct: [argument]
  Why it is wrong: [visual evidence]"
- Finally confirm why your chosen statement IS grounded.
```

**CoT3 — Confidence-Ranked (CI 0.046)**

$$c^{(\text{CoT3})} = \text{"rank the three statements by visual-evidence strength, most to least"}$$

```
- Then rank ALL THREE statements:
  Most grounded: X — [visual evidence]
  Less grounded: Y — [why evidence is weak]
  Least grounded: Z — [why contradicted]
```

**CoT6 — Socratic (CI 0.046)**

$$c^{(\text{CoT6})} = (q_1, q_2, q_3, q_4) \quad \text{a fixed sub-question sequence}$$

```
- Then answer: Q1: Most distinctive visual feature?
  Q2: Does it support statement 1, 2, or 3?
  Q3: What would the image need to show for the others to be true? Is that present?
  Q4: Therefore, which statement is grounded?
```

**CoT1 — Evidence-First (CI 0.044)**

$$c^{(\text{CoT1})} = \text{"describe the image in one neutral sentence before reasoning, without reusing statement vocabulary"}$$

```
- On the second line write ONE sentence describing only what you literally
  see (objects, materials, colours, actions) — do NOT use the statement text.
- Then explain why that statement is grounded and the others are not.
```

**CoT5 — Attribute Checklist (CI 0.042, best zero-shot single-pass)**

$$c^{(\text{CoT5})} = \{a_{\text{colour/texture}}^{(j)},\ a_{\text{shape/form}}^{(j)},\ a_{\text{context}}^{(j)} : j = 1,2,3\}$$

i.e. a per-statement, per-attribute evidence vector evaluated before the conclusion:

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

Identical decision rule to Run 4; only the image tokenization resolution $P_{\max}$ changes,
which alters the number of visual tokens $N_v(\mathcal{I}_i)$ fed to the model:

$$N_v(\mathcal{I}_i) = \lceil \min(\text{pixels}(\mathcal{I}_i), P_{\max}) / (28 \times 28) \rceil$$

$$\hat{y}_i = \arg\max_{j \in \{1,2,3\}} P_\theta(j \mid \mathcal{I}_i^{[N_v]}, s_i^{(1)}, s_i^{(2)}, s_i^{(3)})$$

where $\mathcal{I}_i^{[N_v]}$ denotes the image patch-tokenized at $N_v$ tokens under
$P_{\max}=1280\times28\times28$ (vs. $1024\times28\times28$ for Run 4). Increasing $P_{\max}$
raises $N_v$ and CI rises from 0.050 to 0.054 — the extra visual tokens diffuse the model's
attention across a larger input without improving perception of the fine-grained details
responsible for failure cases. `MAX_PIXELS=1024×28×28` is the optimal setting for zero-shot
inference.

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

Each permutation $\pi \in \{A, D, E\}$ reorders the statements before applying Run 4's
decision rule, then predictions are combined by majority vote over the original label
space (each $\hat{y}_i^{(\pi)}$ is mapped back to its statement identity before voting):

$$\hat{y}_i^{(\pi)} = \arg\max_{j \in \{1,2,3\}} P_\theta(j \mid \mathcal{I}_i, s_i^{(\pi(1))}, s_i^{(\pi(2))}, s_i^{(\pi(3))})$$

$$\hat{y}_i^{\text{ADE}} = \arg\max_{j \in \{1,2,3\}} \sum_{\pi \in \{A,D,E\}} \mathbb{1}[\hat{y}_i^{(\pi)} = j]$$

ties → $\hat{y}_i^{(A)}$. Majority vote (2-of-3 wins; 3-way tie → Run 4). CI drops from
0.050 to **0.042**, closing the dev/devtest gap entirely.

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

For a frozen 4-bit weight $W_0 \in \mathbb{R}^{d \times k}$ in an LLM attention/MLP
layer, LoRA adds a low-rank update $\Delta W = BA$ with rank $r=8$:

$$W = W_0 + \Delta W = W_0 + BA, \qquad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$$

$$h = W_0 x + \frac{\alpha}{r} BAx$$

where $x$ is the layer input, $\alpha$ is the LoRA scaling factor, and only $A, B$
(the frozen ViT is entirely excluded) receive gradients. Training minimizes the
next-token cross-entropy loss over the answer-first target sequence
$y_i = (\text{"Answer: "}\hat{y}_i,\ r_i)$ of length $T_i$, restricted to
the LoRA parameters $\Phi = \{A_\ell, B_\ell\}$ across LLM layers $\ell$:

$$\mathcal{L}(\Phi) = -\frac{1}{N}\sum_{i=1}^{N} \frac{1}{T_i}\sum_{t=1}^{T_i} \log P_{\theta_0, \Phi}(y_{i,t} \mid \mathcal{I}_i, s_i^{(1:3)}, y_{i,<t})$$

$$\Phi^{*} = \arg\min_{\Phi} \mathcal{L}(\Phi), \qquad \theta_0 \ \text{(base + ViT) frozen}$$

**Training dynamics:** loss decreased from 0.396 → 0.041 over 320 logged steps,
indicating strong convergence on the 2,000-item subset. Inference then applies the
CoT5 decision rule with the fine-tuned parameters:

$$\hat{y}_i = \arg\max_{j \in \{1,2,3\}} P_{\theta_0, \Phi^{*}}(j \mid \mathcal{I}_i, s_i^{(1)}, s_i^{(2)}, s_i^{(3)}, c^{(\text{CoT5})})$$

**Result:** CI **0.032**, Combined Acc **0.968**, CFHR **0.000**, Q+ **0.968**, Q− **0.984**.
This is a 23.8% further reduction below the best zero-shot system (CoT5/ADE, CI=0.042),
confirming that the 21 remaining zero-shot failures were learnable from labelled examples
rather than irreducible at 7B scale. (Superseded by RunFT (2.3k) below, CI 0.028 — see
that section for the 2,000→2,300-item follow-up.)

**Speed:** inference ~40 minutes on T4 (same as zero-shot, base model + adapter load).

---

### RunFT-3B — QLoRA Fine-tuned Qwen2.5-VL-3B

**Base model:** `Qwen2.5-VL-3B-Instruct`, 4-bit NF4 QLoRA
**Training data:** 2,000 items subsampled from `train_en.jsonl` (same subset as RunFT)
**Training:** frozen vision encoder, LoRA rank 8 on LLM layers only (same recipe as RunFT)
**Inference prompt:** CoT5 attribute checklist

Identical objective and adapter structure to RunFT, with the frozen base swapped from
$\theta_0^{7B}$ to $\theta_0^{3B}$ (LoRA rank $r=8$, same $A,B$ shapes relative to the
smaller layer widths):

$$\Phi_{3B}^{*} = \arg\min_{\Phi} \mathcal{L}(\Phi; \theta_0^{3B})$$

$$\hat{y}_i = \arg\max_{j \in \{1,2,3\}} P_{\theta_0^{3B}, \Phi_{3B}^{*}}(j \mid \mathcal{I}_i, s_i^{(1)}, s_i^{(2)}, s_i^{(3)}, c^{(\text{CoT5})})$$

where $\mathcal{L}(\cdot; \theta_0^{3B})$ is the same cross-entropy loss defined above,
evaluated with the 3B backbone frozen in place of the 7B one. The resulting CI gap

$$\text{CI(RunFT-3B)} - \text{CI(RunFT)} = 0.050 - 0.032 = 0.018$$

is attributable entirely to $\theta_0^{3B}$ vs. $\theta_0^{7B}$ base capacity, since
$\mathcal{L}$, the LoRA rank, the training data, and $c^{(\text{CoT5})}$ are held fixed.

**Result:** CI **0.050**, Combined Acc **0.950**, CFHR **0.000**, Q+ **0.950**, Q− **0.975**.
Training wall-clock/duration not recorded for this run.

**Takeaway:** applying the exact RunFT fine-tuning recipe to the smaller 3B backbone
recovers the CI 0.050 level of zero-shot Run 4 (7B) but plateaus there — it does not
reach either the best zero-shot prompting system (CoT5, 0.042) or the 7B fine-tuned
system (RunFT, 0.032). Comparing RunFT-3B against Run 5 (3B, answer-first, zero-shot,
CI 0.082) shows fine-tuning still delivers a large relative gain at 3B scale (−39.0%),
but the ceiling it reaches is capped by base model capacity, not by the amount or
quality of task adaptation. In other words: at 3B, QLoRA fine-tuning is necessary but
not sufficient to match what 7B achieves either zero-shot or fine-tuned.

---

### RunFT-3B (3k) — QLoRA Fine-tuned Qwen2.5-VL-3B, Full Train Set

**Base model:** `Qwen2.5-VL-3B-Instruct`, 4-bit NF4 QLoRA
**Training data:** all 3,000 items in `train_en.jsonl` (vs. the 2,000-item subsample used
for RunFT-3B above)
**Training:** frozen vision encoder, LoRA rank 8 on LLM layers only (identical recipe to
RunFT-3B; training duration not recorded)
**Inference prompt:** CoT5 attribute checklist, answer-first
**Inference notebook:** `finetune/step-3-qlora-q3b-3k-inference.ipynb`
(`RUN_ID = runFT_qlora_3b`, `MAX_PIXELS=1024×28×28`, `MAX_NEW_TOKENS=256`)

This isolates the effect of training-set size at fixed model scale and fixed LoRA recipe:
only the number of labelled fine-tuning examples changes relative to RunFT-3B.

$$\Phi_{3B,3k}^{*} = \arg\min_{\Phi} \mathcal{L}(\Phi; \theta_0^{3B}) \ \text{over all } N=3000 \text{ training items}$$

Inference on `devtest` resolved all 500 items on the primary answer-first pass with
**zero fallback triggers** (0.0%), and predictions were submitted to
[Codabench 17051](https://www.codabench.org/competitions/17051).

**Result:** CI **0.044**, Combined Acc **0.956**, CFHR **0.000**, Q+ **0.956**, Q− **0.978**.
Training/inference duration not recorded for this run.

$$\text{CI(RunFT-3B)} - \text{CI(RunFT-3B (3k))} = 0.050 - 0.044 = 0.006 \quad (-12.0\% \text{ relative})$$

**Takeaway:** the extra 1,000 training items buy a real, if partial, improvement — CI
drops from 0.050 to 0.044, matching CoT1's zero-shot evidence-first CoT and beating
CoT4, CoT3, and CoT6. But it still does not reach the best zero-shot ceiling (CoT5/ADE,
0.042) or the 7B fine-tune (RunFT, 0.032). Combined with the earlier RunFT-3B result,
this suggests base model capacity (3B vs. 7B) remains the dominant constraint at this
task: more training data helps at the margin, but scaling the backbone helps more.

---

### RunFT (2.3k) — QLoRA Fine-tuned Qwen2.5-VL-7B, 2,300 items (best overall)

**Base model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4 QLoRA
**Training data:** 2,300 items subsampled from `train_en.jsonl` (vs. the 2,000-item
subsample used for RunFT above; 3,000 total available)
**Training:** frozen vision encoder, LoRA rank 8 on LLM layers only — identical recipe to
RunFT, same training notebook (`finetune/finetune-qlora-train-v6-selfcontained.ipynb`)
with the training subsample size increased from 2,000 to 2,300 items; training duration
not recorded
**Inference prompt:** CoT5 attribute checklist, answer-first (`finetune/qlora-infer-final.ipynb`)

This isolates the effect of training-set size at fixed model scale (7B) and fixed LoRA
recipe, mirroring the RunFT-3B → RunFT-3B (3k) comparison but on the larger backbone:

$$\Phi_{7B,2.3k}^{*} = \arg\min_{\Phi} \mathcal{L}(\Phi; \theta_0^{7B}) \ \text{over all } N=2300 \text{ training items}$$

$$\hat{y}_i = \arg\max_{j \in \{1,2,3\}} P_{\theta_0^{7B}, \Phi_{7B,2.3k}^{*}}(j \mid \mathcal{I}_i, s_i^{(1)}, s_i^{(2)}, s_i^{(3)}, c^{(\text{CoT5})})$$

**Result:** CI **0.028**, Combined Acc **0.972**, CFHR **0.000**, Q+ **0.972**, Q− **0.986**.
Training/inference duration not recorded for this run.

$$\text{CI(RunFT)} - \text{CI(RunFT (2.3k))} = 0.032 - 0.028 = 0.004 \quad (-12.5\% \text{ relative})$$

**Takeaway:** a modest 300-item increase in training data (2,000 → 2,300, +15%) yields a
12.5% relative CI reduction at 7B scale — comparable in relative terms to the 12.0%
reduction that a much larger 1,000-item increase (+50%) bought at 3B scale
(RunFT-3B → RunFT-3B (3k)). This suggests 7B may be more data-efficient per additional
labelled item than 3B in the mid-subsample regime, though the two comparisons differ in
both absolute and relative data increase, so this is suggestive rather than a controlled
comparison. A further increase to 2,600 items, however, does **not** continue the
trend — see RunFT (2.6k) below. RunFT (2.3k) remains the best system produced by this
project to date.

---

### RunFT (2.6k) — QLoRA Fine-tuned Qwen2.5-VL-7B, 2,600 items

**Base model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4 QLoRA
**Training data:** 2,600 items subsampled from `train_en.jsonl` (vs. 2,000 for RunFT and
2,300 for RunFT (2.3k); 3,000 total available). Subsample is deterministic (`SEED=42`).
**Training:** frozen vision encoder, LoRA rank 8 on LLM layers only — same recipe as
RunFT / RunFT (2.3k); `TRAIN_SUBSAMPLE_N=2600`, `MAX_STEPS=510`; training notebooks under
`Development/qwen2p5-3b-7b/qwen-q7b-2p6k-image/` (`resume-from-checkpoint-300.ipynb`);
training duration not recorded (Codabench duration field = −1.0)
**Inference prompt:** CoT5 attribute checklist, answer-first
(`Development/qwen2p5-3b-7b/qwen-q7b-2p6k-image/inference-qlora-2p6k-q7b-v1.ipynb`)

This continues the 7B data-volume ablation at fixed model scale and fixed LoRA recipe:

$$\Phi_{7B,2.6k}^{*} = \arg\min_{\Phi} \mathcal{L}(\Phi; \theta_0^{7B}) \ \text{over all } N=2600 \text{ training items}$$

$$\hat{y}_i = \arg\max_{j \in \{1,2,3\}} P_{\theta_0^{7B}, \Phi_{7B,2.6k}^{*}}(j \mid \mathcal{I}_i, s_i^{(1)}, s_i^{(2)}, s_i^{(3)}, c^{(\text{CoT5})})$$

**Result:** CI **0.034**, Combined Acc **0.966**, CFHR **0.000**, Q+ **0.966**, Q− **0.983**.
Duration not recorded (−1.0 on the leaderboard).

$$\text{CI(RunFT (2.3k))} - \text{CI(RunFT (2.6k))} = 0.028 - 0.034 = -0.006 \quad (+21.4\% \text{ relative CI, regression})$$

$$\text{CI(RunFT)} - \text{CI(RunFT (2.6k))} = 0.032 - 0.034 = -0.002 \quad (+6.3\% \text{ relative CI vs 2k})$$

**Takeaway:** among the three 7B QLoRA subsample sizes tested — 2,000 / 2,300 / 2,600 —
the CI curve is **non-monotonic**: 0.032 → **0.028** → 0.034. Adding 300 items past the
2.3k sweet spot *hurts* rather than helps, and the 2.6k run even lands slightly worse
than the original 2k run. Fine-tuning still clearly beats the zero-shot ceiling
(CoT5/ADE 0.042 → 0.034, −19.0% relative), so the failures remain learnable; the lesson
is that data volume must be tuned, not maximised. A nominal 3,000-item legacy run scored
CI **0.036** on devtest and **0.0400** on the 1,000-image test phase, but its resumed
sessions restarted shuffled dataloaders instead of continuing the prior sample order.
It is therefore not a clean full-data point and cannot determine whether 2.3k is a local
optimum or whether a correctly trained full-data model recovers (or worsens further).

---

### RunFT (3k legacy) — Resumed QLoRA Experiment

**Base model:** `Qwen2.5-VL-7B-Instruct`, 4-bit NF4 QLoRA

**Experiment directory:** `Development/qwen2p5-3b-7b/qlora-q7b-3k-image/`

**Packaged adapter:** resumed step-600 adapter (`adapter_final.zip`)

**Inference prompt:** CoT5 attribute checklist, answer-first

**Results:** devtest CI **0.0360** (500 images); test-phase CI **0.0400** (1,000 images).

**Caveat:** this legacy run resumed at steps 200 and 500 without restoring or skipping
the shuffled dataloader position. The packaged step-600 adapter accumulated 2,400 sample
exposures across restarted sessions, not one clean epoch over all 3,000 unique training
items. The scores are valid for the submitted adapter, but they must not be interpreted
as a controlled 3,000-item data-volume result.

---

### InternVL2 — Cross-Family Zero-Shot Replication

**Motivation.** The Qwen joint-prompt findings could be family-specific. InternVL2-2B and
InternVL2-8B are contemporaries of Qwen2.5-VL-3B and -7B, so re-running the same prompts
isolates model family rather than training-recency. Fine-tuning, hyper-parameter search,
and permutation ensembling are out of scope for this track — zero-shot / prompting only.

**Models:** `OpenGVLab/InternVL2-2B`, `OpenGVLab/InternVL2-8B` (8B under 4-bit NF4).
**Prompts / parser:** identical to the matched Qwen runs (character-for-character joint
prompt and official `evaluate_tf` parser).
**Split:** `devtest` (Codabench 17051). Duration field was −1.0 on scored submissions.

| InternVL run | Matches Qwen | Model | Prompt / scaffold | CI ↓ | Comb ↑ | CFHR ↓ | Q+ ↑ | Q− ↑ |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| Intern-R3 | Run 3 | InternVL2-2B | reason→answer | 0.298 | 0.702 | 0.000 | 0.702 | 0.851 |
| Intern-R5 | Run 5 | InternVL2-2B | answer→reason | 0.232 | 0.768 | 0.000 | 0.768 | 0.884 |
| Intern-R2 | Run 2 | InternVL2-8B | reason→answer | 0.098 | 0.902 | 0.000 | 0.902 | 0.951 |
| **Intern-R4** | **Run 4** | **InternVL2-8B** | **answer→reason (free-form)** | **0.084** | **0.916** | **0.000** | **0.916** | **0.958** |
| **Intern-CoT5** | **CoT5** | **InternVL2-8B** | **Intern-R4 + attribute checklist CoT** | **0.084** | **0.916** | **0.000** | **0.916** | **0.958** |

**Matched Qwen vs InternVL (same prompt, same split):**

| Matched pair | Qwen CI | InternVL CI | InternVL / Qwen error ratio |
|---|:---:|:---:|:---:|
| Run 3 vs Intern-R3 (small, reason-first) | 0.142 | 0.298 | 2.10× |
| Run 5 vs Intern-R5 (small, answer-first) | 0.082 | 0.232 | 2.83× |
| Run 2 vs Intern-R2 (large, reason-first) | 0.092 | 0.098 | 1.07× |
| Run 4 vs Intern-R4 (large, answer-first) | 0.050 | 0.084 | 1.68× |
| CoT5 vs Intern-CoT5 (Run 4 / Intern-R4 + attribute checklist CoT) | 0.042 | 0.084 | **2.00×** |

**Within-family InternVL deltas:**

$$\text{CI(Intern-R3)} - \text{CI(Intern-R5)} = 0.298 - 0.232 = 0.066 \quad (-22.1\% \text{ relative, answer-first @ 2B})$$

$$\text{CI(Intern-R2)} - \text{CI(Intern-R4)} = 0.098 - 0.084 = 0.014 \quad (-14.3\% \text{ relative, answer-first @ 8B})$$

$$\text{CI(Intern-R5)} - \text{CI(Intern-R4)} = 0.232 - 0.084 = 0.148 \quad (-63.8\% \text{ relative, 2B → 8B answer-first})$$

$$\text{CI(Intern-R4)} - \text{CI(Intern-CoT5)} = 0.084 - 0.084 = 0 \quad (0.0\% \text{ relative; CoT5 null on InternVL})$$

**Takeaway:** answer-first and model scale transfer across families, but both levers are
weaker (answer-first) or differently proportioned (scale) than on Qwen, and InternVL never
closes the absolute gap to its Qwen match. The large-model reason-first cell is almost tied
(Intern-R2 0.098 vs Run 2 0.092); answer-first is where Qwen pulls away most (0.050 vs
0.084). **Intern-CoT5** (*Intern-R4 + attribute checklist CoT* — the exact InternVL port of
Qwen *Run 4 + attribute checklist CoT*) ties free-form Intern-R4 exactly at CI 0.084 —
a clean null result: CoT structure that helps Qwen by 16% relative does not move InternVL
at all, and widens the cross-family error ratio from 1.68× (free-form Run 4) to 2.00×
(CoT5). Remaining InternVL CoT ablations (CoT1–CoT4, CoT6 on 8B) live under
`Development/internvl-2b-8b/all-COT-variations-i8b/` and are not yet scored on Codabench.

**Notebooks:** `Development/internvl-2b-8b/` — see that folder's README for Kaggle/T4
quirks (`use_flash_attn=False`, `transformers==4.49.0`, `MAX_TILES` as the resolution knob).

---

## System Comparison

| | R1 | R3 | R2 | R5 | CoT2 | Res1280 | R4 | **FT-3B** | CoT4 | CoT3 | CoT6 | CoT1 | **FT-3B (3k)** | CoT5 | ADE | **FT (2.6k)** | **FT** | **FT (2.3k)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Model | 3B | 3B | 7B | 3B | 7B | 7B | 7B | **3B+LoRA** | 7B | 7B | 7B | 7B | **3B+LoRA** | 7B | 7B×3 | **7B+LoRA** | **7B+LoRA** | **7B+LoRA** |
| Passes | 3 | 1 | 1 | 1 | 1 | 1 | 1 | **1** | 1 | 1 | 1 | 1 | **1** | 1 | 3 | **1** | **1** | **1** |
| Joint | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **✅** | ✅ | ✅ | ✅ | ✅ | **✅** | ✅ | ✅ | **✅** | **✅** | **✅** |
| Ans first | — | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | **✅** | ✅ | ✅ | ✅ | ✅ | **✅** | ✅ | ✅ | **✅** | **✅** | **✅** |
| CoT style | — | free | free | free | elim | free | free | **attr** | devil | rank | socratic | evid | **attr** | attr | free | **attr** | **attr** | **attr** |
| Train items | — | — | — | — | — | — | — | **2,000** | — | — | — | — | **3,000** | — | — | **2,600** | **2,000** | **2,300** |
| Fine-tuned | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** | ❌ | ❌ | ❌ | ❌ | **✅** | ❌ | ❌ | **✅** | **✅** | **✅** |
| Ensemble | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| CI ↓ | 0.257 | 0.142 | 0.092 | 0.082 | 0.056 | 0.054 | 0.050 | **0.050** | 0.048 | 0.046 | 0.046 | 0.044 | **0.044** | 0.042 | 0.042 | **0.034** | **0.032** | **0.028** |
| Comb ↑ | 0.740 | 0.858 | 0.908 | 0.918 | 0.944 | 0.946 | 0.950 | **0.950** | 0.952 | 0.954 | 0.954 | 0.956 | **0.956** | 0.958 | 0.958 | **0.966** | **0.968** | **0.972** |
| CFHR ↓ | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | **0.000** | **0.000** | **0.000** |
| Q+ ↑ | 0.912 | 0.858 | 0.908 | 0.918 | 0.944 | 0.946 | 0.950 | **0.950** | 0.952 | 0.954 | 0.954 | 0.956 | **0.956** | 0.958 | 0.958 | **0.966** | **0.968** | **0.972** |
| Q− ↑ | 0.888 | 0.929 | 0.954 | 0.959 | 0.972 | 0.973 | 0.975 | **0.975** | 0.976 | 0.977 | 0.977 | 0.978 | **0.978** | 0.979 | 0.979 | **0.983** | **0.984** | **0.986** |

### InternVL2 zero-shot matrix (devtest)

| | Intern-R3 | Intern-R5 | Intern-R2 | **Intern-R4** | **Intern-CoT5** | Qwen R3 | Qwen R5 | Qwen R2 | Qwen R4 | Qwen CoT5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Model | 2B | 2B | 8B | **8B** | **8B** | 3B | 3B | 7B | 7B | 7B |
| Base recipe | R3 | R5 | R2 | **R4 free** | **R4 + attr CoT** | R3 | R5 | R2 | R4 free | R4 + attr CoT |
| Joint | ✅ | ✅ | ✅ | **✅** | **✅** | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ans first | ❌ | ✅ | ❌ | **✅** | **✅** | ❌ | ✅ | ❌ | ✅ | ✅ |
| CoT style | free | free | free | **free** | **attr checklist** | free | free | free | free | attr checklist |
| Fine-tuned | ❌ | ❌ | ❌ | **❌** | **❌** | ❌ | ❌ | ❌ | ❌ | ❌ |
| CI ↓ | 0.298 | 0.232 | 0.098 | **0.084** | **0.084** | 0.142 | 0.082 | 0.092 | 0.050 | 0.042 |
| Comb ↑ | 0.702 | 0.768 | 0.902 | **0.916** | **0.916** | 0.858 | 0.918 | 0.908 | 0.950 | 0.958 |
| CFHR ↓ | 0.000 | 0.000 | 0.000 | **0.000** | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Q+ ↑ | 0.702 | 0.768 | 0.902 | **0.916** | **0.916** | 0.858 | 0.918 | 0.908 | 0.950 | 0.958 |
| Q− ↑ | 0.851 | 0.884 | 0.951 | **0.958** | **0.958** | 0.929 | 0.959 | 0.954 | 0.975 | 0.979 |

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
├── logs.txt                        # Codabench score scratchpad (source for new results)
└── Development/
    ├── qwen2p5-3b-7b/              # Main Qwen2.5-VL track (baseline, joint, CoT, QLoRA, …)
    ├── internvl-2b-8b/             # InternVL2 cross-family zero-shot replication
    │   ├── joint-3-i2b/            # Intern-R3: 2B joint reason→answer (CI 0.298)
    │   ├── answer-first-joint-i2b/ # Intern-R5: 2B joint answer→reason (CI 0.232)
    │   ├── joint-3-i8b/            # Intern-R2: 8B joint reason→answer (CI 0.098)
    │   ├── answer-first-joint-i8b/ # Intern-R4: 8B joint answer→reason (CI 0.084)
    │   └── all-COT-variations-i8b/ # Intern-CoT5 = Intern-R4 + attr checklist (CI 0.084, null);
                                    #   CoT1–4,6 still pending
    ├── finetune-qlora-q3b/         # 3B QLoRA-SFT + DPO track (dev split)
    └── permutation-ensembling-q7b/ # ADE majority-vote combiner artifacts
```

Qwen Run 3 / Run 5 notebooks live under `qwen2p5-3b-7b/` with `VLM_MODEL` switched to the
3B or 7B variant as needed. InternVL notebooks are self-contained ports of the same
prompts under `internvl-2b-8b/` (see that folder's README).

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
| RunFT-3B train | same as RunFT train, `VLM_MODEL = Qwen/Qwen2.5-VL-3B-Instruct` | T4×2 | not recorded |
| RunFT-3B infer | same as RunFT infer, `VLM_MODEL = Qwen/Qwen2.5-VL-3B-Instruct` | T4 | not recorded |
| RunFT-3B (3k) infer | `finetune/step-3-qlora-q3b-3k-inference.ipynb` — 3B adapter retrained on the full 3,000-item train set, `SPLIT='devtest'` | T4×2 | not recorded (CI 0.044 via Codabench) |
| RunFT (2.3k) train | same as RunFT train, training subsample increased from 2,000 to 2,300 items | T4×2 | not recorded |
| RunFT (2.3k) infer | same as RunFT infer (`finetune/qlora-infer-final.ipynb`), `SPLIT='devtest'` | T4 | not recorded (CI 0.028 via Codabench, current best) |
| RunFT (2.6k) train | `Development/qwen2p5-3b-7b/qwen-q7b-2p6k-image/resume-from-checkpoint-300.ipynb` (`TRAIN_SUBSAMPLE_N=2600`, `MAX_STEPS=510`) | T4×2 | not recorded |
| RunFT (2.6k) infer | `Development/qwen2p5-3b-7b/qwen-q7b-2p6k-image/inference-qlora-2p6k-q7b-v1.ipynb`, `SPLIT='devtest'` | T4 | not recorded (CI 0.034 via Codabench; non-monotonic vs 2.3k) |
| RunFT (3k legacy) train | `Development/qwen2p5-3b-7b/qlora-q7b-3k-image/step1-qlora-3k-q7b-upto-200ckpt.ipynb` → `step2-continue-from-200chkpoint.ipynb` → `step3-from-chkpoint-500.ipynb` | T4×2 | resumed legacy run; packaged step 600; dataloader position not preserved |
| RunFT (3k legacy) infer | `Development/qwen2p5-3b-7b/qlora-q7b-3k-image/inference-qlora-q7b-frn.ipynb` | T4×2 | devtest CI 0.0360; test-phase CI 0.0400 via `Test/qwen2p5-3b-7b/qlora-q7b-3k-image/` |
| Intern-R3 | `Development/internvl-2b-8b/joint-3-i2b/joint-3-stat-internvl2b.ipynb` | T4 | not recorded (CI 0.298 via Codabench) |
| Intern-R5 | `Development/internvl-2b-8b/answer-first-joint-i2b/answer-first-internvl2b.ipynb` | T4 | not recorded (CI 0.232 via Codabench) |
| Intern-R2 | `Development/internvl-2b-8b/joint-3-i8b/joint-3-stat-internvl8b.ipynb` | T4 | not recorded (CI 0.098 via Codabench) |
| Intern-R4 | `Development/internvl-2b-8b/answer-first-joint-i8b/answer-first-internvl8b.ipynb` | T4 | not recorded (CI 0.084 via Codabench; tied best InternVL zero-shot) |
| Intern-CoT5 (Intern-R4 + attribute checklist CoT) | `Development/internvl-2b-8b/all-COT-variations-i8b/attribute-checklist/cot-attribute-checklist.ipynb` | T4 | not recorded (CI 0.084 via Codabench; ties free-form Intern-R4 — null vs Qwen CoT5) |

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
