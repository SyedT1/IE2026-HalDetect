# Paper fixes — read before writing / revising

Three problems in the current results, in descending order of how badly a reviewer
will punish them. Two are **bugs** (runs that did not do what their label says).
One is **statistical** (claims smaller than the measurement resolution).

None of them touch the paper's real contributions, which are strong and survive
intact. What has to go is a layer of over-claiming on top of them.

**The one-line summary:** we reported a "non-monotonic data-volume curve" that is
actually two broken training runs plus a 3-item difference on a 500-item test set.

---

## Problem 1 — The 2.6k run never finished an epoch (BUG)

**Where:** `Development/qwen2p5-3b-7b/qwen-q7b-2p6k-image/resume-from-checkpoint-300.ipynb`

**Evidence, from the notebook itself:**

```
MAX_STEPS = 510
Step  500/510 |
```

With `GRAD_ACCUM = 4`, 510 optimizer steps consume `510 × 4 = 2,040` training
samples. One epoch over a 2,600-item prefix requires **650** steps. The run
stopped at **78% of one epoch** and **560 of its 2,600 items were never seen.**

Someone on the team already caught this — the fix notebook
(`train-qlora-2p6k-clean-nested.ipynb`) says so in its own header:

> `MAX_STEPS` | hard-coded **510** (only 78% of 2.6k) | **auto** `N // GRAD_ACCUM` → **650**

**Consequence:** "RunFT (2.6k)" is not a 2,600-item run. It trained on 2,040
samples — *fewer than the 2.3k run's 2,348*. Calling its CI a data-volume
*regression* is backwards: it saw **less** data, not more.

**Also confounded:** the cosine LR schedule was sized for 510 steps, so the LR was
fully annealed at the shipped checkpoint. In the other runs it anneals over a
different horizon. The runs do not share an optimisation schedule.

---

## Problem 2 — The 3k run is early-stopped *and selected on dev* (BUG)

**Where:** `Development/qwen2p5-3b-7b/qlora-q7b-3k-image/step3-from-chkpoint-500.ipynb`

**Evidence, verbatim from its output:**

```
Step  700/750 | loss=0.0125 | lr=2.42e-06
  → dev CI=0.0240 ... | best CI so far: 0.0220
  ✗ CI did not improve (best=0.0220 @ step_600) — stopping training now.
Training complete at step 700.
Copied best-CI checkpoint (step_600) → adapter_final
```

And its eval cell loads `task1b/dev_en.jsonl` — **the real 500-item labelled dev
split**.

So the shipped 3k adapter is:
- **step 600 of a 750-step schedule** → `600 × 4 = 2,400` samples seen, **80% of an epoch**;
- **chosen by scanning checkpoints for the best dev CI** → its dev score (0.0220)
  is a *maximum over 7 checkpoints*, not an unbiased estimate;
- trained under a **different protocol** from 2k and 2.3k, which ship the final
  step with no early stopping and no checkpoint selection.

**This run has never been submitted to Codabench**, yet the README says *"The full
3,000-item train set remains untested at 7B."* The artifacts
(`adapter_final.zip`, `prediction_en.zip`) are sitting in the repo.

### What the "data-volume curve" actually is

| Run | Prefix | Steps run | Schedule | **Samples truly seen** | Epoch | Ships | Selected by | devtest CI |
|---|---:|---:|---:|---:|---:|---|---|---:|
| RunFT 2k | 2,000 | 500 | 500 | **2,000** | 1.00 ✅ | final step | — | 0.032 |
| RunFT 2.3k | 2,348 | 587 | 587 | **2,348** | 1.00 ✅ | final step | — | 0.028 |
| RunFT 2.6k | 2,600 | 510 | **650** | **2,040** | 0.78 ❌ | final step | — | 0.034 |
| "RunFT 3k" | 3,000 | 700 | 750 | **2,400** | 0.80 ❌ | **step 600** | **best dev CI** ❌ | unsubmitted |

**Only two of four points share a protocol.** The other two differ in epoch
fraction, LR-anneal state, and (for 3k) model-selection method. There is no curve
here to be non-monotonic. There is an uncontrolled experiment.

---

## Problem 3 — Most of our claims are smaller than our measurement resolution

CI is measured on **500 items**, and for a joint run CI *is* the error rate. So
**one item = 0.002 CI**, and nothing finer than that exists.

Reproduce with `python Development/qwen2p5-3b-7b/dev-eval-adapters/significance.py --bound`.

### Every published CI, as an error count

| Run | CI | **Wrong / 500** | 95% Wilson interval |
|---|---:|---:|---|
| Run 1 (3B baseline) | 0.257 | 128 | [0.220, 0.296] |
| Run 3 (3B reason-first) | 0.142 | 71 | [0.114, 0.175] |
| Run 2 (7B reason-first) | 0.092 | 46 | [0.070, 0.121] |
| Run 5 (3B answer-first) | 0.082 | 41 | [0.061, 0.109] |
| CoT2 (elimination) | 0.056 | 28 | [0.039, 0.080] |
| Res1280 | 0.054 | 27 | [0.037, 0.077] |
| Run 4 (7B answer-first) | 0.050 | 25 | [0.034, 0.073] |
| CoT4 | 0.048 | 24 | [0.032, 0.070] |
| CoT3 | 0.046 | 23 | [0.031, 0.068] |
| CoT6 | 0.046 | 23 | [0.031, 0.068] |
| CoT1 | 0.044 | 22 | [0.029, 0.066] |
| CoT5 | 0.042 | 21 | [0.028, 0.063] |
| Run ADE | 0.042 | 21 | [0.028, 0.063] |
| RunFT 2.6k | 0.034 | 17 | [0.021, 0.054] |
| RunFT 2k | 0.032 | 16 | [0.020, 0.051] |
| RunFT 2.3k | 0.028 | 14 | [0.017, 0.046] |

**All six CoT variants sit between 21 and 24 errors. All three fine-tunes sit
between 14 and 17.** Those are not rankings. Those are ties.

### Which claims can survive — best case

For a paired comparison, the strongest evidence physically available is that every
net error difference is a clean one-directional flip with **zero** offsetting
changes. Real data never does this, so these p-values are a **floor** — the truth
is worse.

| Claim | Error gap | **Best-case p** | Verdict |
|---|---:|---:|---|
| Joint prompting (R1→R3) | 57 | <0.0001 | ✅ can be significant |
| Answer-first @3B (R3→R5) | 30 | <0.0001 | ✅ can be significant |
| Scale 3B→7B (R3→R2) | 25 | <0.0001 | ✅ can be significant |
| Answer-first @7B (R2→R4) | 21 | <0.0001 | ✅ can be significant |
| FT beats best zero-shot (CoT5→FT2.3k) | 7 | 0.0156 | ⚠️ borderline — must run the real test |
| **CoT5 helps (−16%)** | 4 | **0.1250** | ❌ **CANNOT be significant** |
| **CoT1 helps (−12%)** | 3 | **0.2500** | ❌ **CANNOT be significant** |
| **CoT2 hurts (+12%)** | 3 | **0.2500** | ❌ **CANNOT be significant** |
| **Res1280 hurts** | 2 | **0.5000** | ❌ **CANNOT be significant** |
| **CoT5 beats CoT1** | 1 | **1.0000** | ❌ **CANNOT be significant** |
| **FT 2k→2.3k (−12.5%)** | 2 | **0.5000** | ❌ **CANNOT be significant** |
| **FT 2.3k→2.6k (+21.4%)** | 3 | **0.2500** | ❌ **CANNOT be significant** |
| **FT 2k→2.6k** | 1 | **1.0000** | ❌ **CANNOT be significant** |

**Eight of thirteen claims cannot be supported at any possible split of the
items.** Re-running will not save them. Only a larger evaluation set — or dropping
the claim — will.

The percentages are what make this dangerous. **"−16.0%" and "+21.4%" sound
enormous and are 4 items and 3 items.** A reviewer who converts one of those to an
item count will distrust every number in the paper.

---

## What to change — concretely

### `README.md` — Results table

**Add a resolution note directly under the table.** This single paragraph converts
the biggest liability into a credibility signal:

> All metrics are computed on 500 items, so one item corresponds to 0.002 CI and
> differences below ~0.02 CI (10 items) are not resolvable at this sample size. We
> therefore report **tiers** rather than a strict ranking. Where we claim a
> difference, it is verified with a paired McNemar test on per-item predictions.

**Add an error-count column** (`Wrong / 500`) next to CI. Costs nothing, and it
pre-empts the reviewer doing the division themselves and drawing their own
conclusions.

### `README.md` — "The gains decompose cleanly across orthogonal factors"

The word **"cleanly"** must go, and so must most of the rows. Keep only the four
structural factors that clear the resolution limit; move the rest into a single
"below resolution" row. Replacement:

| Factor | Runs | CI reduction | Paired test |
|---|---|---:|---|
| Joint prompting (3B) | Run 1 → Run 3 | −44.7% (128→71 items) | p < 0.001 |
| Answer-first vs reason-first (3B) | Run 3 → Run 5 | −42.3% (71→41) | p < 0.001 |
| Answer-first vs reason-first (7B) | Run 2 → Run 4 | −45.7% (46→25) | p < 0.001 |
| Model scale 3B → 7B | Run 3 → Run 2 | −35.2% (71→46) | p < 0.001 |
| QLoRA fine-tuning vs best zero-shot | CoT5 → RunFT 2.3k | −33.3% (21→14) | *run the test* |
| **CoT scaffold choice (CoT1–CoT6)** | — | **within noise** | n.s. |
| **Training-set size 2k–3k** | — | **within noise** | n.s. |

(Fill the `p` column from the dev run — see "What to run" below.)

### `README.md` — "Key finding (RunFT (2.6k))"

**Delete this section entirely.** Every sentence in it is downstream of the bug.
Specifically these must not appear anywhere:

- *"the 7B data-volume curve is **non-monotonic**"*
- *"a 21.4% relative *regression*"*
- *"The sweet spot among the three tested subsample sizes is therefore **2,300 items**, not 'more is better.'"*

Replace with an honest saturation finding (see "The story that actually works").

### `README.md` — "Key finding (RunFT (2.3k))"

Remove *"7B may be more data-efficient per additional labelled item than 3B"* — it
rests on a 2-item difference.

### `README.md` — CoT Ablation Summary

Keep the table (the numbers are real measurements), but **remove the ranking
narrative.** The current text says structured-per-attribute CoTs "outperform"
holistic ones and that elimination "actively hurts." Neither is supported: the six
variants span 21–24 errors.

Replace the **Pattern** paragraph with:

> All six CoT scaffolds land within 21–24 errors of 500 (CI 0.042–0.048; CoT2 at
> 28). No pairwise difference among them reaches significance under a paired
> McNemar test. We therefore report the CoT ablation as a **negative result**: once
> the answer-first joint format is in place, the specific reasoning scaffold does
> not measurably change CI at 7B scale. The apparent ordering in the table is
> within sampling noise.

**This is a better paper result than the fake ranking.** "We tried six CoT designs
and the choice does not matter" is a real, useful, falsifiable finding. "Attribute
checklists beat Socratic questioning by 2 items" is not.

### `README.md` — Negative Results table

`Res1280` (27 vs 25 errors) and `CoT2` (28 vs 25) are listed as things that "hurt."
Two and three items. Re-file both as **"no measurable difference"**, not as
regressions.

### `README.md` — Reproduction table

The 3k row must disclose the protocol difference, or the run must be dropped:

> RunFT-7B (3k): early-stopped at step 700/750; the shipped adapter is the
> **best-dev-CI checkpoint (step 600, 2,400 samples seen)**, not the final step.
> This differs from the 2k / 2.3k protocol and the two are not directly comparable.

### Everywhere — relative percentages on small differences

Any `−16.0%` / `+21.4%` / `−12.5%` computed from a gap of fewer than ~10 items
must be either removed or written as **"N→M of 500 items (not significant)."**

---

## The story that actually works

The paper does not need the fake precision. Strip it and four **strong, defensible**
contributions remain — and they are what the paper is really about:

**1. The prompting ladder is large and real.**
Per-statement → joint → answer-first, at both scales. 128 → 71 → 41 errors at 3B.
Every step clears significance by a wide margin. This is the paper's backbone.

**2. Prompt order beats model scale.**
Run 5 (3B, answer-first, 41 errors) **beats** Run 2 (7B, reason-first, 46 errors).
A prompt-ordering change outperforms doubling the parameters. Cheap, surprising,
significant, and it supports the M²CQA hypothesis you already cite.

**3. Fine-tuning breaks the zero-shot ceiling — and then saturates immediately.**
All three fine-tunes (14–17 errors) beat all zero-shot systems (21+ errors). But
between 2,000 and 3,000 training items, **nothing further happens.** And there is
direct mechanistic evidence for *why*, already in your logs: training loss collapses
from **0.041** (2k) to **0.0125** (3k) while dev CI does not improve. The model is
**memorising, not generalising.** A teammate wrote exactly this in the 3k notebook:

> *"train loss kept dropping well below the earlier 2k run's loss, but devtest CI got worse, not better"*

That is a far more interesting claim than a spurious optimum at 2,300, it is
supported by the loss curves rather than by 3 items of test noise, and it comes with
a real implication: **the bottleneck is data diversity, not data volume.** ~2,000
items already saturate what LoRA-rank-8 on a frozen-vision 7B can extract from this
distribution.

**4. Cross-family: the levers transfer, the gains do not.**
InternVL2 vs Qwen2.5-VL under an identical prompt/parser/split. The big gaps here
are **17+ items** and comfortably significant. Answer-first and scale transfer;
absolute error does not. And the CoT5 attribute-checklist scaffold — which
"helps" Qwen — is an **exact null** on InternVL2-8B: 42 errors → 42 errors, with 9
of 500 items flipping and cancelling out.

Note the pleasing consistency: **CoT5's Qwen "gain" is not significant, and its
InternVL transfer is exactly zero.** Both facts point the same way — the CoT
scaffold does not do anything. Reported honestly, these corroborate each other.
Reported as "−16% on Qwen, 0% on InternVL," they look like an unexplained mystery.

---

## What to run — priority order

### 1. Submit the 3k `prediction_en.zip` to Codabench — **zero GPU, do it today**
`Development/qwen2p5-3b-7b/qlora-q7b-3k-image/prediction_en.zip` exists and has
never been scored. devtest was not used for its checkpoint selection, so the score
is unbiased *for that (dev-selected) system*. Free data point.

### 2. Run `dev-eval-4-adapters.ipynb` — **~4 GPU-hours, highest value**
`Development/qwen2p5-3b-7b/dev-eval-adapters/dev-eval-4-adapters.ipynb`

Import to Kaggle → **GPU T4 x2**, **Internet ON** → Save & Run All. Nothing to
upload; it pulls all four adapters from GitHub itself.

It scores all four fine-tunes on the **labelled dev split**, which gives what
devtest structurally cannot:
- **per-item correctness** → a real paired McNemar test;
- an **independent second measurement** of each run.

If the devtest ordering (2.3k < 2k < 2.6k) fails to reproduce on dev, that is
*direct experimental proof* the ordering was noise — far stronger than an argument,
and it lets you retract the curve with evidence.

It also settles the one borderline claim that actually matters:
**does fine-tuning significantly beat CoT5?** (best-case p = 0.0156, so it hinges
on the real b/c split.) That claim is the paper's headline. It should not rest on
an unverified 7-item gap.

Download the CSVs into `dev-eval-adapters/`, then run `python significance.py`.

### 3. Do **not** re-train 2.6k or 3k
The 3k log shows **~137 s per optimizer step**. A clean 650-step run is ≈25 GPU-hours
— three Kaggle sessions with checkpoint/resume, and ~6 sessions for both. After all
that, the points will *still* be within noise of each other (they are 2–3 items
apart). You would spend two weeks of quota to draw a flat line.

Retract the curve instead. It costs nothing and is more honest.

### 4. If GPU frees up later, in this order
- **Seed variance on the winner.** Retrain 2.3k with 2–3 seeds, report **mean ± std**.
  A single-seed "best system" claim is the other thing reviewers reliably attack. If
  seed-to-seed std is ±0.006, that alone explains the entire "curve" — and saying so
  with data is a strong result.
- **Evaluate on the 1,000-item `test` split** in the final phase. Doubling *n* is
  the only thing that actually raises resolution. It shrinks the interval by ~30%
  and would make a 7-item gap testable.

---

## Files added

| File | Purpose |
|---|---|
| `Development/qwen2p5-3b-7b/dev-eval-adapters/dev-eval-4-adapters.ipynb` | Score all 4 QLoRA adapters on labelled dev; prints Wilson + paired McNemar |
| `Development/qwen2p5-3b-7b/dev-eval-adapters/build_dev_eval.py` | Builder for the above (edit this, not the `.ipynb`) |
| `Development/qwen2p5-3b-7b/dev-eval-adapters/significance.py` | `--bound` audits published CIs with no new data; default mode scores the dev CSVs |
