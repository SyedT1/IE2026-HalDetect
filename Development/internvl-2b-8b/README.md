# InternVL — cross-family replication

The `../qwen2p5-3b-7b/` experiments are re-run here on **InternVL** so the paper can
report whether the findings (joint prompting, answer-first ordering, CoT structure)
are properties of the *task* or artefacts of the *Qwen family*.

Models: [`OpenGVLab/InternVL2-2B`](https://huggingface.co/OpenGVLab/InternVL2-2B) and
[`OpenGVLab/InternVL2-8B`](https://huggingface.co/OpenGVLab/InternVL2-8B), standing in for
Qwen2.5-VL-3B and -7B respectively.

Both are deliberately from the **InternVL2** generation. `InternVL3_5-8B` was the other
candidate for the large slot and is the stronger model, but it postdates Qwen2.5-VL by a
wide margin — a win there could not be attributed to the model *family* rather than to a
more recent training run. InternVL2 is Qwen2.5-VL's contemporary, so the comparison
isolates what we actually want to measure.

Fine-tuning (QLoRA), hyper-parameter tuning, and permutation ensembling are **out of
scope** — this is the zero-shot / prompting arm only.

---

## Run matrix

Each notebook is self-contained: upload to Kaggle, set the accelerator to **GPU T4**,
Run All. Nothing to install locally, no shared module to import, no dataset to upload —
the split and the images are pulled from Hugging Face at runtime.

Run them **in this order**. The 2B block is cheap and proves the port; the 8B block is
where the paper's headline comparison lives; the CoT block is an ablation on top of run 5,
so there is no point running it until run 5 has landed.

| # | Notebook | Model | Method | Replicates | Qwen CI | Passes | ~Time |
|---|---|---|---|---|---|---|---|
| 1 | [baseline/internvl2b-baseline.ipynb](baseline/internvl2b-baseline.ipynb) | 2B | per-statement True/False | Run 1 | 0.257 | 1500 | 50 min |
| 2 | [joint-3-i2b/joint-3-stat-internvl2b.ipynb](joint-3-i2b/joint-3-stat-internvl2b.ipynb) | 2B | joint, reason-first | Run 3 | 0.142 | 500 | 20 min |
| 3 | [answer-first-joint-i2b/answer-first-internvl2b.ipynb](answer-first-joint-i2b/answer-first-internvl2b.ipynb) | 2B | joint, answer-first | Run 5 | 0.082 | 500 | 20 min |
| 4 | [joint-3-i8b/joint-3-stat-internvl8b.ipynb](joint-3-i8b/joint-3-stat-internvl8b.ipynb) | 8B | joint, reason-first | Run 2 | 0.092 | 500 | 40 min |
| 5 | [answer-first-joint-i8b/answer-first-internvl8b.ipynb](answer-first-joint-i8b/answer-first-internvl8b.ipynb) | 8B | joint, answer-first | Run 4 | 0.050 | 500 | 40 min |
| 6 | [all-COT-variations-i8b/evidence-first/cot-evidence-first.ipynb](all-COT-variations-i8b/evidence-first/cot-evidence-first.ipynb) | 8B | CoT: describe image first | CoT1 | 0.044 | 500 | ~1 h |
| 7 | [all-COT-variations-i8b/elimination/cot-elimination.ipynb](all-COT-variations-i8b/elimination/cot-elimination.ipynb) | 8B | CoT: rule out two, confirm one | CoT2 | 0.056 | 500 | ~1 h |
| 8 | [all-COT-variations-i8b/confidence-ranked/cot-confidence-ranked.ipynb](all-COT-variations-i8b/confidence-ranked/cot-confidence-ranked.ipynb) | 8B | CoT: rank all three | CoT3 | 0.046 | 500 | ~1 h |
| 9 | [all-COT-variations-i8b/devils-advocate/cot-devils-advocate.ipynb](all-COT-variations-i8b/devils-advocate/cot-devils-advocate.ipynb) | 8B | CoT: steelman then rebut | CoT4 | 0.048 | 500 | ~1 h |
| 10 | [all-COT-variations-i8b/attribute-checklist/cot-attribute-checklist.ipynb](all-COT-variations-i8b/attribute-checklist/cot-attribute-checklist.ipynb) | 8B | CoT: colour/form/context | CoT5 | **0.042** | 500 | ~1 h |
| 11 | [all-COT-variations-i8b/socratic/cot-socratic.ipynb](all-COT-variations-i8b/socratic/cot-socratic.ipynb) | 8B | CoT: sub-questions | CoT6 | 0.046 | 500 | ~1 h |

Measured wall-clock, not estimates: ~10 GPU-hours for the suite, against a Kaggle quota of
30 GPU-hours/week. 4-bit dequantisation and eager attention (the T4 is Turing, so no
FlashAttention-2) dominate the cost, not the model size. Run 1 is the
expensive one — it judges each statement separately, so it costs 3× the joint runs for what
Qwen showed is by far the worst method. It exists only to reproduce the Run 1 → Run 3 delta.

Runs 6–11 differ from run 5 by **one string**: the prompt. Same model, same decoding, same
parser. That is what makes the CoT ablation an ablation.

## Results

All eleven runs are complete, on **devtest**, matching the split every published Qwen number
uses. devtest is blind, so **CI comes from Codabench** ([competition 17051](https://www.codabench.org/competitions/17051/)) —
submit each run's `prediction_en.zip` and fill the column in.

The `health` column is computed offline by `score_local.py` and needs no answer key. Gold
always marks exactly one statement True, so a *joint* run must emit exactly one True per item
by construction; anything less means the `Answer: X` parse fell back to per-statement judging
and the run is measuring the wrong thing. **All ten joint runs are at 100%** — the port is
sound end to end, and no result here is a parsing artefact.

| Run | Model | Method | InternVL CI ↓ | Comb ↑ | CFHR ↓ | Q+ ↑ | Q− ↑ | Qwen CI ↓ | health |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Run 1 | InternVL2-2B | baseline, per-statement | _Codabench_ (**≥ 0.374**) | — | — | — | — | 0.257 | 62.6% ⚠ |
| Run 3 | InternVL2-2B | joint, reason-first | **0.298** | 0.702 | 0.000 | 0.702 | 0.851 | 0.142 | 100% |
| Run 5 | InternVL2-2B | joint, answer-first | **0.232** | 0.768 | 0.000 | 0.768 | 0.884 | 0.082 | 100% |
| Run 2 | InternVL2-8B | joint, reason-first | **0.098** | 0.902 | 0.000 | 0.902 | 0.951 | 0.092 | 100% |
| Run 4 | InternVL2-8B | joint, answer-first | **0.084** | 0.916 | 0.000 | 0.916 | 0.958 | 0.050 | 100% |
| CoT1 | InternVL2-8B | evidence-first | _Codabench_ | — | — | — | — | 0.044 | 100% |
| CoT2 | InternVL2-8B | elimination | _Codabench_ | — | — | — | — | 0.056 | 100% |
| CoT3 | InternVL2-8B | confidence-ranked | _Codabench_ | — | — | — | — | 0.046 | 100% |
| CoT4 | InternVL2-8B | devil's advocate | _Codabench_ | — | — | — | — | 0.048 | 100% |
| CoT5 | InternVL2-8B | attribute checklist | _Codabench_ | — | — | — | — | **0.042** | 100% |
| CoT6 | InternVL2-8B | socratic | _Codabench_ | — | — | — | — | 0.046 | 100% |

Codabench scores for Runs 2–5 come from `logs.txt` (duration −1.0 on all four submissions).

### What we already know without Codabench

**Run 4 was additionally run on `dev`** (artifacts kept as `*_DEV`, the only InternVL result
scorable offline). It is the answer to the paper's central question:

| | InternVL2-8B | Qwen2.5-VL-7B |
|---|:---:|:---:|
| CI (dev) | **0.078** | 0.042 |
| Combined accuracy | 0.922 | 0.958 |
| CFHR | 0.000 | 0.000 |
| exactly one True per item | 100% | 100% |

**InternVL2-8B runs at roughly 1.9x Qwen2.5-VL-7B's error rate** under an identical prompt,
parser, decoding setup and split. The run is healthy — the `Answer: X` format held on every
item and CFHR is 0, so it never marks a distractor True after finding the grounded statement.
It simply picks the wrong statement more often. This is a capability gap, not a porting
artefact.

**Run 1 has a floor even though it is blind.** Only 62.6% of its items receive exactly one
`true`, where gold always has exactly one — so combined accuracy is capped at 0.626 and
**CI >= 0.374** whatever Codabench returns, against Qwen Run 1's 0.257. The per-statement
baseline has no structural guarantee of a single True (unlike the joint runs), so this is a
genuine weakness of InternVL2-2B rather than a broken parse.

### Codabench summary (Runs 2–5)

Joint-prompt InternVL is **strictly worse than the matched Qwen cell** on every scored run:

| Matched pair | InternVL CI | Qwen CI | Ratio |
|---|:---:|:---:|:---:|
| Run 3 (small, reason-first) | 0.298 | 0.142 | 2.10× |
| Run 5 (small, answer-first) | 0.232 | 0.082 | 2.83× |
| Run 2 (large, reason-first) | 0.098 | 0.092 | 1.07× |
| Run 4 (large, answer-first) | **0.084** | 0.050 | 1.68× |

Within InternVL, answer-first still helps (2B: 0.298 → 0.232, −22.1%; 8B: 0.098 → 0.084,
−14.3%) and scale 2B → 8B is the dominant lever (answer-first: −63.8%). Best InternVL
zero-shot so far: **Run 4, CI 0.084**. CoT1–CoT6 remain unscored on Codabench.

### Known caveat: Run 3 and Run 5 precision

Both ran at **fp16**; the corresponding Qwen 3B joint runs used **4-bit NF4**. The notebooks
are now fixed to 4-bit, but these two results predate the fix.

The error favours InternVL — higher precision is an advantage Qwen did not get — so the
reported InternVL gap on Runs 3 and 5 is a **lower bound**: InternVL still loses by a wide
margin (2.10× / 2.83× Qwen error) despite the precision advantage. A footnote is enough:
*"InternVL2-2B joint runs used fp16 vs Qwen's 4-bit NF4; the precision difference favours
InternVL, so the reported gap is a lower bound."* The Run 3 → Run 5 delta (reason-first vs
answer-first) is unaffected, since both ran at fp16.

---

## What is held fixed vs Qwen (and what is not)

Held **identical**, so any difference in CI is attributable to the model:

- the prompts, character for character;
- the official `evaluate_tf` parser (lifted verbatim from the Qwen baseline notebook at
  build time by `build_internvl_notebooks.py`, so it cannot drift);
- the scoring arithmetic (CI / combined / CFHR / Q+ / Q−) and the submission zip layout;
- greedy decoding, `max_new_tokens`, and the NF4 4-bit quantisation recipe on the large
  model (the small model runs unquantised, mirroring Qwen 3B).

Necessarily **different**, because the architectures differ:

- **Image resolution.** Qwen caps vision tokens with `max_pixels`; InternVL has no such
  knob and instead splits the image into up to `MAX_TILES` 448×448 tiles plus a
  thumbnail. `MAX_TILES` is the analogue: **12** for the joint/CoT runs (≈ Qwen's
  `1024*28*28`), **6** for the baseline (≈ `512*28*28`).
- **Call surface.** InternVL exposes `model.chat(tokenizer, pixel_values, question, cfg)`
  and requires a leading `<image>` placeholder, rather than Qwen's processor + chat
  template.

## Kaggle / T4 notes

These are the things that break if the model card is followed literally:

- **`use_flash_attn=False`.** The card says `True`; FlashAttention-2 needs Ampere or
  newer and the T4 is Turing. Left on, it fails at load.
- **`torch.float16`, not `bfloat16`.** The T4 has no bf16.
- **`transformers==4.49.0`, pinned, and the pin is load-bearing.** Both models vendor their
  own `InternLM2ForCausalLM` through `trust_remote_code`, and that class never declared
  `GenerationMixin`. From v4.50, `PreTrainedModel` stopped inheriting the mixin — so on any
  newer transformers the language model has no `.generate()` and `model.chat()` dies with
  `AttributeError: 'InternLM2ForCausalLM' object has no attribute 'generate'`. It fails at
  the *first inference call*, i.e. after the model has already loaded cleanly, which makes it
  look like a bug in the inference code rather than a version problem. (Learned the hard way.)

  The model-loading cell also re-attaches `GenerationMixin` if it finds it missing, so a
  drifting pin announces itself at load time rather than 40 minutes into a run.
- **8B is 4-bit.** At fp16 it needs ~16 GB and will not fit a single T4; NF4 brings it to
  ~5.5 GB, so `device_map={"": 0}` keeps everything on one GPU and no sharding is needed.
- **No hardcoded HF token.** The dataset is public, so the notebooks download anonymously.
  If you hit a rate limit, add a read token as a Kaggle Secret named `HF_TOKEN`.

## Per-run workflow

Repeat this for each notebook in the table, one at a time.

**1. Import.** Kaggle → New Notebook → **File → Import Notebook** → upload the `.ipynb`.
Kaggle takes a *copy*; editing the repo file afterwards does not change the Kaggle one, so
after any rebuild you must re-import.

**2. Settings** (right-hand panel):
- **Accelerator → GPU T4** (T4 x2 is fine; the code pins to GPU 0).
- **Internet → On.** Off by default, and needs a phone-verified account. Without it both
  the pip install and the Hugging Face download fail.

**3. Run.** Change nothing in the notebook — it ships on `devtest`, which is the split every
published Qwen number uses. Use **Save Version → Save & Run All (Commit)** so it runs
detached on Kaggle's servers; an interactive session dies when you close the tab.

**4. Collect.** From the finished run's **Output** tab, download two files into that run's
folder, next to its notebook (the same convention the Qwen folders use):
- `prediction_en.zip` — the Codabench submission (`id,statement_index,prediction`).
- `predictions_<RUN_ID>_en.csv` — the same predictions **plus the model's raw reply text**.
  Keep this. It is the only record of what the model actually said, it is what `score_local.py`
  reads, and it is what any error analysis has to read.

**5. Check it is sound.** devtest is blind, so the notebook cannot print a score. Run:

```bash
python score_local.py --all
```

On a devtest run it does a label-free health check — the fraction of items given exactly one
`true`, versus a gold set where exactly one is always true. A **joint** run must hit 100% by
construction; anything less means the `Answer: X` parse is falling back to per-statement
judging and the run is measuring the wrong thing. This catches a broken run without spending
a Codabench submission on it.

**6. Score it.** Submit `prediction_en.zip` to
[Codabench 17051](https://www.codabench.org/competitions/17051/) and put the CI in the results
table. Codabench is the only holder of the devtest answer key.

### Getting a number without Codabench

Rebuild the suite against the labelled split and the notebooks grade themselves, printing
CI / combined / CFHR / Q+ / Q− in their own logs:

```bash
python build_internvl_notebooks.py --all --split dev
```

Useful for validating a run, or for an unattended run that would otherwise report nothing.
But a dev number **cannot go in the results table**: the splits disagree (Run 4 is 0.042 on
dev, 0.050 on devtest), and Qwen's dev CI is only published for Run 4 — there is nothing to
compare the other ten against. Validate on dev, report on devtest.

## Regenerating the notebooks

Edit `build_internvl_notebooks.py`, not the `.ipynb` files — every notebook shares one
cell library, so a fix in one place lands in all eleven.

```bash
python build_internvl_notebooks.py          # the 2 smoke-test notebooks
python build_internvl_notebooks.py --all    # all 11
```

The builder compiles every generated code cell and refuses to write a notebook with a
syntax error in it.
