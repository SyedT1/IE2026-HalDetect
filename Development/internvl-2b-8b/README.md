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
| 6 | [all-COT-variations-i8b/evidence-first/cot-evidence-first.ipynb](all-COT-variations-i8b/evidence-first/cot-evidence-first.ipynb) | 8B | CoT: describe image first | CoT1 | 0.044 | 500 | 45 min |
| 7 | [all-COT-variations-i8b/elimination/cot-elimination.ipynb](all-COT-variations-i8b/elimination/cot-elimination.ipynb) | 8B | CoT: rule out two, confirm one | CoT2 | 0.056 | 500 | 45 min |
| 8 | [all-COT-variations-i8b/confidence-ranked/cot-confidence-ranked.ipynb](all-COT-variations-i8b/confidence-ranked/cot-confidence-ranked.ipynb) | 8B | CoT: rank all three | CoT3 | 0.046 | 500 | 45 min |
| 9 | [all-COT-variations-i8b/devils-advocate/cot-devils-advocate.ipynb](all-COT-variations-i8b/devils-advocate/cot-devils-advocate.ipynb) | 8B | CoT: steelman then rebut | CoT4 | 0.048 | 500 | 45 min |
| 10 | [all-COT-variations-i8b/attribute-checklist/cot-attribute-checklist.ipynb](all-COT-variations-i8b/attribute-checklist/cot-attribute-checklist.ipynb) | 8B | CoT: colour/form/context | CoT5 | **0.042** | 500 | 45 min |
| 11 | [all-COT-variations-i8b/socratic/cot-socratic.ipynb](all-COT-variations-i8b/socratic/cot-socratic.ipynb) | 8B | CoT: sub-questions | CoT6 | 0.046 | 500 | 45 min |

Total ≈ 7 GPU-hours per split, against a Kaggle quota of 30 GPU-hours/week. Run 1 is the
expensive one — it judges each statement separately, so it costs 3× the joint runs for what
Qwen showed is by far the worst method. It exists only to reproduce the Run 1 → Run 3 delta.

Runs 6–11 differ from run 5 by **one string**: the prompt. Same model, same decoding, same
parser. That is what makes the CoT ablation an ablation.

## Results

Filled in as runs land. Qwen numbers are the devtest figures from the top-level README.

| Run | InternVL CI ↓ | Qwen CI ↓ | Δ |
|---|:---:|:---:|:---:|
| _pending_ | | | |

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

**3. Smoke test.** In the config cell set `MAX_ITEMS = 5`, `SPLIT = "dev"`. Run All, ~6 min
(mostly the model download). Check `unparseable -> defaulted to 'false': 0` — that is the
real signal. Ignore the CI, it is computed over 5 items and means nothing.

**4. Full run.** `MAX_ITEMS = None`, `SPLIT = "dev"`. Use **Save Version → Save & Run All
(Commit)** so it runs detached on Kaggle's servers — an interactive session dies if you
close the tab. The notebook prints CI next to the Qwen reference when it finishes.

**5. Collect.** From the finished run's **Output** tab, download:
- `prediction_en.zip` — the Codabench-format submission (`id,statement_index,prediction`).
- `predictions_<RUN_ID>_en.csv` — same predictions **plus the model's raw reply text**.
  Keep this: it is the only record of what the model actually said, and it is what any
  error analysis has to read.

Commit both into that run's folder, next to its notebook — the same convention the Qwen
folders use. Then add the row to the results table above.

**6. Leaderboard number (optional).** Flip `SPLIT = "devtest"`, Run All again, and submit
the new `prediction_en.zip` to [Codabench 17051](https://www.codabench.org/competitions/17051/).
devtest is blind, so this is the only way to score it — and the Qwen table in the top-level
README is devtest, so this is the number that belongs in the paper. If you keep both, name
the dev artifacts `prediction_en_dev.zip` / `predictions_<RUN_ID>_en_dev.csv` so they do not
collide with the devtest ones.

## Regenerating the notebooks

Edit `build_internvl_notebooks.py`, not the `.ipynb` files — every notebook shares one
cell library, so a fix in one place lands in all eleven.

```bash
python build_internvl_notebooks.py          # the 2 smoke-test notebooks
python build_internvl_notebooks.py --all    # all 11
```

The builder compiles every generated code cell and refuses to write a notebook with a
syntax error in it.
