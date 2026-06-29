# CLAUDE.md — project memory (auto-loaded each session)

## What this is
Research code for **ImageEval 2026 — Ayn-VQA Task 1b (English)**: image + 3
statements, exactly one is True (grounded), two are culturally-plausible
hallucinations. Predict the True one. Metric: **Contrastive Instability (CI)**,
lower is better. Leaderboard: Codabench 17051.

## Current state
- Model: **Qwen2.5-VL** (3B / 7B), zero-shot, prompt-only. No training yet.
- Best so far: **CI 0.042** (CoT5 single-pass, and Run ADE permutation ensemble)
  — this is the prompt-engineering ceiling. Beating it needs **training**.
- Remaining errors: ~21 dev failures, all type-A cultural traps (texture, intent,
  fine geometry).

## What we're working on now
**QLoRA-SFT fine-tuning on Google Colab (T4).** Plan:
1. Small-scale test: **200 train items**, model **Qwen2.5-VL-3B**.
2. Run our own **baseline (zero-shot)** eval first, then **QLoRA-SFT**, compare CI
   on the SAME held-out dev items.
3. If it works: add CoT5 prompt + ensemble, then replicate on 7B.
- Eval discipline: TRAIN on train split, EVALUATE on dev split (never on train
  items). Same dev subset for baseline and fine-tuned = fair comparison.
- Next training methods after SFT: DPO, then GRPO/RLVR (verifiable reward from
  labels). See LEARNING_ROADMAP.md / PROGRESS.md.

## Repo conventions
- **All new work goes on branch `playground/concept-testing`.** Keep `main` clean
  (it is the teammate's published research).
- `.venv` = local **CPU-only** env (no local GPU). GPU work runs on **Colab/Kaggle T4**.
- Notebooks are delivered as `.ipynb` files for the user to upload to Colab.

## Key files
- `PROGRESS.md` — living tracker: concepts, experiments (done/todo), submissions, paper.
- `LEARNING_ROADMAP.md` — what to learn and why (QLoRA, DPO, RL/GRPO, etc.), with links.
- `Development/permutation-ensembling-q7b/combine_local.py` — local no-GPU ensemble combiner.
- `samples/` — 4 dev example images + statements/labels for problem study.
- `requirements-local.txt` (CPU) / `requirements-kaggle.txt` (GPU).

## Dataset
- HF: **QCRI/AynVQA-ArabicNLP26**, files under `task1b/`:
  `train_en.jsonl` (3,000 labeled), `dev_en.jsonl` (500 labeled),
  `devtest_en.jsonl` / `test` (blind). Public — anonymous download works.
- Each record: `id, image, statements[3], country, category, subcategory, labels[3]`.

## Security
- 🔴 A real HF token is hardcoded in 15 original notebooks — **must be revoked**.
  Never hardcode tokens; use Kaggle Secrets / Colab env vars.

## Environment notes
- Windows + PowerShell. Local Python via miniconda; repo `.venv` for CPU work.
- No local GPU (`nvidia-smi` absent). All fine-tuning/inference on Colab/Kaggle T4.
