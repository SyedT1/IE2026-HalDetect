# QLoRA three-seed, four-size training matrix

This directory contains 12 standalone Kaggle notebooks for a controlled Qwen2.5-VL-7B
QLoRA scaling experiment. They train adapters only; test inference is intentionally
separate.

## Experiment matrix

| Folder | Training examples | Optimizer steps | Seed notebooks |
|---|---:|---:|---|
| `2k/` | 2,000 | 500 | 13, 73, 101 |
| `2348/` | 2,348 | 587 | 13, 73, 101 |
| `2600/` | 2,600 | 650 | 13, 73, 101 |
| `3000/` | 3,000 | 750 | 13, 73, 101 |

All runs use one clean epoch, batch size 1, and gradient accumulation 4. Seed 42 is
excluded from the matrix because it is the historical paper run; it is still reported
below as the fourth seed of each size.

## Test-split results (1,000 items)

The Task-1b `test` answer key is now public (`QCRI/ImageEval-ArabicNLP26`,
`task1b/test_en.jsonl` carries `labels`), so test CI is recomputed offline from the
committed prediction ZIPs — no Codabench submission needed. Reproduce with:

```powershell
python Test\qwen2p5-3b-7b\qlora-three-seed-scaling\score_test_predictions.py
```

`CI = 1 - (items fully correct) / (items at least partly correct)`, the Codabench 1b
scorer. Every run emits exactly one `true` per item on all 1,000 items, so here
`CI = 1 - combined accuracy` and `CFHR = 0`.

### Full matrix — test CI by training size and seed

| n | Seed 13 | Seed 73 | Seed 101 | Seed 42 (submitted) | Fresh mean ± sd | 4-seed mean ± sd |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2,000 | 0.0390 | 0.0390 | 0.0410 | 0.0490 | 0.0397 ± 0.0012 | 0.0420 ± 0.0048 |
| 2,348 | 0.0430 | 0.0410 | **0.0370** | 0.0390 | 0.0403 ± 0.0031 | 0.0400 ± 0.0026 |
| 2,600 | 0.0380 | 0.0440 | 0.0400 | **0.0350** | 0.0407 ± 0.0031 | 0.0393 ± 0.0039 |
| 3,000 | 0.0380 | 0.0450 | 0.0390 | 0.0400 | 0.0407 ± 0.0038 | 0.0405 ± 0.0031 |

"Fresh" = seeds 13/73/101 only. All 16 runs: mean 0.0404, sd 0.0034, range 0.035–0.049.

Per-run detail (combined accuracy = 1 − CI, Q+ = combined, Q− = 1 − CI/2, CFHR = 0 for
every row):

| n | Seed | CI ↓ | Artifact |
|---:|---:|:---:|---|
| 2,000 | 13 / 73 / 101 | 0.0390 / 0.0390 / 0.0410 | `2k/prediction_seed{13,73,101}_2k_q7b.zip` |
| 2,000 | 42 | 0.0490 | `../qlora-q7b-2k-image/prediction_en.zip` |
| 2,348 | 13 / 73 / 101 | 0.0430 / 0.0410 / 0.0370 | `2348/prediction_seed{13,73,101}_2p3k_q7b.zip` |
| 2,348 | 42 | 0.0390 | `../qlora-q7b-2p3k-image/prediction_en.zip` |
| 2,600 | 13 / 73 / 101 | 0.0380 / 0.0440 / 0.0400 | `2600/prediction_q7b_*_seed*_en.zip` |
| 2,600 | 42 | 0.0350 | `../qlora-q7b-2p6k-image/prediction_en.zip` |
| 3,000 | 13 / 73 / 101 | 0.0380 / 0.0450 / 0.0390 | run scores; prediction archives not committed |
| 3,000 | 42 (legacy) | 0.0400 | `../qlora-q7b-3k-image/prediction_en.zip` |

The four seed-42 rows reproduce the published root-README test CIs exactly, which
validates the offline scorer against the Codabench numbers. The 3,000-item fresh-seed
CIs are recorded from their runs; those prediction archives are not in the repo, so
`score_test_predictions.py` covers the other 13 runs.
The 3,000 seed-42 row is the *legacy* resumed step-600 adapter, not a clean one-epoch run.

### Finding: the data-scaling curve is entirely a seed artifact

Fresh-seed means are flat across the whole size axis — 0.0397, 0.0403, 0.0407, 0.0407 —
a total spread of 0.0010, i.e. **one test item**. On seed 42 alone the same axis spans
0.049 → 0.035, i.e. **fourteen items**, and reads as a clean monotone improvement up to
2,600 items.

Per-seed 2,000 → 2,600 deltas do not agree in sign or magnitude: 13 −0.001, 73 +0.005,
101 −0.001, 42 −0.014. Each seed also prefers a different training size: seed 13 is best
at 2,600, seed 73 at 2,000, seed 101 at 2,348, seed 42 at 2,600.

The submitted 2,600 run (0.035) is the best of all 16 runs, but the three fresh seeds at
that same size average 0.0407 — slightly *worse* than the 2,000-item fresh mean. The
published "more data helps up to 2,600" conclusion is therefore not reproducible: it is
one fortunate seed at 2,600 combined with one unfortunate seed at 2,000.

### Seed disagreement

Adapters that differ **only** in training seed disagree on 16–32 of 1,000 predictions:

| n | 13–73 | 13–101 | 13–42 | 73–101 | 73–42 | 101–42 |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2,000 | 17 | 16 | 24 | 23 | 22 | 28 |
| 2,348 | 22 | 19 | 23 | 18 | 23 | 23 |
| 2,600 | 23 | 32 | 24 | 22 | 20 | 24 |

At n=2,000, seeds 13 and 73 both score CI 0.039 yet disagree on 17 items — identical
scores, different systems. Across the 12 archived runs, 92.0% of items are correct
everywhere, 1.8% nowhere, and **6.2% flip with the seed**; every size difference in the
table above lives inside that 6.2%. An oracle over the 12 runs would reach CI 0.018.

## Controlled data design

Dataset membership does not change with the training seed. A single deterministic
seed-42 permutation of the 3,000 training items defines nested prefixes:

```text
2,000 ⊂ 2,348 ⊂ 2,600 ⊂ 3,000
```

Seeds 13, 73, and 101 vary LoRA initialization, dataloader order, dropout, worker RNG,
NumPy, PyTorch, and CUDA RNG. Consequently, across-seed variation estimates training
randomness rather than different sampled datasets.

## Kaggle execution

1. Upload one notebook to Kaggle and enable a dual-T4 GPU runtime.
2. If authentication is needed, add `HF_TOKEN` through Kaggle Secrets. Never paste a
   token into a notebook.
3. Run all cells. Each notebook trains only its declared seed/size combination.
4. Download the final ZIP from its unique `/kaggle/working/qlora_q7b_n<size>_seed<seed>/`
   directory.
5. Preserve the ZIP and manifest, then attach the ZIP as a Kaggle dataset for test
   inference.

Each ZIP contains `adapter_final/`, `manifest.json`, and `training_log.csv`. The manifest
records the selected IDs and their SHA-256 hash, initialization hash, package versions,
and complete training configuration.

## Checkpoint and resume

Every 100 optimizer steps, and again at the final non-100-aligned step, the notebook
writes a two-slot atomic rolling checkpoint. It includes:

- LoRA adapter weights and processor configuration;
- AdamW optimizer state;
- cosine scheduler state;
- Python, NumPy, CPU PyTorch, and every CUDA RNG state;
- completed optimizer and microbatch counts;
- cumulative loss counters, elapsed time, and training log;
- seed, subset hash, recipe, and package-version compatibility checks.

Rerunning the notebook in the same Kaggle runtime automatically restores
`checkpoints/latest/` and continues at the next unseen example. It does not restart the
epoch or repeat prior optimizer updates.

The same checkpoint is also written as
`resume_checkpoint_n<size>_seed<seed>.zip`. For a fresh Kaggle session:

1. Download/persist the latest resume ZIP before the old session is discarded.
2. Upload it as a private Kaggle Dataset and attach that dataset to the new notebook.
3. Set `EXTERNAL_CHECKPOINT_PATH` in the configuration cell to either the attached ZIP
   itself or its extracted directory, for example
   `Path('/kaggle/input/q7b-n2348-seed73-checkpoint/resume_checkpoint.zip')`.
4. Run all cells. The notebook verifies the seed, size, subset hash, step budget, and
   package versions before restoring.

The rolling checkpoint ZIP is for resuming training. The final adapter ZIP is the
smaller artifact intended for inference.

## Shared recipe

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Quantization: 4-bit NF4 with double quantization
- Vision tower: frozen
- LoRA: rank 8, alpha 16, dropout 0.05
- Targets: q/k/v/o and gate/up/down projections in language layers
- Learning rate: `2e-4`, cosine schedule, 5% warmup
- Sequence length: 1,280
- Training image budget: `256 × 28 × 28`
- Target: first-line `Answer: X` only

## Regeneration

The notebooks share one canonical builder to prevent configuration drift:

```powershell
python Test\qwen2p5-3b-7b\qlora-three-seed-scaling\build_training_notebooks.py
```
