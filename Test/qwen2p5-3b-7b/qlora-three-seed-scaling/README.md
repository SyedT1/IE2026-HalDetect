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
below as the fourth seed of each completed size.

## Test-split results (1,000 items)

The Task-1b `test` answer key is now public (`QCRI/ImageEval-ArabicNLP26`,
`task1b/test_en.jsonl` carries `labels`), so test CI is recomputed offline from the
committed prediction ZIPs — no Codabench submission needed. Reproduce with:

```powershell
python Development\qwen2p5-3b-7b\qlora-three-seed-scaling\score_test_predictions.py
```

`CI = 1 - (items fully correct) / (items at least partly correct)`, the Codabench 1b
scorer. Every run emits exactly one `true` per item on all 1,000 items, so here
`CI = 1 - combined accuracy` and `CFHR = 0`.

| n | Seed | CI ↓ | Combined Acc ↑ | Q+ Acc ↑ | Q− Acc ↑ | Artifact |
|---:|---:|:---:|:---:|:---:|:---:|---|
| 2,000 | 13 | 0.0390 | 0.9610 | 0.9610 | 0.9805 | `2k/prediction_seed13_2k_q7b.zip` |
| 2,000 | 73 | 0.0390 | 0.9610 | 0.9610 | 0.9805 | `2k/prediction_seed73-2k-q7b.zip` |
| 2,000 | 101 | 0.0410 | 0.9590 | 0.9590 | 0.9795 | `2k/prediction_seed101_2k_q7b.zip` |
| 2,000 | 42 (paper) | 0.0490 | 0.9510 | 0.9510 | 0.9755 | `Test/qwen2p5-3b-7b/qlora-q7b-2k-image/prediction_en.zip` |
| 2,348 | 13 | 0.0430 | 0.9570 | 0.9570 | 0.9785 | `2348/prediction_seed13_2p3k_q7b.zip` |
| 2,348 | 73 | 0.0410 | 0.9590 | 0.9590 | 0.9795 | `2348/prediction_seed73_2p3k_q7b.zip` |
| 2,348 | 101 | **0.0370** | **0.9630** | 0.9630 | 0.9815 | `2348/prediction_seed101_2p3k_q7b.zip` |
| 2,348 | 42 (paper) | 0.0390 | 0.9610 | 0.9610 | 0.9805 | `Test/qwen2p5-3b-7b/qlora-q7b-2p3k-image/prediction_en.zip` |
| 2,600 | 42 (paper) | 0.0350 | 0.9650 | 0.9650 | 0.9825 | `Test/qwen2p5-3b-7b/qlora-q7b-2p6k-image/prediction_en.zip` |
| 3,000 | 42 (paper, legacy) | 0.0400 | 0.9600 | 0.9600 | 0.9800 | `Test/qwen2p5-3b-7b/qlora-q7b-3k-image/prediction_en.zip` |

The four seed-42 rows reproduce the published root-README test CIs exactly, which
validates the offline scorer against the Codabench numbers.

The 2,600 and 3,000 cells have training notebooks only — no adapters and no predictions
yet — so seeds 13/73/101 are unscored at those sizes. The 3,000 seed-42 row is the
*legacy* resumed step-600 adapter, not a clean one-epoch run.

### Seed variance

| n | Seeds 13/73/101 | 4 seeds (incl. 42) | Range (4 seeds) |
|---:|:---:|:---:|:---:|
| 2,000 | 0.0397 ± 0.0012 | 0.0420 ± 0.0048 | 0.039 – 0.049 |
| 2,348 | 0.0403 ± 0.0031 | 0.0400 ± 0.0026 | 0.037 – 0.043 |

(mean ± sample standard deviation)

**Finding: the 2,000 → 2,348 data-scaling effect does not survive reseeding.** On the
paper seed alone the step looks like a clear −0.010 CI win (0.049 → 0.039). Across seeds
13/73/101 the same step is +0.0007 — flat, and in the wrong direction. The per-seed
deltas do not even agree in sign: 13 +0.004, 73 +0.002, 101 −0.004, 42 −0.010.

Seed 42 at n=2,000 (0.049) is the outlier of the eight scored runs; the other three
2,000-item seeds land at 0.039–0.041. The published 2,000 → 2,348 improvement is
therefore mostly an unlucky baseline seed, not a data-volume effect. At 1,000 test items
one flipped item moves CI by 0.001, so the whole effect is ~10 items.

This does not overturn the 2,600 result (CI 0.035), which remains the best single run —
but that number is also a single seed and, on this evidence, carries a seed uncertainty
of roughly ±0.003–0.005. Reseeding 2,600 and 3,000 is the outstanding work.

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
python Development\qwen2p5-3b-7b\qlora-three-seed-scaling\build_training_notebooks.py
```
