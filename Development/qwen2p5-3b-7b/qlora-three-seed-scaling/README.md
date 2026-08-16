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
excluded because it is the historical paper run.

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
