# Experiment results (paper evidence)

Small, version-controlled artifacts for the paper narrative. Large binaries
(model adapters, checkpoints) stay on Google Drive — not here.

## T1b — 3B QLoRA-SFT (first real gain)

| Artifact | Status | Note |
|---|:--:|---|
| `T1b_scaled_trainloss.csv` | ✅ here | training loss per 10 steps (loss curve) |
| `baseline_dev.csv` (500 @768px) | ⬜ to add | matched baseline predictions (account 1) |
| `T1_scaled_ft_dev.csv` | ⬜ to add | fine-tuned predictions (500 dev) |
| `T1_scaled_results.json` | ⬜ to add | metrics bundle (CI/delta/config) |
| `baseline_results.json` | ⬜ to add | baseline metrics |

**Headline:** baseline CI 0.096 → fine-tuned 0.062 (500 dev @768px, 1500 train,
1.6/3 epochs). ≈ −35% relative error. See `../../PROGRESS.md` §T1b.

Loss curve reading: drops 3.30 → ~0.25 by step ~50, then flat → the model learns
the task fast; extra epochs add little (consistent with the format-only target).
