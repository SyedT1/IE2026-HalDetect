# Experiment results (paper evidence)

Small, version-controlled artifacts for the paper narrative. Large binaries
(model adapters, checkpoints) stay on Google Drive — not here.

## T1b — 3B QLoRA-SFT (first real gain)

| Artifact | Status | Note |
|---|:--:|---|
| `T1b_scaled_trainloss.csv` | ✅ | training loss per 10 steps (loss curve) |
| `baseline_dev.csv` (500 @768px) | ✅ | matched baseline predictions (account 1), CI 0.096 |
| `T1_scaled_results.json` | ✅ | metrics bundle (CI 0.062, delta −0.034, config) |
| `baseline_results.json` | ✅ | baseline metrics (CI 0.096) |
| `T1_scaled_ft_dev.csv` | ✅ | fine-tuned predictions (500 dev), CI 0.062 |
| `T1a_fasttest_baseline_dev_100.csv` | ✅ | earlier 100-item fast-test baseline (CI 0.08) |

## T1b-full — SFT finished to 3 epochs (Kaggle T4×2)

| Artifact | Note |
|---|---|
| `T1bfull_sft_infer_results.json` | CI **0.058**, acc 0.942, baseline 0.096, delta −0.038 |
| `T1bfull_sft_dev_preds.csv` | 500-dev predictions (29 wrong) |
| `T1bfull_sft_adapter.zip` | trained LoRA adapter (full 3 epochs), reusable |

**Progress:** baseline 0.096 → SFT partial (1.6ep) 0.062 → **SFT full (3ep) 0.058**
(−40% rel error). Nearing the 7B CoT5 result (0.042) with a 3B model. Next: DPO.

---

**Headline (T1b partial):** baseline CI 0.096 → fine-tuned 0.062 (500 dev @768px,
1500 train, 1.6/3 epochs). ≈ −35% relative error. See `../../PROGRESS.md` §T1b.

Loss curve reading: drops 3.30 → ~0.25 by step ~50, then flat → the model learns
the task fast; extra epochs add little (consistent with the format-only target).
