# Gemma-4-E4B — CoT5 cross-family test

Cross-family port of the **best zero-shot Qwen notebook** (CoT5 attribute-checklist,
Qwen2.5-VL-7B, dev **CI 0.042**) onto **`google/gemma-4-E4B-it`** — Google's small
multimodal Gemma 4 (effective ~4.5B, ~8B with embeddings), run 4-bit NF4 to fit one T4.

**Only the backbone changes.** Prompt, answer parser, scoring cells, and CSV/zip save
are byte-identical to the Qwen CoT5 run. So any CI difference is the model, not the
harness — same logic as the InternVL2 cross-family replication.

## What differs from the Qwen notebook

| | Qwen CoT5 | This notebook |
|---|---|---|
| Loader | `Qwen2_5_VLForConditionalGeneration` + `qwen_vl_utils` | `Gemma4ForConditionalGeneration`, image via chat template |
| Resolution knob | `MAX_PIXELS = 1024×28×28` | `IMG_SOFT_TOKENS` ∈ {70,140,280,560,1120}, default **560** |
| transformers | ≥4.49 | **≥5.13** (Gemma4 classes) |
| HF token | ⚠️ was hardcoded | **Kaggle Secret / env `HF_TOKEN`** |

## Run (Kaggle)

1. **Accept the licence** at <https://huggingface.co/google/gemma-4-E4B-it> (Gemma is gated).
2. Add your HF token as a Kaggle **Secret** named `HF_TOKEN`.
3. Upload `cot5-gemma4-e4b.ipynb` → enable **T4** → Run All.
4. `SPLIT='dev'` prints CI locally (compare to Qwen 0.042). `SPLIT='devtest'` →
   download `prediction_CoT5_gemma4_e4b_en.zip` → submit to Codabench 17051.

## Notes

- If OOM: drop `IMG_SOFT_TOKENS` to 280, or set `QUANTIZE=True` (already default).
- Gemma 4 uses variable aspect-ratio images at a fixed soft-token budget — no
  ImageNet normalization. `560` (~1.3M px) is the closest analog to Qwen's ~0.8M px.
- Expectation: this is a *small* model vs Qwen 7B, so CI ≥ 0.042 is likely — that IS
  the finding (does the CoT5 lever transfer to Gemma 4, or fail like it did on InternVL?).
- Fair baseline: to separate "Gemma small" from "CoT5 doesn't transfer", also run a
  free-form (no-checklist) Gemma pass — swap the prompt back to Run 4's plain
  answer-first text. Left as a follow-up.
