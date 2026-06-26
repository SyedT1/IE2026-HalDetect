# Learning & Research Roadmap — IE2026 Task 1b Hallucination Detection

**Task recap:** image + 3 statements, exactly one True (grounded), two False
(culturally-plausible hallucinations). Predict the True one. Metric: Contrastive
Instability (CI), lower = better. Leaderboard: Codabench 17051.

**Where we stand:** Qwen2.5-VL-7B, zero-shot, prompt-only. Best CI = **0.042**
(CoT5 single-pass / Run ADE ensemble). This is the *prompt-engineering ceiling* —
going lower needs **training**. Remaining 21 dev errors = fine cultural traps
(texture, intent, fine geometry).

Everything below is ordered by **likely payoff for this task**. Each item:
what it is → why for THIS task → what to learn → tools/papers.

---

## A. Supervised Fine-Tuning (SFT) — the main next step

Teach the model on our **3,000 labeled train items** so it learns the task's
specific traps instead of guessing from priors.

### A1. LoRA (Low-Rank Adaptation)
- **What:** freeze the big model, train tiny "adapter" matrices injected into
  attention layers. ~0.1–1% of params trained.
- **Why here:** full fine-tune of 7B won't fit a T4. LoRA does.
- **Learn:** rank `r`, `alpha`, target modules, merge-after-train.
- **Tools:** HuggingFace `peft`, `trl` (SFTTrainer).

### A2. QLoRA (Quantized LoRA) ⭐ start here
- **What:** LoRA on top of a **4-bit (NF4) quantized** base model. Same 4-bit
  trick already used at inference, now during training.
- **Why here:** lets us actually fine-tune Qwen2.5-VL-7B on a free T4. Cheapest
  path to beat 0.042.
- **Learn:** NF4, double quantization, `bitsandbytes`, paged optimizers,
  gradient checkpointing.
- **Tools:** `peft` + `bitsandbytes` + `trl`. Plan: QLoRA-SFT the 7B on train,
  validate on dev, watch for overfit.

**Paper angle:** report SFT vs prompt-only ablation; show which error sub-types
(texture/intent/geometry) fine-tuning actually fixes.

---

## B. Preference / Alignment Tuning — squeeze out remaining hallucinations

After SFT, teach the model to *prefer* the grounded answer over the plausible
hallucination. Our dataset is **already contrastive** (1 true vs 2 false) →
natural preference pairs.

### B1. DPO (Direct Preference Optimization) ⭐
- **What:** train on (chosen, rejected) pairs directly, no reward model.
- **Why here:** each item gives ready-made pairs: true statement = chosen,
  a hallucinated one = rejected. Directly targets the failure mode.
- **Learn:** the DPO loss, reference model, `beta`.
- **Tools:** `trl` DPOTrainer.

### B2. Variants to know (lighter / alternatives)
- **ORPO** — combines SFT + preference in one stage (no reference model).
- **KTO** — needs only good/bad labels, not paired. Useful if pairing is messy.
- **RLHF/PPO** — classic but heavier; usually overkill here.

**Paper angle:** "SFT then DPO" is a strong, publishable recipe for grounding.

---

## C. Test-Time Methods — no training, but extra compute

### C1. Self-consistency / ensembling
- **What:** run the model several ways, majority vote.
- **Status:** DONE — Latin-square permutation ensemble (Run ADE), CI 0.042.
- **Learn:** voting, position-bias cancellation.

### C2. Visual Contrastive Decoding (VCD) / DoLa
- **What:** contrast logits (clean vs distorted image, or early vs late layers)
  to suppress language-prior hallucinations at decoding time.
- **Status:** DoLa TRIED → **failed** here (broke answer-first format).
  VCD (image-distortion variant) not yet tried — possible angle.
- **Learn:** contrastive decoding mechanics, why it can clash with format
  constraints.

### C3. Retrieval / external knowledge (RAG) — untried idea
- **What:** fetch Arab-cultural facts (e.g., flag geometry, regional dress) and
  feed as context.
- **Why here:** directly attacks the "fine cultural fact" error bucket.
- **Learn:** multimodal retrieval, grounding retrieved text to the image.

---

## D. Data Strategy — cheap wins, paper-worthy

- **Hard negatives:** the 2 false statements ARE hard negatives. Mining/curating
  more sharpens contrastive training.
- **Augmentation:** crop/zoom on the salient region (texture, flag) before
  asking — attacks the perception bottleneck.
- **Error-driven:** all 21 failures are type-A cultural traps; build targeted
  train subsets for them.

---

## E. Evaluation Discipline — don't fool yourself (critical for a paper)

- Understand **CI** exactly (combined per-item correctness, lower better) plus
  CFHR, Q+, Q−.
- Always validate on **dev** (labeled) before submitting blind devtest/test.
- Watch **overfitting**: fine-tuning can memorize dev. Hold out a clean split.
- Keep **clean ablations** (change one thing at a time) — the repo already does
  this well; match that rigor.

---

## Suggested learning order (fastest path to results)

```
1. How VLMs work + why they hallucinate   (foundation — Section "core concepts")
2. QLoRA-SFT on the 3,000 train items      (A2)  ← biggest expected payoff
3. DPO on contrastive pairs                (B1)  ← stack on top of SFT
4. (optional) VCD / RAG / region-crop      (C2, C3, D)  ← novelty for the paper
5. Rigorous eval + ablations throughout    (E)
```

---

## Key references (task-relevant)

- **QLoRA** — Dettmers et al., 2023 (4-bit fine-tuning).
- **LoRA** — Hu et al., 2021.
- **DPO** — Rafailov et al., 2023.
- **VCD** — Leng et al., 2023 (visual contrastive decoding for VLM hallucination).
- **DoLa** — Chuang et al., 2023 (layer contrastive decoding).
- **POPE / CHAIR** — standard VLM hallucination benchmarks (concepts to know).
- **Qwen2.5-VL** technical report — the model we use.
- M²CQA (arXiv:2602.05437) — cited in repo for answer-first robustness.

> Toolchain across all training work: HuggingFace `transformers` + `peft` +
> `trl` + `bitsandbytes`, on Kaggle T4. CPU-only steps (scoring, combining)
> run locally via `.venv` (see `requirements-local.txt`).
