# Progress Tracker — IE2026 Task 1b Hallucination Detection

> Living file. Update status as we go. Legend: ✅ done · 🟡 in progress · ⬜ todo · ❌ tried, failed/negative · ❓ unknown/verify

**Task:** image + 3 statements, pick the 1 grounded (True). Metric: **CI** (lower better).
**Current best:** CI **0.042** — CoT5 (single-pass) & Run ADE (ensemble). Prompt-only ceiling.
**Target:** beat 0.042 via training; submit blind splits; write the paper.

---

## 0. Infra & Admin

| Task | Status | Note |
|---|:--:|---|
| Clone repo | ✅ | `e:\experiment\IE2026-HalDetect` |
| Local CPU env (`.venv`) | ✅ | `requirements-local.txt` |
| Branch `playground/concept-testing` | ✅ | all new work isolated from `main` |
| Reproduce ensemble combiner locally | ✅ | `combine_local.py`, 449/49/2 match |
| 🔴 **Revoke leaked HF token** | ⬜ | hardcoded in 15 notebooks — security risk |
| Move tokens to Kaggle Secrets | ⬜ | replace `login(token=...)` calls |
| GPU env on Kaggle (T4) | ✅ | notebooks pip-install inline |

---

## 1. Core Concepts to Learn (foundation)

Order = learn top-down. Tick when comfortable.

| # | Topic | Status | Link |
|---|---|:--:|---|
| 1 | How VLMs work (ViT/CLIP encoder → projector → LLM decoder) | ⬜ | [CLIP](https://arxiv.org/abs/2103.00020) · [LLaVA](https://arxiv.org/abs/2304.08485) |
| 2 | Why VLMs hallucinate (language prior, object/attribute/relation, co-occurrence) | ⬜ | [VLM Hallucination Survey](https://arxiv.org/abs/2402.00253) |
| 3 | Decoding: greedy/sampling, contrastive decoding | ⬜ | [VCD](https://arxiv.org/abs/2311.16922) · [DoLa](https://arxiv.org/abs/2309.03883) |
| 4 | Quantization (4-bit NF4) + PEFT basics | ⬜ | [bitsandbytes docs](https://huggingface.co/docs/bitsandbytes) · [PEFT docs](https://huggingface.co/docs/peft) |
| 5 | LoRA / QLoRA fine-tuning | ⬜ | [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) |
| 6 | Alignment: RLHF → DPO and variants | ⬜ | [DPO](https://arxiv.org/abs/2305.18290) · [ORPO](https://arxiv.org/abs/2403.07691) · [KTO](https://arxiv.org/abs/2402.01306) |
| 7 | Hallucination benchmarks (POPE, CHAIR) + our CI metric | ⬜ | [POPE](https://arxiv.org/abs/2305.10355) · [CHAIR](https://arxiv.org/abs/1809.02156) |
| 8 | Grounding: hard negatives & contrastive pairs (our data shape) | ⬜ | (see DPO + survey above) |
| 9 | Our model internals | ⬜ | [Qwen2.5-VL report](https://arxiv.org/abs/2502.13923) |

**Training toolchain:** [transformers](https://huggingface.co/docs/transformers) · [peft](https://huggingface.co/docs/peft) · [trl](https://huggingface.co/docs/trl) · [bitsandbytes](https://huggingface.co/docs/bitsandbytes)

---

## 2. Experiments — DONE (from repo)

| Run | Method | CI ↓ | Status |
|---|---|:--:|:--:|
| Run 1 | Baseline, per-statement, 3B | 0.257 | ✅ |
| Run 3 | Joint prompt, reason→answer, 3B | 0.142 | ✅ |
| Run 2 | Joint prompt, reason→answer, 7B | 0.092 | ✅ |
| Run 5 | Answer-first, 3B | 0.082 | ✅ |
| Run 4 | Answer-first, 7B (base system) | 0.050 | ✅ |
| CoT2 | Elimination CoT | 0.056 | ❌ negative |
| CoT4 | Devil's advocate CoT | 0.048 | ✅ |
| CoT3 | Confidence-ranked CoT | 0.046 | ✅ |
| CoT6 | Socratic CoT | 0.046 | ✅ |
| CoT1 | Evidence-first CoT | 0.044 | ✅ |
| **CoT5** | **Attribute-checklist CoT** | **0.042** | ✅ **best single-pass** |
| **Run ADE** | **Latin-square perm ensemble (A+D+E)** | **0.042** | ✅ **best overall** |
| Res1280 (hp1) | MAX_PIXELS 1280 | 0.054 | ❌ negative |
| DoLa | Layer contrastive decoding | 0.132 | ❌ negative |
| Caption→verify | Two-stage cascade | 0.084 | ❌ negative |
| Cultural hint (A6) | Cultural grounding prompt | — | ❓ inconclusive (wrong split) |

### Hyperparameter notebooks — results NOT in README (verify)
| Notebook | What | Status |
|---|---|:--:|
| hp2_beam_search_4 | beam search (4) | ❓ verify CI |
| hp3_diverse_beam | diverse beam | ❓ verify CI |
| hp4_topp_temperature | top-p / temperature | ❓ verify CI |
| hp5_rep_penalty | repetition penalty | ❓ verify CI |
| hp6_qwen3vl_8b | **alt model: Qwen3-VL-8B** | ❓ verify CI |
| hp7_combined | combined settings | ❓ verify CI |

---

## 3. Experiments — TODO (the research plan)

Ordered by expected payoff. This is where new work goes.

| # | Experiment | Goal | Status | Depends on |
|---|---|---|:--:|---|
| T1 | **QLoRA-SFT** Qwen2.5-VL-7B on 3,000 train | beat CI 0.042 | ⬜ | concepts 4–5 |
| T2 | **DPO** on contrastive (true vs false) pairs | cut residual hallucination | ⬜ | T1 |
| T3 | ORPO / KTO alternative to T2 | cheaper preference tuning | ⬜ | concept 6 |
| T4 | **VCD** (image-distortion contrastive decoding) | suppress language prior, no training | ⬜ | concept 3 |
| T5 | **RAG**: inject Arab-cultural facts | fix fine-factual errors (flag/dress) | ⬜ | concept 1 |
| T6 | Region crop/zoom augmentation | fix texture/geometry perception | ⬜ | — |
| T7 | Alt-VLM comparison (InternVL, Qwen3-VL) | stronger backbone | 🟡 | hp6 partial |
| T8 | GNN scene-graph grounding | optional paper novelty | ⬜ | concepts 1–2 |
| T9 | Verify/score hp2–hp7 notebooks | fill the ❓ gaps above | ⬜ | local env |

---

## 4. Submissions

| Split | Labels | Use | Status |
|---|:--:|---|:--:|
| dev (500) | ✅ | local validation | ✅ scored |
| devtest (500) | ❌ blind | dev-phase leaderboard | 🟡 some submitted |
| test (1000) | ❌ blind | final ranking | ⬜ |

---

## 5. Paper Checklist

| Section | Status | Note |
|---|:--:|---|
| Problem / motivation | ⬜ | cultural hallucination, contrastive setup |
| Related work | ⬜ | VLM hallucination, decoding, PEFT, DPO |
| Method | 🟡 | prompt ladder done; training methods pending |
| Experiments & ablations | 🟡 | strong prompt ablations done; add training |
| Error analysis | ✅ | 21 type-A cultural traps (in README) |
| Results tables | 🟡 | prompt results done; add SFT/DPO rows |
| Efficiency finding | ✅ | CoT5 = 3-pass ensemble at 1/3 compute |
| Writeup / submission | ⬜ | — |

---

### How to use this file
Update the Status column as each item moves. New experiments → add a row in §3 with a
CI result once measured, then promote to §2 when done. Keep CI numbers honest (dev split).
