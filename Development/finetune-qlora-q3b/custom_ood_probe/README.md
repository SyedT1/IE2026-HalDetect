# Custom OOD Probe — Arab-cultural hallucination set (101 items)

A small **out-of-distribution sanity set** to check whether our QLoRA-SFT / CoT5 models
generalise beyond the official AynVQA dev split. Same task shape as Task 1b: one image +
3 statements, exactly one grounded (True), two culturally-plausible hallucinations.

> ⚠️ **This is a probe, not a benchmark.** Images are sourced from **Wikimedia Commons**
> (CC-licensed, provenance recorded); statements were drafted by an AI grounded in each
> image's Commons description, then **human-verified**. Use it for a qualitative
> generalisation check and an error-analysis appendix — NOT as authoritative ground truth,
> and NOT for training (self-authored labels would be circular with our own model).

## Contents
| File | What |
|---|---|
| `custom_ood_en.jsonl` | 101 records, official format (`id, image, statements[3], labels[3], country, category, subcategory`) + provenance fields (`image_url, source_page, visible_content, confidence, license, true_index, verified`) |
| `index.json` | viewer-friendly list |
| `verify.html` | **open in a browser** — click-through viewer: image + 3 statements (true one highlighted) + confidence + Commons link. Use this to verify/fix labels. |
| `download_images.py` | after verification, pulls images into `images/` to make the set self-contained |
| `assemble.py` | rebuilds the outputs from `raw/*.json` (edit raw, re-run) |
| `raw/*.json` | per-region collector batches (source of truth) |
| `SPEC.md` | the build spec the collectors followed |

## Composition (101)
- **Countries (17):** Saudi 9, Egypt 8, Iraq 8, Sudan 8, Oman 7, Morocco 6, Qatar 6,
  Tunisia 6, Algeria 5, Bahrain 5, Kuwait 5, Syria 5, UAE 6, Jordan 4, Lebanon 4,
  Palestine 4, Yemen 5.
- **Categories (8):** Geography/Buildings/Landmarks 35, Arts & Culture 12, Nature & Animals 12,
  Religion & Spirituality 12, Food & Cooking 11, Objects/Materials/Clothing 10,
  Sports & Recreation 5, People/Society/Education 4.
- **Answer balance:** true_index 34 / 33 / 34 (no position bias).
- **Self-rated confidence:** high 84, medium 14, low 3 — **scrutinise medium/low first.**

## How to verify (you + teammate)
1. Open `verify.html` in a browser. Each card shows the image, the 3 statements (grounded
   one in green), the model's stated `visible_content`, and a link to the Commons page.
2. For each item confirm: (a) the image loads and matches `visible_content`; (b) the
   green statement is actually True for that image; (c) the two others are False but
   plausible. Tick the "hide high-confidence" box to focus on medium/low first.
3. Fix any wrong item by editing its entry in the matching `raw/<region>.json`
   (`true_index`, statement text, or drop it), then re-run `python assemble.py`.
4. When satisfied, set `verified` handling as you like and run `python download_images.py`
   to localise images, then upload `custom_ood_en.jsonl` + `images/` as a Kaggle Dataset.

## How to score a model on it
Reuse `../kaggle-infer-q3b.ipynb` but point the dev loader at `custom_ood_en.jsonl` and the
local `images/` (instead of the HF dev split). Compare CI on this probe vs our dev CI 0.058:
- probe CI ≈ dev CI → model generalises to fresh Arab-cultural images ✅
- probe CI ≫ dev CI → dev-split overfitting / narrower competence ⚠️

## Honesty notes for the paper
- Distractors are AI-drafted → may be easier or subtly different from the organisers'
  hand-crafted traps. Report this as a limitation.
- Grounding is from Commons captions, not pixel inspection → human verification is the
  ground truth, hence the `verified` flag and the mandatory review step above.
- Stronger generalisation evidence is still the **Codabench devtest** blind leaderboard —
  run that too.
