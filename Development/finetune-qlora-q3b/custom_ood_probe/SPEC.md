# Custom OOD Probe — build spec (shared by all collector agents)

**Purpose:** a small out-of-distribution sanity set (~100 items) to check whether our
QLoRA-SFT / CoT5 models generalise beyond the official dev split. **NOT a benchmark, NOT
training data.** Every item is human-verified before use.

## Image source (MANDATORY)
- Use **Wikimedia Commons** images ONLY (CC-BY / CC-BY-SA / public domain — safe provenance).
- You MUST obtain the **real** direct file URL from an actual Commons search / API result —
  never invent a URL. Prefer `https://upload.wikimedia.org/...` direct links.
- Also record the Commons **file page** (`https://commons.wikimedia.org/wiki/File:...`) so a
  human can open and verify the picture.
- Only use images whose Commons **description/caption clearly states what is visible** — you
  ground the TRUE statement in that described, verifiable content.

## Domain
Arab-world cultural images. Spread across countries: Egypt, Saudi Arabia, UAE, Kuwait,
Bahrain, Qatar, Oman, Jordan, Palestine, Syria, Lebanon, Iraq, Tunisia, Algeria, Morocco,
Yemen, Sudan.

Categories (use these exact strings):
- `Objects, Materials & Clothing`
- `People, Society & Education`
- `Sports & Recreation`
- `Geography, Buildings & Landmarks`
- `Food & Cooking`
- `Religion & Spirituality`
- `Nature & Animals`
- `Arts & Culture`

## Statement design (the hard, important part)
Each item = **3 statements sharing ONE frame**, with a single swapped slot. Exactly ONE is
True (grounded in the image); the other two are **culturally-plausible hallucinations** —
same type, could-plausibly-be-real, NOT obviously wrong. This plausibility is the whole point.

Frame patterns (mirror the official data — pick whichever fits the image):
- Attribute: `"{X} is prominently featured in this image. True or False?"`
- Association: `"The {object} is most closely associated with {concept}. True or False?"`
- Location: `"The {activity} is taking place {location-phrase}. True or False?"`
- Detail: `"The {object} displays / shows {detail}. True or False?"`
- Material/colour: `"The {object} is made of {material}. True or False?"`

Rules:
- All 3 statements must be the SAME frame, differing only in the swapped slot.
- The 2 false fillers must be from the same semantic class as the true one (e.g. if true =
  "silk", falses = "cotton"/"wool", NOT "concrete").
- End every statement with ` True or False?` (match official style).
- Ground the TRUE filler in the Commons description; do not guess beyond what is described.

## Output — return STRICT JSON, a list of objects, each exactly:
```json
{
  "country": "Kuwait",
  "category": "Geography, Buildings & Landmarks",
  "subcategory": "Famous Landmarks",
  "image_url": "https://upload.wikimedia.org/wikipedia/commons/....jpg",
  "source_page": "https://commons.wikimedia.org/wiki/File:....jpg",
  "visible_content": "one sentence: what the Commons description says is actually shown",
  "statements": ["... True or False?", "... True or False?", "... True or False?"],
  "true_index": 0,
  "confidence": "high|medium|low",
  "license": "CC-BY-SA-4.0 (or as stated on the file page)"
}
```
- `true_index` = 0-based index of the grounded statement.
- `confidence` = how sure you are the TRUE statement matches the actual image, from the
  description. Mark `low` if the description is thin — humans will scrutinise those.
- Vary `true_index` across items (don't always put the true one first).
- Return ONLY the JSON list. No prose.
