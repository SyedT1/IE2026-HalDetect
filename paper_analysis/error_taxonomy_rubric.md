# Error-taxonomy second-rater protocol

Use this rubric for an independent second annotation of the 35 incorrect test items.
The second rater must not see the first rater's labels before completing their pass.

Assign exactly one label per item:

- `function_intent_event`: The candidates share the main visible entities but differ in
  purpose, function, intended use, activity, or event interpretation.
- `visual_material`: The candidates differ in a directly observable property such as
  colour, texture, material, shape, script, lighting, or geometry.
- `recognition`: The candidates differ in the identity of an object, place, instrument,
  food, scene, or other depicted entity.

Tie-breaking order:

1. Use `visual_material` when a visible property alone distinguishes the candidates.
2. Otherwise use `recognition` when the alternatives name different entity identities.
3. Use `function_intent_event` when the same entity is accepted but its role or activity
   differs.

Record a CSV with columns `id,rater1,rater2`. Run
`python paper_analysis/error_taxonomy_kappa.py annotations.csv` to obtain raw agreement
and Cohen's kappa. Report the number of double-annotated items and the adjudication rule.
