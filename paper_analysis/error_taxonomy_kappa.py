"""Compute raw agreement and Cohen's kappa for the paper's error taxonomy."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


LABELS = {"function_intent_event", "visual_material", "recognition"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()

    with args.annotations.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"id", "rater1", "rater2"}.issubset(rows[0]):
        raise ValueError("Expected non-empty CSV with id,rater1,rater2 columns")

    pairs = [(row["rater1"].strip(), row["rater2"].strip()) for row in rows]
    invalid = [(a, b) for a, b in pairs if a not in LABELS or b not in LABELS]
    if invalid:
        raise ValueError(f"Unknown or blank labels: {invalid[:5]}")

    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    count1 = Counter(a for a, _ in pairs)
    count2 = Counter(b for _, b in pairs)
    expected = sum((count1[label] / n) * (count2[label] / n) for label in LABELS)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    print(f"n={n}")
    print(f"raw_agreement={observed:.4f}")
    print(f"cohen_kappa={kappa:.4f}")


if __name__ == "__main__":
    main()
