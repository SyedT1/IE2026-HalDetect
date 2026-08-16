"""Reproduce the paired item-bootstrap intervals reported in the system paper.

Download the released Task 1b English devtest JSONL from
QCRI/ImageEval-ArabicNLP26 and run, from the repository root:

    python paper_analysis/bootstrap_devtest.py --gold path/to/devtest_en.jsonl

CI equals item error for these format-compliant joint systems because each prediction
marks exactly one of the three statements true.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = {
    "reason-first": ROOT / "Development/qwen2p5-3b-7b/joint-3-q7b/prediction_en.zip",
    "answer-first": ROOT / "Development/qwen2p5-3b-7b/answer-first-joint-q7b/prediction_en.zip",
    "checklist": ROOT / (
        "Development/qwen2p5-3b-7b/all-COT-variations-q7b/"
        "attribute-checklist/prediction_en.zip"
    ),
    "QLoRA-2k": ROOT / "Development/qwen2p5-3b-7b/qlora-q7b-2k-image/prediction_en.zip",
    "QLoRA-2,348": ROOT / (
        "Development/qwen2p5-3b-7b/qlora-q7b-2p3k-image/prediction_en.zip"
    ),
    "QLoRA-2.6k": ROOT / (
        "Development/qwen2p5-3b-7b/qwen-q7b-2p6k-image/prediction_en.zip"
    ),
}
PAIRS = (
    ("reason-first", "answer-first"),
    ("answer-first", "checklist"),
    ("QLoRA-2k", "QLoRA-2,348"),
    ("QLoRA-2,348", "QLoRA-2.6k"),
)


def load_gold(path: Path) -> tuple[list[str], dict[str, list[bool]]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row["id"] for row in records], {row["id"]: row["labels"] for row in records}


def load_item_errors(
    archive: Path, ids: list[str], gold: dict[str, list[bool]]
) -> np.ndarray:
    predictions: dict[str, dict[int, bool]] = defaultdict(dict)
    with zipfile.ZipFile(archive) as bundle:
        csv_name = next(name for name in bundle.namelist() if name.lower().endswith(".csv"))
        with io.TextIOWrapper(bundle.open(csv_name), encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                predictions[row["id"]][int(row["statement_index"])] = (
                    row["prediction"].strip().lower() == "true"
                )

    if set(predictions) != set(ids):
        raise ValueError(f"ID mismatch in {archive}")
    return np.asarray(
        [
            [predictions[item_id][j] for j in range(3)] != gold[item_id]
            for item_id in ids
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ids, gold = load_gold(args.gold)
    names = list(PREDICTIONS)
    errors = np.stack([load_item_errors(PREDICTIONS[name], ids, gold) for name in names])
    rng = np.random.default_rng(args.seed)
    samples: list[np.ndarray] = []
    batch_size = 1_000
    for start in range(0, args.resamples, batch_size):
        size = min(batch_size, args.resamples - start)
        indices = rng.integers(0, len(ids), size=(size, len(ids)))
        samples.append(errors[:, indices].mean(axis=2).T)
    boot = np.concatenate(samples, axis=0)

    print("System CI and 95% percentile interval")
    for row, name in enumerate(names):
        low, high = np.quantile(boot[:, row], [0.025, 0.975])
        print(f"{name:15s} {errors[row].mean():.3f} [{low:.3f}, {high:.3f}]")

    print("\nPaired reduction (first minus second)")
    index = {name: i for i, name in enumerate(names)}
    for first, second in PAIRS:
        delta = boot[:, index[first]] - boot[:, index[second]]
        low, high = np.quantile(delta, [0.025, 0.975])
        point = errors[index[first]].mean() - errors[index[second]].mean()
        print(f"{first} -> {second}: {point:.3f} [{low:.3f}, {high:.3f}]")


if __name__ == "__main__":
    main()
