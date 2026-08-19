"""Score the seed-scaling prediction ZIPs against the released Task-1b answer keys.

The `test` split answer key is now public in `QCRI/ImageEval-ArabicNLP26`
(`task1b/test_en.jsonl` carries `labels`), so test-phase CI no longer needs a
Codabench submission. This script recomputes CI offline from the committed ZIPs.

The metric arithmetic is identical to the Codabench 1b scorer and to
`Development/internvl-2b-8b/score_local.py`:

    CI = 1 - (items fully correct) / (items at least partly correct)

The split is detected per file by id overlap, so devtest/dev artifacts are never
silently scored against the test key.

Validation: the four seed-42 rows reproduce the published root-README test CIs
(0.049 / 0.039 / 0.035 / 0.040) exactly.

Usage:
    python score_test_predictions.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
REPO = HERE.parents[2]                      # repo root
Q7B = HERE.parent                           # Development/qwen2p5-3b-7b

SPLITS = ("test", "devtest", "dev", "train")

# (training items, seed, artifact path relative to the repo root)
TARGETS = [
    (2000, 42,  "Test/qwen2p5-3b-7b/qlora-q7b-2k-image/prediction_en.zip"),
    (2348, 42,  "Test/qwen2p5-3b-7b/qlora-q7b-2p3k-image/prediction_en.zip"),
    (2600, 42,  "Test/qwen2p5-3b-7b/qlora-q7b-2p6k-image/prediction_en.zip"),
    (3000, 42,  "Test/qwen2p5-3b-7b/qlora-q7b-3k-image/prediction_en.zip"),
    (2000, 13,  "Development/qwen2p5-3b-7b/qlora-three-seed-scaling/2k/prediction_seed13_2k_q7b.zip"),
    (2000, 73,  "Development/qwen2p5-3b-7b/qlora-three-seed-scaling/2k/prediction_seed73-2k-q7b.zip"),
    (2000, 101, "Development/qwen2p5-3b-7b/qlora-three-seed-scaling/2k/prediction_seed101_2k_q7b.zip"),
    (2348, 13,  "Development/qwen2p5-3b-7b/qlora-three-seed-scaling/2348/prediction_seed13_2p3k_q7b.zip"),
    (2348, 73,  "Development/qwen2p5-3b-7b/qlora-three-seed-scaling/2348/prediction_seed73_2p3k_q7b.zip"),
    (2348, 101, "Development/qwen2p5-3b-7b/qlora-three-seed-scaling/2348/prediction_seed101_2p3k_q7b.zip"),
]


def load_gold(split: str) -> dict[str, int]:
    """id -> index of the True statement."""
    path = hf_hub_download("QCRI/ImageEval-ArabicNLP26", f"task1b/{split}_en.jsonl",
                           repo_type="dataset")
    gold = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            gold[r["id"]] = r["labels"].index(True)
    return gold


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            text = z.read(name).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def score(rows: list[dict], gold: dict[str, int]) -> dict:
    col = "prediction_parsed" if "prediction_parsed" in rows[0] else "prediction"
    by_item: dict[str, dict[int, str]] = {}
    for r in rows:
        by_item.setdefault(r["id"], {})[int(r["statement_index"])] = r[col].strip().lower()

    matched = [i for i in by_item if i in gold]
    total = q_plus = q_minus = q_minus_total = combined = 0
    n_partial = n_consistent = 0
    cfhr_num = cfhr_den = 0
    for iid in matched:
        true_idx = gold[iid]
        pr = by_item[iid]
        total += 1
        q_minus_total += 2
        ok_t = pr.get(true_idx) == "true"
        ok_f = [pr.get(i) == "false" for i in range(3) if i != true_idx]
        q_plus += ok_t
        q_minus += sum(ok_f)
        all_ok, any_ok = ok_t and all(ok_f), ok_t or any(ok_f)
        combined += all_ok
        if any_ok:
            n_partial += 1
            n_consistent += all_ok
        if ok_t:
            cfhr_den += 1
            cfhr_num += not all(ok_f)

    one_true = sum(sum(v == "true" for v in by_item[i].values()) == 1 for i in matched)
    return {
        "items": total,
        "unmatched": len(by_item) - total,
        "CI": 1 - n_consistent / n_partial if n_partial else 0.0,
        "combined": combined / total,
        "CFHR": cfhr_num / cfhr_den if cfhr_den else 0.0,
        "q_plus": q_plus / total,
        "q_minus": q_minus / q_minus_total,
        "exactly_one_true": one_true / total,
    }


def main() -> None:
    golds = {s: load_gold(s) for s in SPLITS}
    out = []
    print(f"{'split':>8} {'n':>5} {'seed':>5} {'items':>6} {'CI':>7} {'comb':>7} "
          f"{'Q+':>7} {'Q-':>7} {'CFHR':>6} {'1true':>6}")
    for n, seed, rel in TARGETS:
        path = REPO / rel
        if not path.exists():
            print(f"MISSING {rel}", file=sys.stderr)
            continue
        rows = read_rows(path)
        ids = {r["id"] for r in rows}
        split = max(golds, key=lambda k: len(ids & golds[k].keys()))
        s = score(rows, golds[split])
        out.append({"n": n, "seed": seed, "split": split, "file": rel, **s})
        print(f"{split:>8} {n:>5} {seed:>5} {s['items']:>6} {s['CI']:>7.4f} "
              f"{s['combined']:>7.4f} {s['q_plus']:>7.4f} {s['q_minus']:>7.4f} "
              f"{s['CFHR']:>6.3f} {s['exactly_one_true']:>6.3f}")

    (HERE / "test_scores.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {HERE / 'test_scores.json'}")


if __name__ == "__main__":
    main()
