"""
Score a run's predictions against the labelled `dev` split, offline.

The notebooks already print these numbers when SPLIT="dev". This exists so the scores
can be recovered later from the committed artifacts alone -- without a Kaggle session,
without the log, and without a Codabench submission.

It only works on `dev`. `devtest` is blind: the answer key is not in the file, not on
Hugging Face, and not anywhere else you can reach -- only Codabench holds it. Point this
at a devtest run and it will tell you so rather than print a wrong number.

The metric arithmetic is the same as the notebooks' scoring cell, which is the same as
the Codabench 1b scorer.

Usage:
    python score_local.py answer-first-joint-i8b/predictions_run4_answer_first_internvl8b_en.csv
    python score_local.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Qwen reference CIs, keyed by the InternVL run folder that replicates them.
#
# CAREFUL: we score `dev`, but nearly all the published Qwen numbers are `devtest`. The
# two splits do not agree -- Run 4 is 0.042 on dev and 0.050 on devtest -- so comparing a
# dev run against a devtest reference silently flatters or penalises us by ~0.008. Where a
# Qwen dev number is actually known it is used and marked as such; otherwise we fall back
# to devtest and say so, rather than quietly pretending the splits are interchangeable.
#
#   folder -> (name, devtest CI, dev CI or None)
QWEN_REF = {
    "baseline":               ("Run 1 (Qwen-3B)", 0.257, None),
    "joint-3-i2b":            ("Run 3 (Qwen-3B)", 0.142, None),
    "answer-first-joint-i2b": ("Run 5 (Qwen-3B)", 0.082, None),
    "joint-3-i8b":            ("Run 2 (Qwen-7B)", 0.092, None),
    # dev 0.042 is from the README error analysis: "Run 4 on the labelled dev split
    # (CI 0.042, 21 failures)".
    "answer-first-joint-i8b": ("Run 4 (Qwen-7B)", 0.050, 0.042),
    "evidence-first":         ("CoT1 (Qwen-7B)",  0.044, None),
    "elimination":            ("CoT2 (Qwen-7B)",  0.056, None),
    "confidence-ranked":      ("CoT3 (Qwen-7B)",  0.046, None),
    "devils-advocate":        ("CoT4 (Qwen-7B)",  0.048, None),
    "attribute-checklist":    ("CoT5 (Qwen-7B)",  0.042, None),
    "socratic":               ("CoT6 (Qwen-7B)",  0.046, None),
}


def load_gold() -> dict[str, int]:
    """id -> index of the True statement, from the labelled dev split."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("QCRI/AynVQA-ArabicNLP26", filename="task1b/dev_en.jsonl",
                           repo_type="dataset")
    gold = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            gold[r["id"]] = r["labels"].index(True)
    return gold


def score(csv_path: Path, gold: dict[str, int]) -> dict | None:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    by_item: dict[str, dict[int, str]] = {}
    for r in rows:
        by_item.setdefault(r["id"], {})[int(r["statement_index"])] = r["prediction_parsed"]

    matched = [i for i in by_item if i in gold]
    if not matched:
        return None                      # devtest: none of these ids are in the dev key

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
        if ok_t:
            q_plus += 1
        q_minus += sum(ok_f)
        all_ok, any_ok = ok_t and all(ok_f), ok_t or any(ok_f)
        combined += all_ok
        if any_ok:                       # CI = 1 - consistent / partial
            n_partial += 1
            n_consistent += all_ok
        if ok_t:                         # CFHR = P(miss any Q- | Q+ correct)
            cfhr_den += 1
            cfhr_num += not all(ok_f)

    # Statements predicted True per item. Gold always has exactly one; anything else is
    # structurally impossible and can never be fully correct. A useful health signal.
    one_true = sum(sum(v == "true" for v in by_item[i].values()) == 1 for i in matched)

    return {
        "items": total,
        "CI": 1 - n_consistent / n_partial if n_partial else 0.0,
        "combined": combined / total,
        "CFHR": cfhr_num / cfhr_den if cfhr_den else 0.0,
        "q_plus": q_plus / total,
        "q_minus": q_minus / q_minus_total,
        "exactly_one_true": one_true / total,
    }


def health(csv_path: Path) -> dict:
    """Soundness checks that need no labels, so they also work on a blind devtest run.

    They cannot give you CI, but they catch the failure that actually matters: the model
    ignoring the "Answer: X" format, which silently drops the run onto the per-statement
    fallback path and makes it measure something other than what it claims to.

    Gold always marks exactly one statement True. A joint run therefore has to emit exactly
    one True per item by construction -- if it does not, the joint parse failed. A
    per-statement run (the baseline) judges each statement independently and has no such
    guarantee, so a low rate there is a real weakness of the model, not a broken parse.
    """
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    by_item: dict[str, dict[int, str]] = {}
    empty_raw = 0
    for r in rows:
        by_item.setdefault(r["id"], {})[int(r["statement_index"])] = r["prediction_parsed"]
        if r["statement_index"] == "0" and not r["raw_prediction"].strip():
            empty_raw += 1

    counts = [sum(v == "true" for v in d.values()) for d in by_item.values()]
    n = len(by_item)
    return {
        "items": n,
        "rows": len(rows),
        "exactly_one_true": sum(c == 1 for c in counts) / n,
        "no_true": sum(c == 0 for c in counts) / n,
        "multi_true": sum(c > 1 for c in counts) / n,
        "empty_raw": empty_raw,
    }


def report(csv_path: Path, gold: dict[str, int]) -> None:
    folder = csv_path.parent.name
    ref_name, ref_devtest, ref_dev = QWEN_REF.get(folder, ("Qwen", float("nan"), None))
    m = score(csv_path, gold)

    print(f"\n{csv_path.relative_to(HERE)}")
    if m is None:
        h = health(csv_path)
        joint = folder != "baseline"
        print(f"  devtest (blind) — {h['items']} items. No labels exist for these ids, so CI "
              f"cannot be computed here; only Codabench can score it.")
        print(f"  Health check (label-free):")
        print(f"    {'exactly one True per item':<30} {h['exactly_one_true']:6.1%}   (gold: 100%)")
        print(f"    {'no True at all':<30} {h['no_true']:6.1%}")
        print(f"    {'two or three True':<30} {h['multi_true']:6.1%}")
        if h["empty_raw"]:
            print(f"    {'empty model replies':<30} {h['empty_raw']}")
        if joint and h["exactly_one_true"] < 0.999:
            print(f"  !! A joint run must emit exactly one True per item by construction. "
                  f"{1 - h['exactly_one_true']:.1%} did not — the 'Answer: X' parse is falling "
                  f"back to per-statement judging. Treat this run as suspect.")
        elif joint:
            print(f"  Sound: the joint parse held on every item.")
        else:
            # Per-statement baseline: no structural guarantee, so this is a capability read.
            floor = 1 - h["exactly_one_true"]
            print(f"  Per-statement run, so a sub-100% rate is the model's weakness, not a "
                  f"parse failure.\n  Combined accuracy is capped at {h['exactly_one_true']:.1%}, "
                  f"so CI >= {floor:.3f} whatever Codabench returns.")
        return

    # Compare like with like: we scored dev, so use Qwen's dev number when we have it.
    ref_ci = ref_dev if ref_dev is not None else ref_devtest
    ref_split = "dev" if ref_dev is not None else "devtest, splits differ"

    delta = m["CI"] - ref_ci
    verdict = "InternVL better" if delta < 0 else "Qwen better" if delta > 0 else "tie"
    print(f"  {m['items']} items, scored against dev\n")
    print(f"  {'Contrastive Instability (CI)':<30} {m['CI']:.4f}   vs {ref_name} {ref_ci:.4f}"
          f" ({ref_split})   [{delta:+.4f}, {verdict}]")
    print(f"  {'Combined accuracy':<30} {m['combined']:.4f}")
    print(f"  {'CFHR':<30} {m['CFHR']:.4f}")
    print(f"  {'Q+ accuracy':<30} {m['q_plus']:.4f}")
    print(f"  {'Q- accuracy':<30} {m['q_minus']:.4f}")
    print(f"  {'exactly one True per item':<30} {m['exactly_one_true']:.1%}  (gold: 100%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*", type=Path, help="predictions_*.csv to score")
    ap.add_argument("--all", action="store_true", help="score every predictions_*.csv found")
    args = ap.parse_args()

    paths = sorted(HERE.glob("**/predictions_*.csv")) if args.all else args.csv
    if not paths:
        sys.exit("nothing to score — pass a CSV or use --all")

    gold = load_gold()
    for p in paths:
        report(p, gold)
    print()


if __name__ == "__main__":
    main()
