"""
Is a CI difference an effect, or a dice roll?

Contrastive Instability is measured on 500 items, so one item = 0.002 CI and
nothing finer than that exists. Several claims in the README rest on gaps of
2-4 items. This decides which of them survive.

Two modes.

  --bound   Needs no new data at all. For any two published CIs it computes the
            BEST CASE a paired McNemar test could possibly give -- i.e. assuming
            every net error difference is a clean one-directional flip with zero
            offsetting changes, which never actually happens. If a claim cannot
            reach p<0.05 even in its best case, it cannot be significant, full
            stop, and no re-run will save it. This is the cheapest possible audit
            and it is the one a reviewer will do in their head.

  (default) Reads the per-item CSVs produced by dev-eval-4-adapters.ipynb and
            runs the real paired McNemar plus Wilson intervals.

Why McNemar and not a two-proportion z-test: the same 500 images go to every
model, so the runs are paired. Items that both models get right (or both get
wrong) carry no information about which model is better. McNemar throws them out
and looks only at the disagreements. An unpaired test keeps them, inflates the
variance, and loses most of the power.

Usage:
    python significance.py --bound
    python significance.py                    # scores ./dev_predictions_*.csv
    python significance.py --csv-dir path/to/csvs
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import math
import os
from pathlib import Path

N_ITEMS = 500

# Published devtest CIs (Codabench). Error count = CI * 500, because for a joint
# run every item is at least partly correct, so CI reduces to the error rate.
PUBLISHED = {
    "Run 1 (3B baseline)":      0.257,
    "Run 3 (3B reason-first)":  0.142,
    "Run 2 (7B reason-first)":  0.092,
    "Run 5 (3B answer-first)":  0.082,
    "CoT2 (elimination)":       0.056,
    "Res1280":                  0.054,
    "Run 4 (7B answer-first)":  0.050,
    "CoT4 (devil's advocate)":  0.048,
    "CoT3 (confidence)":        0.046,
    "CoT6 (socratic)":          0.046,
    "CoT1 (evidence-first)":    0.044,
    "CoT5 (attr checklist)":    0.042,
    "Run ADE (ensemble)":       0.042,
    "RunFT 2.6k":               0.034,
    "RunFT 2k":                 0.032,
    "RunFT 2.3k":               0.028,
}

# The claims the paper actually makes. These are what must survive.
CLAIMS = [
    ("Joint prompting",        "Run 1 (3B baseline)",     "Run 3 (3B reason-first)"),
    ("Answer-first @3B",       "Run 3 (3B reason-first)", "Run 5 (3B answer-first)"),
    ("Answer-first @7B",       "Run 2 (7B reason-first)", "Run 4 (7B answer-first)"),
    ("Scale 3B->7B",           "Run 3 (3B reason-first)", "Run 2 (7B reason-first)"),
    ("CoT5 helps (-16%)",      "Run 4 (7B answer-first)", "CoT5 (attr checklist)"),
    ("CoT1 helps (-12%)",      "Run 4 (7B answer-first)", "CoT1 (evidence-first)"),
    ("CoT2 hurts (+12%)",      "Run 4 (7B answer-first)", "CoT2 (elimination)"),
    ("CoT5 beats CoT1",        "CoT1 (evidence-first)",   "CoT5 (attr checklist)"),
    ("Res1280 hurts",          "Run 4 (7B answer-first)", "Res1280"),
    ("FT beats best zero-shot","CoT5 (attr checklist)",   "RunFT 2.3k"),
    ("FT 2k -> 2.3k (-12.5%)", "RunFT 2k",                "RunFT 2.3k"),
    ("FT 2.3k -> 2.6k (+21%)", "RunFT 2.3k",              "RunFT 2.6k"),
    ("FT 2k -> 2.6k",          "RunFT 2k",                "RunFT 2.6k"),
]


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p0=0.5. Hand-rolled so this runs with no
    scipy -- it must work in any environment, including a bare checkout."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] + 1e-12))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval. The normal approximation is unusable at p~0.03,
    n=500 (it can dip below zero); Wilson stays valid."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def mode_bound() -> None:
    """Best-case McNemar from published CIs alone. No per-item data required."""
    print(f"\nPublished devtest CIs as ERROR COUNTS (n={N_ITEMS}; 1 item = "
          f"{1/N_ITEMS:.4f} CI)\n")
    print(f'{"run":<26} {"CI":>6} {"wrong":>6} {"95% Wilson":>20}')
    print("-" * 62)
    for name, ci in PUBLISHED.items():
        k = round(ci * N_ITEMS)
        lo, hi = wilson(k, N_ITEMS)
        print(f"{name:<26} {ci:>6.3f} {k:>6} {f'[{lo:.3f}, {hi:.3f}]':>20}")

    print(f"\n\nBEST-CASE McNEMAR FOR EACH PAPER CLAIM\n")
    print("The 'best case' assumes every net error difference is a clean")
    print("one-directional flip with zero offsetting changes -- the most")
    print("generous outcome physically available. Real p-values are worse.\n")
    print(f'{"claim":<26} {"gap":>4} {"best-case p":>12}   verdict')
    print("-" * 74)
    for label, a, b in CLAIMS:
        ka, kb = round(PUBLISHED[a] * N_ITEMS), round(PUBLISHED[b] * N_ITEMS)
        d = abs(ka - kb)
        p = binom_two_sided(0, d) if d else 1.0
        if p >= 0.05:
            verdict = "CANNOT be significant, at any b/c split"
        else:
            verdict = "could be significant -- needs the real paired test"
        print(f"{label:<26} {d:>4} {p:>12.4f}   {verdict}")

    print("\n" + "=" * 74)
    print("Every row marked CANNOT is a claim the data cannot support, no matter")
    print("how the individual items fall. Re-running will not fix it -- only a")
    print("bigger evaluation set, or dropping the claim, will.")
    print("=" * 74 + "\n")


def mode_csv(csv_dir: Path) -> None:
    files = sorted(glob.glob(str(csv_dir / "dev_predictions_*.csv")))
    if not files:
        raise SystemExit(
            f"no dev_predictions_*.csv in {csv_dir}\n"
            f"Run dev-eval-4-adapters.ipynb on Kaggle first, then drop its CSVs here.")

    runs: dict[str, dict[str, bool]] = {}
    for f in files:
        name = os.path.basename(f).replace("dev_predictions_", "").replace(".csv", "")
        runs[name] = {r["id"]: r["correct"] == "1"
                      for r in csv.DictReader(open(f, encoding="utf-8"))}

    ids = set.intersection(*(set(v) for v in runs.values()))
    n = len(ids)
    print(f"\nDEV SPLIT — {len(runs)} runs over {n} shared items\n")
    print(f'{"run":<12} {"wrong":>6} {"CI":>8} {"95% Wilson":>20}')
    print("-" * 50)
    for name, c in runs.items():
        k = sum(not c[i] for i in ids)
        lo, hi = wilson(k, n)
        print(f"{name:<12} {k:>6} {k/n:>8.4f} {f'[{lo:.4f}, {hi:.4f}]':>20}")

    print(f"\n\nPAIRED McNEMAR (exact, two-sided)\n")
    print(f'{"comparison":<26} {"A>B":>4} {"B>A":>4} {"p":>8}   verdict')
    print("-" * 70)
    for a, b in itertools.combinations(runs, 2):
        only_a = sum(runs[a][i] and not runs[b][i] for i in ids)
        only_b = sum(runs[b][i] and not runs[a][i] for i in ids)
        p = binom_two_sided(only_a, only_a + only_b)
        v = "SIGNIFICANT" if p < 0.05 else "not significant — consistent with noise"
        print(f"{a + ' vs ' + b:<26} {only_a:>4} {only_b:>4} {p:>8.3f}   {v}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bound", action="store_true",
                    help="best-case McNemar from published CIs; needs no new data")
    ap.add_argument("--csv-dir", type=Path, default=Path(__file__).parent)
    a = ap.parse_args()
    mode_bound() if a.bound else mode_csv(a.csv_dir)


if __name__ == "__main__":
    main()
