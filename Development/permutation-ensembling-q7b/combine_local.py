"""Local CPU-only reproduction of the Run ADE majority-vote ensemble.

Runs off the prediction CSVs committed in this repo (no Kaggle, no GPU).
Mirrors permutation-ensembling-q7b.ipynb:
  Run 4 (perm A) + Run 5a (perm D) + Run 5b (perm E) -> 2-of-3 majority vote,
  3-way tie broken by Run 4. Writes prediction_en.csv / .zip next to this file.
"""
import csv
import os
import zipfile
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "run4": os.path.join(HERE, "perm-A", "prediction_run4_answer_first_en.csv"),
    "run5a": os.path.join(HERE, "perm-D", "prediction_run5a_perm_D_en.csv"),
    "run5b": os.path.join(HERE, "perm-E", "prediction_run5a_perm_E_en.csv"),
}


def load_votes(path):
    """Long CSV (id, statement_index, prediction) -> {id: chosen_index}."""
    df = pd.read_csv(path)
    return {
        row["id"]: int(row["statement_index"])
        for _, row in df.iterrows()
        if str(row["prediction"]).lower() == "true"
    }


def main():
    votes = {run: load_votes(p) for run, p in FILES.items()}
    ids = sorted(votes["run4"])

    final = {}
    unanimous = majority = tiebreak = 0
    for iid in ids:
        v4 = votes["run4"].get(iid)
        v5a = votes["run5a"].get(iid, v4)
        v5b = votes["run5b"].get(iid, v4)
        winner, top = Counter([v4, v5a, v5b]).most_common(1)[0]
        if top == 3:
            unanimous += 1
        elif top == 2:
            majority += 1
        else:
            winner, _ = v4, tiebreak  # 3-way tie -> Run 4
            tiebreak += 1
        final[iid] = winner

    n = len(ids)
    print(f"Total items:         {n}")
    print(f"Unanimous (3/3):     {unanimous}  ({unanimous/n*100:.1f}%)")
    print(f"Majority (2/3):      {majority}  ({majority/n*100:.1f}%)")
    print(f"3-way tie (-> Run4): {tiebreak}  ({tiebreak/n*100:.1f}%)")

    out_csv = os.path.join(HERE, "prediction_en.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "statement_index", "prediction"])
        for iid in ids:
            for si in range(3):
                w.writerow([iid, si, "true" if si == final[iid] else "false"])
    with zipfile.ZipFile(os.path.join(HERE, "prediction_en.zip"), "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(out_csv, "prediction_en.csv")
    print(f"Wrote {out_csv} and prediction_en.zip ({n*3} rows)")


if __name__ == "__main__":
    main()
