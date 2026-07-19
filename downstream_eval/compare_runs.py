"""
compare_runs.py

Summarizes the *_metrics.json files produced by train_dr_classifier.py into
one comparison table -- this table (or a version of it) is the main results
table for the contribution.

Usage:
    python compare_runs.py --results_dir ./downstream_results
"""

import argparse
import glob
import json
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="./downstream_results")
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*_metrics.json"))):
        run_name = os.path.basename(path).replace("_metrics.json", "")
        with open(path) as f:
            m = json.load(f)
        row = {
            "run": run_name,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "qwk": m["qwk"],
        }
        for g in range(5):
            key = f"grade_{g}"
            if key in m["per_class_report"]:
                row[f"recall_{key}"] = m["per_class_report"][key]["recall"]
        rows.append(row)

    if not rows:
        print(f"No *_metrics.json files found in {args.results_dir}")
        return

    df = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df)

    out_csv = os.path.join(args.results_dir, "comparison_table.csv")
    df.to_csv(out_csv)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
