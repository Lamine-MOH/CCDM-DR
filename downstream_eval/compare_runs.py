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
        if "backbone" in m:
            row["backbone"] = m["backbone"]
        if "seed" in m:
            row["seed"] = m["seed"]
        if "best_epoch" in m:
            row["best_epoch"] = m["best_epoch"]
        if "selection_on" in m:
            row["selection_on"] = m["selection_on"]
        if "val_qwk" in m:
            row["val_qwk"] = m["val_qwk"]
        for g in range(5):
            key = f"grade_{g}"
            if key in m["per_class_report"]:
                row[f"recall_{key}"] = m["per_class_report"][key]["recall"]
                row[f"support_{key}"] = m["per_class_report"][key]["support"]
        rows.append(row)

    if not rows:
        print(f"No *_metrics.json files found in {args.results_dir}")
        return

    df = pd.DataFrame(rows).set_index("run")

    # sanity warnings: mixing selection protocols / backbones / seed protocols
    if "selection_on" in df.columns:
        sel = df["selection_on"].dropna().unique()
        if len(sel) > 1 and "test" in sel:
            print(f"WARNING: runs mix selection protocols ({sorted(sel)}); "
                  f"'test'-selected rows are deprecated and not comparable.")
    if "backbone" in df.columns:
        bb = df["backbone"].dropna().unique()
        if len(bb) > 1:
            print(f"WARNING: table mixes backbones {list(dict.fromkeys(bb))}; "
                  "cross-backbone deltas are not apples-to-apples.")
    if "seed" in df.columns:
        seeds = df["seed"].dropna().unique()
        missing = df["seed"].isna().sum()
        if missing:
            print(f"WARNING: {missing} run(s) carry no 'seed' field (older JSON); seed-aware averaging excludes them.")
        if len(seeds) > 1 and len(seeds) < 5:
            print(f"WARNING: seed set {sorted(seeds)} has <5 seeds; Mean±std over these is underpowered.")

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df)

    out_csv = os.path.join(args.results_dir, "comparison_table.csv")
    df.to_csv(out_csv)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
