"""seed_report.py — aggregate *_metrics.json across classifier seeds.

train_dr_classifier.py writes {run_name}_metrics.json with accuracy/macro_f1/qwk
and per-grade recall (nested under per_class_report.grade_{i}.recall). This
helper groups runs (e.g. one group = baseline seeds, another = the augmented
protocol) and prints mean +/- std per metric plus index-aligned paired deltas
between groups, so seed-sensitivity is visible at a glance.

Usage:
    python analysis/seed_report.py --results_dir ./downstream_results \
        --group real_only real_only_s111 real_only_s112 real_only_s113 \
        --group augmented real_plus_synth_blendA_s111 real_plus_synth_blendA_s112 \
                          real_plus_synth_blendA_s113
"""

import argparse
import json
import os
import statistics


def metric_keys(metrics):
    keys = ["accuracy", "macro_f1", "qwk"]
    keys += ["recall_{}".format("grade_{}".format(g)) for g in range(5)
             if "grade_{}".format(g) in metrics["per_class_report"]]
    return keys


def get_metric(metrics, key):
    if key.startswith("recall_"):
        grade = key[len("recall_"):]
        return metrics["per_class_report"][grade]["recall"]
    return metrics[key]


def fmt(x):
    return "{:.4f}".format(x)


def mean_sd(vals):
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, sd


def main():
    p = argparse.ArgumentParser(description="Aggregate train_dr_classifier metrics across seeds")
    p.add_argument("--results_dir", default="./downstream_results")
    p.add_argument("--group", action="append", nargs="+",
                   help="group name followed by one or more run_names")
    args = p.parse_args()

    if not args.group:
        p.error("at least one --group is required")

    groups = []
    for entry in args.group:
        name = entry[0]
        runs = entry[1:]
        rows = []
        for run in runs:
            path = os.path.join(args.results_dir, run + "_metrics.json")
            if not os.path.exists(path):
                p.error("missing {}".format(path))
            with open(path) as f:
                rows.append((run, json.load(f)))
        groups.append((name, rows))

    all_keys = None
    for _, rows in groups:
        for _, m in rows:
            kk = metric_keys(m)
            if all_keys is None:
                all_keys = kk
            elif kk != all_keys:
                p.error("metric key mismatch between groups")

    for name, rows in groups:
        print("\n== group: {} (n={}) ==".format(name, len(rows)))
        header = "  {:16}".format("metric")
        for run, _ in rows:
            header += "{:>14}".format(run[:12])
        header += "{:>22}".format("mean +/- std")
        print(header)
        for key in all_keys:
            col = [get_metric(m, key) for _, m in rows]
            mean, sd = mean_sd(col)
            vals = "".join("{:>14}".format(fmt(v)) for v in col)
            print("  {:16} {}{:>20} +/- {:<10}".format(key, vals, fmt(mean), fmt(sd)))

    if len(groups) >= 2 and all(len(g[1]) == len(groups[0][1]) for g in groups):
        base_name, base_rows = groups[0]
        for aug_name, aug_rows in groups[1:]:
            print("\n== paired deltas ({} - {}) ==".format(aug_name, base_name))
            for key in all_keys:
                deltas = [get_metric(a[1], key) - get_metric(b[1], key)
                          for a, b in zip(aug_rows, base_rows)]
                dmean, dsd = mean_sd(deltas)
                pos = sum(1 for d in deltas if d > 1e-12)
                neg = sum(1 for d in deltas if d < -1e-12)
                print("  {:16} delta {:+.4f} +/- {:.4f}   (+{} / -{} of {})".format(
                    key, dmean, dsd, pos, neg, len(deltas)))


if __name__ == "__main__":
    main()