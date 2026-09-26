#!/usr/bin/env python3
"""select_blend_cs.py — data-driven per-grade cond_scale selection for the sweep.

Reads the uni_cs sweep results (train_dr_classifier metrics JSONs) and picks, per
grade, the cond_scale that maximizes mean (across seeds) VAL-SET per-grade recall,
then assembles blend_sel.h5 so each grade is sourced from its winning pool.

Selection is strictly on the val split (val_per_class_report), never the test set.
The merge step reuses analysis/merge_h5_by_grade.py (per-grade source + cap).

Usage:
    python analysis/select_blend_cs.py \
        --results_dir ./downstream_results --backbone densenet121 \
        --seeds "111 112 113 114 115" \
        --pools "1.5=/abs/generated_cs1.5/generated.h5 2.5=/abs/generated_cs2.5/generated.h5 4.0=/abs/generated_cs4.0/generated.h5" \
        --out output/blends/blend_sel.h5 --cap 1000
    Pass --dry-run to print the selection without writing the blend.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys


def parse_spec(spec):
    mapping = {}
    for token in spec.split():
        key, _, value = token.partition("=")
        if not key or not value:
            raise SystemExit("token must be 'key=value', got '{}'".format(token))
        if key in mapping:
            raise SystemExit("key mapped twice in spec: '{}'".format(key))
        mapping[key] = value
    return mapping


def load_metric(results_dir, run):
    path = os.path.join(results_dir, run + "_metrics.json")
    if not os.path.isfile(path):
        raise SystemExit("missing sweep run: {}".format(path))
    with open(path) as f:
        return json.load(f)


def val_recall(metrics):
    vpc = metrics.get("val_per_class_report")
    if vpc is None:
        raise SystemExit(
            "{} lacks 'val_per_class_report' (stale JSON) -- rerun the sweep with "
            "the updated train_dr_classifier.py".format(metrics.get("run_name", "run")))
    return [vpc["grade_{}".format(g)]["recall"] for g in range(5)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default="./downstream_results")
    p.add_argument("--backbone", default="densenet121")
    p.add_argument("--seeds", default="111 112 113 114 115")
    p.add_argument("--pools", required=True, help="whitespace-separated '1.5=path 2.5=path 4.0=path'")
    p.add_argument("--arm_prefix", default="uni",
                   help="run-name prefix of the sweep arms; the run looked up is "
                        "'{backbone}_{arm_prefix}_cs{cs}_s{seed}'. Use e.g. "
                        "'uni_sde' for a per-sampler sweep (Exp 5 factorial).")
    p.add_argument("--out", default="output/blends/blend_sel.h5")
    p.add_argument("--cap", type=int, default=1000)
    p.add_argument("--caps_override", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pools = parse_spec(args.pools)
    seeds = args.seeds.split()
    cses = sorted(pools)
    for path in pools.values():
        if not os.path.isfile(path):
            raise SystemExit("pool file missing: {}".format(path))

    rec = {}
    for cs in cses:
        rec[cs] = []
        for g in range(5):
            vals = []
            for s in seeds:
                run = "{}_{}_s{}".format(args.backbone, "{}_cs{}".format(args.arm_prefix, cs), s)
                vals.append(val_recall(load_metric(args.results_dir, run))[g])
            rec[cs].append(statistics.mean(vals))

    print("\n mean VAL per-grade recall across seeds {} (backbone {})".format(seeds, args.backbone))
    header = "{:8}".format("grade")
    for cs in cses:
        header += "{:>12}".format("cs{}".format(cs))
    header += "{:>14}".format("selected")
    print(header)
    chosen = []
    for g in range(5):
        best = max(cses, key=lambda cs: rec[cs][g])
        chosen.append(best)
        row = "{:8}".format("g{}".format(g))
        for cs in cses:
            row += "{:>12.4f}".format(rec[cs][g])
        row += "{:>14}".format("cs{}".format(best))
        print(row)

    sources = " ".join("{}={}".format(g, pools[c]) for g, c in enumerate(chosen))
    print("\n blend_sel sources:\n   " + sources)
    if args.dry_run:
        print(" [dry-run] no blend written")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cmd = [sys.executable, "analysis/merge_h5_by_grade.py",
           "--sources", sources, "--out", args.out, "--cap", str(args.cap)]
    if args.caps_override:
        cmd += ["--caps_override", args.caps_override]
    subprocess.run(cmd, check=True)
    print("\n blend_sel written to {}".format(args.out))


if __name__ == "__main__":
    main()