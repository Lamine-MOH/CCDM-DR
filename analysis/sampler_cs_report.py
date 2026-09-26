#!/usr/bin/env python3
"""sampler_cs_report.py — report the Exp 5 sampler x cond_scale factorial.

Reads the train_dr_classifier metrics JSONs of a --factorial run_matrix.sh
execution (arm name ``uni_{sampler}_cs{cond_scale}``) and prints, per backbone:

  1. the 2xK grid (sampler x cond_scale) of mean +/- std over seeds, for the
     headline metrics and for per-grade recall;
  2. the paired delta of every cell against ``real_only`` (index-aligned by seed),
     with the +n/-n win count;
  3. the head-to-head sampler contrast at matched cond_scale (sde - ode, paired
     by seed) — the direct A/B;
  4. main effects: one row per sampler (averaged over cond_scale) and one column
     per cond_scale (averaged over samplers).

Selection was on the val split (``--val_frac``/``--val_seed``, fixed across arms),
so ``--side val`` reports the selection-side numbers and ``--side test`` the
final held-out ones. Both are printed by default; read decisions off the val side
and the test side as confirmation.

Usage:
    python analysis/sampler_cs_report.py \
        --results_dir ./downstream_results/Exp5_sampler_cs \
        --backbones "densenet121 resnet50 efficientnet_b4" \
        --seeds "111 112 113 114 115" \
        --samplers sde ode --cses 1.0 2.0 3.0 4.0 \
        --out_csv ./downstream_results/Exp5_sampler_cs/sampler_cs_grid.csv
"""

import argparse
import csv
import json
import os
import statistics

HEADLINE = ["qwk", "accuracy", "macro_f1"]
PER_GRADE = ["recall_g{}".format(g) for g in range(5)]
# val side: only qwk + per-grade recall are stored in the metrics JSON
VAL_HEADLINE = ["qwk"]


def load_metric(results_dir, run):
    path = os.path.join(results_dir, run + "_metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_metric(metrics, key, side):
    """Fetch one metric. `key` is a headline name ('qwk', ...) or 'recall_g{g}'.

    The val side is only partially recorded by train_dr_classifier.py: it stores
    `val_qwk` (the selection scalar) and `val_per_class_report` (per-grade
    recall), but no val accuracy / macro_f1. Those return None on the val side
    rather than silently falling back to the test value.
    """
    if key.startswith("recall_"):
        grade = "grade_{}".format(key[len("recall_g"):])
        if side == "val":
            vpc = metrics.get("val_per_class_report")
            return None if vpc is None else vpc[grade]["recall"]
        return metrics["per_class_report"][grade]["recall"]
    if side == "val":
        if key == "qwk":
            return metrics.get("val_qwk")
        return None
    return metrics[key]


def mean_sd(vals):
    if not vals:
        return None, None
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, sd


def fmt_pm(mean, sd):
    if mean is None:
        return "     --      "
    if sd is None:
        return "{:.4f}".format(mean)
    return "{:.4f}±{:.4f}".format(mean, sd)


def fmt_pm_short(mean, sd):
    if mean is None:
        return "--"
    if sd is None:
        return "{:.4f}".format(mean)
    return "{:.4f}±{:.4f}".format(mean, sd)


class Cell:
    """One (backbone, sampler, cond_scale) cell = the metric vectors of its seeds."""

    def __init__(self, backbone, sampler, cs, runs, missing=()):
        self.backbone = backbone
        self.sampler = sampler
        self.cs = cs
        self.runs = runs          # {seed: metrics} for the seeds that exist
        self.missing = list(missing)

    def seeds(self):
        return sorted(self.runs)

    def values(self, key, side):
        return [get_metric(m, key, side) for m in self.runs.values()]

    def mean_sd(self, key, side):
        vals = [v for v in self.values(key, side) if v is not None]
        return mean_sd(vals)

    def n(self, key, side):
        return sum(1 for v in self.values(key, side) if v is not None)


def collect(results_dir, backbone, sampler, cs, seeds):
    runs, missing = {}, []
    for s in seeds:
        m = load_metric(results_dir, "{}_uni_{}_cs{}_s{}".format(backbone, sampler, cs, s))
        if m is None:
            missing.append(s)
        else:
            runs[s] = m
    return Cell(backbone, sampler, cs, runs, missing), missing


def baseline(results_dir, backbone, seeds):
    runs, missing = {}, []
    for s in seeds:
        m = load_metric(results_dir, "{}_real_only_s{}".format(backbone, s))
        if m is None:
            missing.append(s)
        else:
            runs[s] = m
    return Cell(backbone, "real_only", "-", runs), missing


def paired_delta(base, cell, key, side):
    """Index-aligned per-seed delta cell - base; returns (vals, wins, losses)."""
    deltas = []
    for s in cell.seeds():
        if s not in base.runs:
            continue
        v = get_metric(cell.runs[s], key, side)
        b = get_metric(base.runs[s], key, side)
        if v is None or b is None:
            continue
        deltas.append(v - b)
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    return deltas, wins, losses


def paired_contrast(a, b, key, side):
    """Index-aligned per-seed delta a - b (used for the head-to-head sampler A/B)."""
    deltas = []
    for s in a.seeds():
        if s not in b.runs:
            continue
        v = get_metric(a.runs[s], key, side)
        w = get_metric(b.runs[s], key, side)
        if v is None or w is None:
            continue
        deltas.append(v - w)
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    return deltas, wins, losses


def check_protocol(cells):
    """Warn loudly if the grid is not a single clean protocol."""
    sel, val_seeds, val_fracs, seeds = set(), set(), set(), set()
    pretrained = set()
    for cell in cells:
        for m in cell.runs.values():
            sel.add(m.get("selection_on"))
            val_seeds.add(m.get("val_seed"))
            val_fracs.add(m.get("val_frac"))
            seeds.add(m.get("seed"))
            pretrained.add(m.get("pretrained"))
    if len(sel) > 1:
        print("WARNING: grid mixes selection protocols {}".format(sorted(map(str, sel))))
    if len(val_seeds) > 1 or len(val_fracs) > 1:
        print("WARNING: grid mixes val splits (val_seed={}, val_frac={}) — the arms are not comparable"
              .format(sorted(map(str, val_seeds)), sorted(map(str, val_fracs))))
    if len(pretrained) > 1:
        print("WARNING: grid mixes pretrained={}".format(sorted(map(str, pretrained))))
    if len(seeds) < 5:
        print("WARNING: seed set {} has <5 seeds; Mean±std is underpowered".format(sorted(map(str, seeds))))


def print_grid(cells, samplers, cses, base, side, csv_rows):
    for key in (VAL_HEADLINE if side == "val" else HEADLINE):
        print("\n  {} ({} side)".format(key, side))
        header = "  {:<10}".format("cond_scale") + "".join("{:>22}".format(s) for s in samplers)
        print(header)
        for cs in cses:
            row = "  {:<10}".format("cs{}".format(cs))
            for s in samplers:
                cell = cells.get((s, cs))
                if cell is None:
                    row += "{:>22}".format("[no runs]")
                    continue
                m, sd = cell.mean_sd(key, side)
                row += "{:>22}".format(fmt_pm(m, sd))
                csv_rows.append(dict(backbone=cell.backbone, sampler=s, cond_scale=cs,
                                     side=side, metric=key, n=cell.n(key, side),
                                     mean=m, std=sd, kind="cell"))
            print(row)

    for g in range(5):
        key = "recall_g{}".format(g)
        print("\n  {} ({} side)".format(key, side))
        header = "  {:<10}".format("cond_scale") + "".join("{:>22}".format(s) for s in samplers)
        if base is not None:
            header += "{:>22}".format("real_only")
        print(header)
        for cs in cses:
            row = "  {:<10}".format("cs{}".format(cs))
            for s in samplers:
                cell = cells.get((s, cs))
                if cell is None:
                    row += "{:>22}".format("[no runs]")
                    continue
                m, sd = cell.mean_sd(key, side)
                row += "{:>22}".format(fmt_pm(m, sd))
                csv_rows.append(dict(backbone=cell.backbone, sampler=s, cond_scale=cs,
                                     side=side, metric=key, n=cell.n(key, side),
                                     mean=m, std=sd, kind="cell"))
            if base is not None:
                m, sd = base.mean_sd(key, side)
                row += "{:>22}".format(fmt_pm(m, sd))
            print(row)


def print_vs_baseline(cells, samplers, cses, base, side, csv_rows):
    if base is None:
        print("\n(no real_only runs found — skipping the paired-vs-baseline tables)")
        return
    for key in (VAL_HEADLINE if side == "val" else HEADLINE) + PER_GRADE:
        print("\n  delta vs real_only — {} ({} side)".format(key, side))
        print("  {:<12}{:<8}{:>16}{:>10}".format("cond_scale", "sampler", "mean±std", "+/-"))
        for cs in cses:
            for s in samplers:
                cell = cells.get((s, cs))
                if cell is None:
                    continue
                deltas, wins, losses = paired_delta(base, cell, key, side)
                m, sd = mean_sd(deltas)
                print("  {:<12}{:<8}{:>16}{:>10}".format(
                    "cs{}".format(cs), s, fmt_pm_short(m, sd), "+{}/-{}".format(wins, losses)))
                csv_rows.append(dict(backbone=cell.backbone, sampler=s, cond_scale=cs,
                                     side=side, metric=key, n=len(deltas),
                                     mean=m, std=sd, wins=wins, losses=losses,
                                     kind="delta_vs_real_only"))


def print_sampler_contrast(cells, samplers, cses, side, csv_rows):
    if len(samplers) != 2:
        return
    a_name, b_name = samplers
    for key in (VAL_HEADLINE if side == "val" else HEADLINE) + PER_GRADE:
        print("\n  sampler contrast {} - {} — {} ({} side)".format(a_name, b_name, key, side))
        print("  {:<10}{:>16}{:>10}".format("cond_scale", "mean±std", "+/-"))
        for cs in cses:
            ca, cb = cells.get((a_name, cs)), cells.get((b_name, cs))
            if ca is None or cb is None:
                continue
            deltas, wins, losses = paired_contrast(ca, cb, key, side)
            m, sd = mean_sd(deltas)
            print("  {:<10}{:>16}{:>10}".format(
                "cs{}".format(cs), fmt_pm_short(m, sd), "+{}/-{}".format(wins, losses)))
            csv_rows.append(dict(backbone=ca.backbone, sampler="{} minus {}".format(a_name, b_name),
                                 cond_scale=cs, side=side, metric=key, n=len(deltas),
                                 mean=m, std=sd, wins=wins, losses=losses,
                                 kind="sampler_contrast"))


def print_main_effects(cells, samplers, cses, side, csv_rows):
    for key in (VAL_HEADLINE if side == "val" else HEADLINE) + PER_GRADE:
        print("\n  main effects — {} ({} side)".format(key, side))
        print("  by sampler (mean over cond_scale):")
        for s in samplers:
            per_cs = []
            for cs in cses:
                cell = cells.get((s, cs))
                if cell is None:
                    continue
                m, _ = cell.mean_sd(key, side)
                if m is not None:
                    per_cs.append(m)
            m, _ = mean_sd(per_cs)
            print("    {:<8}{}".format(s, fmt_pm_short(m, None)))
            csv_rows.append(dict(backbone=cells[next(iter(cells))].backbone, sampler=s,
                                 cond_scale="ALL", side=side, metric=key,
                                 n=len(per_cs), mean=m, std=None, kind="main_effect_sampler"))
        print("  by cond_scale (mean over samplers):")
        for cs in cses:
            per_s = []
            for s in samplers:
                cell = cells.get((s, cs))
                if cell is None:
                    continue
                m, _ = cell.mean_sd(key, side)
                if m is not None:
                    per_s.append(m)
            m, _ = mean_sd(per_s)
            print("    {:<8}{}".format("cs{}".format(cs), fmt_pm_short(m, None)))
            csv_rows.append(dict(backbone=cells[next(iter(cells))].backbone, sampler="ALL",
                                 cond_scale=cs, side=side, metric=key,
                                 n=len(per_s), mean=m, std=None, kind="main_effect_cs"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="./downstream_results")
    p.add_argument("--backbones", default="densenet121 resnet50 efficientnet_b4")
    p.add_argument("--seeds", default="111 112 113 114 115")
    p.add_argument("--samplers", default="sde ode")
    p.add_argument("--cses", default="1.0 2.0 3.0 4.0")
    p.add_argument("--sides", default="test val", help="which metric side(s) to report")
    p.add_argument("--out_csv", default=None, help="optional long-format CSV of every number")
    args = p.parse_args()

    seeds = args.seeds.split()
    backbones = args.backbones.split()
    samplers = args.samplers.split()
    cses = args.cses.split()
    sides = args.sides.split()

    csv_rows = []
    any_cell = False
    for backbone in backbones:
        cells, missing_cells = {}, []
        for s in samplers:
            for cs in cses:
                cell, missing = collect(args.results_dir, backbone, s, cs, seeds)
                if cell is None:
                    missing_cells.append((s, cs, missing))
                else:
                    cells[(s, cs)] = cell
        base, base_missing = baseline(args.results_dir, backbone, seeds)

        print("\n" + "=" * 78)
        print(" {}  (seeds {} | results_dir {})".format(backbone, seeds, args.results_dir))
        print("=" * 78)
        if missing_cells:
            print("\n  MISSING cells (no runs at all):")
            for s, cs, missing in missing_cells:
                print("    uni_{}_cs{}: absent (seeds tried: {})".format(s, cs, seeds))
        partial = [(k, v.missing) for k, v in cells.items() if v.missing]
        if partial:
            print("\n  PARTIAL cells (fewer than {} seeds):".format(len(seeds)))
            for (s, cs), missing in sorted(partial, key=lambda x: (x[0][0], x[0][1])):
                print("    uni_{}_cs{}: missing seeds {}".format(s, cs, missing))
        if base_missing:
            print("\n  real_only: missing seeds {}".format(base_missing))
        if not cells:
            print("\n  no cells for this backbone — skipping.")
            continue
        any_cell = True
        check_protocol(list(cells.values()) + ([base] if base else []))

        for side in sides:
            print("\n" + "-" * 78)
            print(" SIDE: {}   (selection was on val; read decisions off the val rows)".format(side))
            print("-" * 78)
            print_grid(cells, samplers, cses, base, side, csv_rows)
            print_vs_baseline(cells, samplers, cses, base, side, csv_rows)
            print_sampler_contrast(cells, samplers, cses, side, csv_rows)
            print_main_effects(cells, samplers, cses, side, csv_rows)

    if not any_cell:
        raise SystemExit("no factorial runs found under {}".format(args.results_dir))

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        fields = ["backbone", "sampler", "cond_scale", "side", "metric", "n",
                  "mean", "std", "wins", "losses", "kind"]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
        print("\nCSV written to {}".format(args.out_csv))


if __name__ == "__main__":
    main()
