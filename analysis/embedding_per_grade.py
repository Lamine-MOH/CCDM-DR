#!/usr/bin/env python3
"""
embedding_per_grade.py — per-grade companion to the all-grade pair plots.

`embedding_map.py --pairs real` writes a global panel plus one panel per grade
in a 3x2 figure. The presentation uses compact all-grade panels; this script
adds a single 2x5 view that keeps the per-grade detail:

  top row    : real (grey) + cs1.5 (green) + cs4.0 (amber)
  bottom row : real (grey) + aug_train (teal) + aug_popular (orange)

One panel per ICDR grade, shared axes, so the viewer can see where each pool
covers the real cloud inside each grade.

No metrics are recomputed: the UMAP coordinates and grade labels come from
`embedding_projection.csv`. Coverage numbers remain in PCA space.

Usage:
  python analysis/embedding_per_grade.py \
      --projection experiments_outputs/embedding_map_rerun/embedding_projection.csv \
      --out experiments_outputs/embedding_map_rerun/map_per_grade_pairs.png
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {
    "real": "#bdbdbd",
    "cs1.5": "#5aa832",
    "cs4.0": "#e0a526",
    "aug_train": "#2f9e6f",
    "aug_popular": "#d2691e",
}
ROWS = [
    ("Guidance", [("real", "Real"), ("cs1.5", "cs1.5"), ("cs4.0", "cs4.0")]),
    ("Augmentation", [("real", "Real"), ("aug_train", "aug_train"),
                      ("aug_popular", "aug_popular")]),
]


def read_rows(path, method="umap"):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == method]
    out = {}
    for r in rows:
        out.setdefault((r["source"], int(r["grade"])), []).append(
            (float(r["x"]), float(r["y"])))
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="umap")
    ap.add_argument("--seed", type=int, default=111)
    args = ap.parse_args()

    data = read_rows(args.projection, args.method)
    all_xy = np.concatenate(list(data.values()))
    lim_x = np.percentile(all_xy[:, 0], [0.5, 99.5])
    lim_y = np.percentile(all_xy[:, 1], [0.5, 99.5])
    pad = 0.06 * np.array([lim_x[1] - lim_x[0], lim_y[1] - lim_y[0]])

    fig, axes = plt.subplots(2, 5, figsize=(18.0, 8.2), sharex=True, sharey=True)
    for row, (row_name, sources) in enumerate(ROWS):
        for grade in range(5):
            ax = axes[row, grade]
            for src, label in sources:
                pts = data.get((src, grade))
                if pts is None or len(pts) == 0:
                    continue
                # real is the grey background; the comparison sources sit on top
                is_real = src == "real"
                ax.scatter(pts[:, 0], pts[:, 1],
                           s=5 if not is_real else 4,
                           c=COLORS[src],
                           alpha=0.55 if not is_real else 0.40,
                           linewidths=0, rasterized=True, label=label,
                           zorder=1 if is_real else 2)
                if not is_real:
                    ax.scatter(pts[:, 0].mean(), pts[:, 1].mean(),
                               s=90, marker="x", c=COLORS[src],
                               linewidths=1.8, zorder=4)
            n_real = len(data.get(("real", grade), []))
            n_syn = len(data.get((sources[1][0], grade), []))
            ax.set_title(f"grade {grade}  (real n={n_real}, {sources[1][1]} n={n_syn})",
                         fontsize=11)
            ax.set_xlim(lim_x[0] - pad[0], lim_x[1] + pad[0])
            ax.set_ylim(lim_y[0] - pad[1], lim_y[1] + pad[1])
            ax.grid(alpha=0.16, linewidth=0.5)
            if grade == 0:
                ax.set_ylabel(f"{row_name}\n{args.method.upper()}-2", fontsize=11)
            if row == 1:
                ax.set_xlabel(f"{args.method.upper()}-1", fontsize=10)
            # per-row legend in the first panel
            if grade == 0:
                handles = [Line2D([], [], marker="o", ls="", ms=5,
                                  color=COLORS[s], label=l)
                           for s, l in sources]
                ax.legend(handles=handles, loc="upper right", fontsize=8,
                          frameon=False, markerscale=1.4)

    fig.suptitle("Per-grade UMAP: real vs guidance pools (top) and augmentation (bottom)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=190, bbox_inches="tight")
    print("[write]", args.out)


if __name__ == "__main__":
    main()
