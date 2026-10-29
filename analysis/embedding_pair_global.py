#!/usr/bin/env python3
"""
embedding_pair_global.py — readable 1-vs-1 panels from an existing projection.

`embedding_map.py --pairs real` writes full 3x2 pair figures (global + one panel
per grade). At slide scale those subpanels are too small. This script re-plots
the *global* panel only, from `embedding_projection.csv`, with the real cloud in
grey and the comparison source in its colour. Numbers are read from
`embedding_metrics.csv` / `pairwise_metrics.csv` when available.

Usage:
  python analysis/embedding_pair_global.py \
      --projection experiments_outputs/embedding_map_rerun/embedding_projection.csv \
      --sources cs1.5 cs4.0 --out experiments_outputs/embedding_map_rerun/umap_pair_global_cs.png
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "cs1.5": "#5aa832",
    "cs4.0": "#e0a526",
    "aug_train": "#2f9e6f",
    "aug_popular": "#d2691e",
    "blendA": "#7b5ea7",
    "blendA_g3cs4": "#d1495b",
}


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def metric(path, source):
    if not os.path.exists(path):
        return None
    for r in read_rows(path):
        if r["source"] == source and r["grade"] == "all":
            return r
    return None


def pair_metric(path, source):
    if not os.path.exists(path):
        return None
    for r in read_rows(path):
        if r["a"] == "real" and r["b"] == source and r["grade"] == "all":
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--method", default="umap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--metrics", default="",
                    help="embedding_metrics.csv (for silhouette)")
    ap.add_argument("--pairwise", default="",
                    help="pairwise_metrics.csv (for kNN/energy)")
    args = ap.parse_args()

    rows = [r for r in read_rows(args.projection) if r["method"] == args.method]
    if not rows:
        raise SystemExit(f"no {args.method} rows in {args.projection}")

    def xy(src):
        return np.array([[float(r["x"]), float(r["y"])] for r in rows
                         if r["source"] == src], dtype=np.float32)

    real = xy("real")
    panels = [(s, xy(s)) for s in args.sources]

    allpts = np.concatenate([real] + [p for _, p in panels])
    lim_x = np.percentile(allpts[:, 0], [0.5, 99.5])
    lim_y = np.percentile(allpts[:, 1], [0.5, 99.5])
    pad = 0.06 * np.array([lim_x[1] - lim_x[0], lim_y[1] - lim_y[0]])

    fig, axes = plt.subplots(1, len(panels), figsize=(7.4 * len(panels), 6.0),
                             sharex=True, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (src, pts) in zip(axes, panels):
        ax.scatter(real[:, 0], real[:, 1], s=5, c="#c7c7c7", alpha=0.55,
                   linewidths=0, rasterized=True, label="Real (train)")
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c=COLORS.get(src, "#333333"),
                   alpha=0.65, linewidths=0, rasterized=True, label=src)
        ax.set_xlim(lim_x[0] - pad[0], lim_x[1] + pad[0])
        ax.set_ylim(lim_y[0] - pad[1], lim_y[1] + pad[1])
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_xlabel(f"{args.method.upper()}-1", fontsize=10)
        ax.set_title(f"Real vs {src}  (n={len(pts)})", fontsize=13)
        # metric annotation
        pm = pair_metric(args.pairwise, src) if args.pairwise else None
        em = metric(args.metrics, src) if args.metrics else None
        bits = []
        if pm:
            bits.append(f"kNN-real {float(pm['knn_a_frac']):.3f}")
            bits.append(f"energy {float(pm['energy_distance']):.3f}")
        if em and em.get("silhouette_grade"):
            bits.append(f"silhouette {float(em['silhouette_grade']):+.3f}")
        if bits:
            ax.text(0.02, 0.02, "  |  ".join(bits), transform=ax.transAxes,
                    fontsize=10, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
        ax.legend(loc="upper right", fontsize=9, frameon=False, markerscale=3)
    axes[0].set_ylabel(f"{args.method.upper()}-2", fontsize=10)
    if args.title:
        fig.suptitle(args.title, fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=190, bbox_inches="tight")
    print("[write]", args.out)


if __name__ == "__main__":
    main()
