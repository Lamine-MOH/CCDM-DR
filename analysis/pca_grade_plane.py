#!/usr/bin/env python3
"""
pca_grade_plane.py — the PCA view of the embedding-map run.

`embedding_map.py` writes UMAP/t-SNE figures and computes all distances in
64-d PCA space, but it never plots the PCA plane itself. This script fills that
gap from the cached feature matrix (`embedding_features.npz`, `--reuse_features`
output), so it needs no h5 data and re-uses the exact same PCA:

    fundus crop -> 48x48 RGB -> flatten -> PCA-64 (seed 111)

It answers one question directly: **which principal components carry the
grade?** In the 2026-10-27 APTOS-128 run, PC1 (37.5 % of variance, brightness /
contrast / redness) explains only 2.8 % of grade variance, while PC2 (15.7 %)
explains 63.5 %. The UMAP layout is driven by PC1, which is why the global map
looks grade-blind even though the grade signal is present in PC2.

Outputs (into --out_dir):
  pca_grade_plane.png   real + aug + synthetic arms in PC2-PC3, coloured by grade
  pca_variance.png      scree (top 10) and between-grade variance share per PC
  pca_metrics.csv       explained variance + eta^2 per PC (long format)
  pca_centroids.csv     per source x grade centroid and within-grade radius

Usage:
  python analysis/pca_grade_plane.py \
      --features experiments_outputs/embedding_map_rerun/embedding_features.npz \
      --out_dir experiments_outputs/embedding_map_rerun
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

GRADE_COLORS = {
    0: "#3b6fb6",  # blue
    1: "#2f9e6f",  # green
    2: "#e0a526",  # amber
    3: "#d1495b",  # red
    4: "#7b5ea7",  # purple
}
GRADE_LABELS = {0: "g0 (no DR)", 1: "g1", 2: "g2", 3: "g3", 4: "g4"}

# Panel order: the real reference, the training augmentation, the guidance
# extremes, then the two blend arms (which differ only in g3).
PANELS = [
    ("real", "Real train"),
    ("aug_train", "Classical aug (training pipeline)"),
    ("cs1.5", "Synthetic cs1.5"),
    ("cs4.0", "Synthetic cs4.0"),
    ("blendA", "blendA"),
    ("blendA_g3cs4", "blendA_g3cs4"),
]


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    return (
        z["features"].astype(np.float32),
        z["source"].astype(str),
        z["grade"].astype(int),
        z["index"].astype(int),
    )


def grade_eta2(values, grades):
    """Share of a component's variance explained by grade (eta^2)."""
    grand = values.mean()
    ssb = sum(
        (grades == k).sum() * (values[grades == k].mean() - grand) ** 2
        for k in np.unique(grades)
    )
    sst = ((values - grand) ** 2).sum()
    return float(ssb / sst) if sst > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="embedding_features.npz from embedding_map.py --reuse_features")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pca_components", type=int, default=64)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--n_pcs_plot", type=int, default=10)
    ap.add_argument("--max_per_grade", type=int, default=0,
                    help="optional per-(source,grade) subsample for the scatter (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X, source, grade, _index = load_cache(args.features)
    print(f"[load] {X.shape[0]} images x {X.shape[1]} features")

    k = min(args.pca_components, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=args.seed)
    Z = pca.fit_transform(X).astype(np.float32)
    evr = pca.explained_variance_ratio_
    print(f"[pca] top {k}: {evr.sum():.4f} explained; "
          f"PC1-3 = {evr[0]:.3f}/{evr[1]:.3f}/{evr[2]:.3f}")

    eta2 = np.array([grade_eta2(Z[:, j], grade) for j in range(args.n_pcs_plot)])
    print(f"[grade] eta^2 PC1={eta2[0]:.3f} PC2={eta2[1]:.3f} PC3={eta2[2]:.3f}")

    # ---- CSV: variance + eta^2 ------------------------------------------------
    with open(os.path.join(args.out_dir, "pca_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pc", "explained_variance_ratio", "cumulative", "grade_eta2"])
        cum = 0.0
        for j in range(len(evr)):
            cum += float(evr[j])
            w.writerow([
                j + 1,
                f"{evr[j]:.6f}",
                f"{cum:.6f}",
                f"{grade_eta2(Z[:, j], grade):.6f}" if j < args.n_pcs_plot else "",
            ])

    # ---- CSV: per source x grade centroid + radius ----------------------------
    with open(os.path.join(args.out_dir, "pca_centroids.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "grade", "n", "pc2_mean", "pc3_mean",
                    "pc2_std", "pc3_std", "radius_mean"])
        for s in sorted(set(source)):
            for g in range(5):
                m = (source == s) & (grade == g)
                if m.sum() == 0:
                    continue
                xy = Z[m][:, [1, 2]]
                c = xy.mean(0)
                r = np.linalg.norm(xy - c, axis=1)
                w.writerow([
                    s, g, int(m.sum()),
                    f"{c[0]:.4f}", f"{c[1]:.4f}",
                    f"{xy[:, 0].std():.4f}", f"{xy[:, 1].std():.4f}",
                    f"{r.mean():.4f}",
                ])

    # ---- scatter --------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    keep = np.ones(len(X), dtype=bool)
    if args.max_per_grade > 0:
        keep[:] = False
        for s in set(source):
            for g in range(5):
                idx = np.flatnonzero((source == s) & (grade == g))
                if len(idx) > args.max_per_grade:
                    idx = rng.choice(idx, args.max_per_grade, replace=False)
                keep[idx] = True

    xy = Z[:, [1, 2]]
    # Shared limits on the dense middle; outliers would otherwise flatten every panel.
    lim_x = np.percentile(xy[keep, 0], [0.5, 99.5])
    lim_y = np.percentile(xy[keep, 1], [0.5, 99.5])
    pad_x = 0.06 * (lim_x[1] - lim_x[0])
    pad_y = 0.06 * (lim_y[1] - lim_y[0])

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2), sharex=True, sharey=True)
    real_mask = source == "real"
    for ax, (s, title) in zip(axes.ravel(), PANELS):
        m = (source == s) & keep
        if s != "real":
            bg = real_mask & keep
            ax.scatter(xy[bg, 0], xy[bg, 1], s=3, c="#c9c9c9", alpha=0.35,
                       linewidths=0, rasterized=True, zorder=1)
        for g in range(5):
            gm = m & (grade == g)
            if gm.sum() == 0:
                continue
            ax.scatter(xy[gm, 0], xy[gm, 1], s=5, c=GRADE_COLORS[g],
                       alpha=0.55, linewidths=0, rasterized=True,
                       label=GRADE_LABELS[g], zorder=2)
            # grade centroid marker
            ax.scatter(xy[gm, 0].mean(), xy[gm, 1].mean(), s=90, marker="x",
                       c=GRADE_COLORS[g], linewidths=2.0, zorder=4)
        ax.set_title(f"{title}  (n={int(m.sum())})", fontsize=11)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_xlim(lim_x[0] - pad_x, lim_x[1] + pad_x)
        ax.set_ylim(lim_y[0] - pad_y, lim_y[1] + pad_y)
        # grade-0 / DR separation guide
        g0 = xy[(source == "real") & (grade == 0), 0].mean()
        ax.axvline(g0, color="#777777", lw=0.8, ls=":", zorder=0)

    for ax in axes[-1]:
        ax.set_xlabel(f"PC2  ({evr[1] * 100:.1f}% variance, grade $\\eta^2$={eta2[1]:.2f})",
                      fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC3  ({evr[2] * 100:.1f}% variance)", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "PC2-PC3 grade plane (PCA-64 fit on all sources, seed 111) — "
        f"PC1 carries {evr[0] * 100:.1f}% of variance but only grade $\\eta^2$={eta2[0]:.2f}; "
        f"PC2 carries {evr[1] * 100:.1f}% and $\\eta^2$={eta2[1]:.2f}",
        fontsize=12.5)
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    out_scatter = os.path.join(args.out_dir, "pca_grade_plane.png")
    fig.savefig(out_scatter, dpi=180)
    plt.close(fig)
    print(f"[write] {out_scatter}")

    # ---- variance / eta^2 figure ---------------------------------------------
    n = args.n_pcs_plot
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    pcs = np.arange(1, n + 1)
    axes[0].bar(pcs, evr[:n] * 100, color="#4c72b0", width=0.72)
    axes[0].plot(pcs, np.cumsum(evr[:n]) * 100, color="#222222", marker="o",
                 ms=4, lw=1.4, label="cumulative")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("% of total variance")
    axes[0].set_title("Scree: appearance dominates PC1")
    axes[0].set_xticks(pcs)
    axes[0].legend(frameon=False, fontsize=9, loc="center right")
    axes[0].grid(axis="y", alpha=0.2)

    colors = ["#c44e52" if j == 1 else "#8c8c8c" for j in range(n)]
    axes[1].bar(pcs, eta2 * 100, color=colors, width=0.72)
    axes[1].set_xlabel("principal component")
    axes[1].set_ylabel("% of component variance explained by grade ($\\eta^2$)")
    axes[1].set_title("Grade lives in PC2, not PC1")
    axes[1].set_xticks(pcs)
    axes[1].axhline(20, color="#999999", ls=":", lw=1)
    axes[1].text(n, 21, "chance for 5 balanced classes", ha="right", va="bottom",
                 fontsize=8, color="#666666")
    axes[1].grid(axis="y", alpha=0.2)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_var = os.path.join(args.out_dir, "pca_variance.png")
    fig.savefig(out_var, dpi=180)
    plt.close(fig)
    print(f"[write] {out_var}")


if __name__ == "__main__":
    main()
