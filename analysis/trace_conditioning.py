#!/usr/bin/env python3
"""
trace_conditioning.py (recreation of the A1 cross-checkpoint conditioning trace)

Goal: given the per-checkpoint `sample_{ode,sde}_<step>.png` grids written to a
training run's `results/` folder, quantify how grade-separable the generated
images are as a function of training step. Flat / low separability across steps
== the label conditioning is not learning; a rising separability that
approaches the (same-script) real-train reference == the fix is working.

Grid layout (built by main.py + trainer.py for DR 128x128):
  - 10 rows x 10 cols, one image per cell
  - row r is conditioned on raw grade r*max_label/(n_rows-1)
    (0.0, 0.444, 0.888, ..., 4.0) -> rounded to the nearest integer grade
    for the 5-class separability metric (chance = 0.20).
  - save_image() is called with padding=1, so each 128x128 cell starts at
    offset pad + i*(cell+pad).

Per-image features (identical schema to the session's per_image_features/*.csv):
  brightness, R, G, B, std, RminusG, edge_mag

Metric: stratified 5-fold RandomForest-CV accuracy of a 5-grade classifier on
those features. Compare against the same metric computed on real training
images (--real_h5) and against chance (0.20).

Usage:
  python analysis/trace_conditioning.py \
      --grid_dir  output/DRGrading_128/setup1_dr/results \
      --out_dir   output/trace_conditioning \
      --real_h5   data/DRGrading/Aptos/DRGrading_128x128_train.h5 \
      --max_label 4
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# --------------------------------------------------------------------------- #
def extract_features_from_grid(png_path, cell=128, pad=1, n_rows=10, n_cols=10):
    """Split a save_image() grid PNG into cells and return per-cell features."""
    img = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32)  # H,W,3
    H, W = img.shape[:2]
    stride = cell + pad
    assert H >= n_rows * stride and W >= n_cols * stride, (
        f"grid {png_path} too small ({H}x{W}) for {n_rows}x{n_cols} cells of {cell}px"
    )

    rows = []
    for i in range(n_rows):
        for j in range(n_cols):
            y0 = pad + i * stride
            x0 = pad + j * stride
            cell_img = img[y0 : y0 + cell, x0 : x0 + cell]  # cell,cell,3
            rows.append(per_image_features(cell_img))
    return np.array(rows)


def per_image_features(cell):
    """Return the 7 features used by the A1 trace (order-sensitive)."""
    r = cell[..., 0]
    g = cell[..., 1]
    b = cell[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    brightness = cell.mean()
    std = cell.std()
    r_chan = r.mean()
    g_chan = g.mean()
    b_chan = b.mean()
    r_minus_g = r_chan - g_chan
    edge_mag = _edge_magnitude(gray).mean()
    return [brightness, r_chan, g_chan, b_chan, std, r_minus_g, edge_mag]


def _edge_magnitude(gray):
    # Sobel magnitude on a float grayscale image (0-255).
    from scipy.ndimage import sobel

    gx = sobel(gray, axis=0, mode="reflect")
    gy = sobel(gray, axis=1, mode="reflect")
    return np.hypot(gx, gy)


# --------------------------------------------------------------------------- #
def rf_separability(features, labels, n_splits=5, n_estimators=200, random_state=111):
    """Stratified RF-CV accuracy; chance = 1/n_classes."""
    labels = np.asarray(labels)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(clf, features, labels, cv=skf, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def row_labels(n_rows, max_label):
    return np.array([r * max_label / (n_rows - 1) for r in range(n_rows)])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_dir", required=True, help="dir containing sample_{ode,sde}_<step>.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--real_h5", default=None, help="optional real train h5 for the same-script reference")
    ap.add_argument("--max_label", type=float, default=4.0)
    ap.add_argument("--n_rows", type=int, default=10)
    ap.add_argument("--n_cols", type=int, default=10)
    ap.add_argument("--cell", type=int, default=128, help="cell size in px (must match trained image_size)")
    ap.add_argument("--pad", type=int, default=1, help="save_image padding used during training")
    ap.add_argument("--cap_per_grade", type=int, default=100, help="real-reference balance cap")
    ap.add_argument("--seed", type=int, default=111)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "per_image_features"), exist_ok=True)

    png_files = [f for f in sorted(os.listdir(args.grid_dir)) if re.match(r"^sample_(ode|sde)_\d+\.png$", f)]
    if not png_files:
        sys.exit(f"no sample_{'{ode,sde}'}_<step>.png grids found in {args.grid_dir}")

    row_labels_arr = row_labels(args.n_rows, args.max_label)
    grade_of_row = np.round(row_labels_arr).astype(int)
    chance = 1.0 / len(np.unique(grade_of_row))

    summaries = []
    per_rows = []
    for fname in png_files:
        m = re.match(r"^sample_(ode|sde)_(\d+)\.png$", fname)
        sampler, step = m.group(1), int(m.group(2))
        feats = extract_features_from_grid(
            os.path.join(args.grid_dir, fname), cell=args.cell, pad=args.pad, n_rows=args.n_rows, n_cols=args.n_cols
        )
        labels = np.repeat(grade_of_row, args.n_cols)
        acc, acc_std = rf_separability(feats, labels, random_state=args.seed)

        df_img = pd.DataFrame(feats, columns=["brightness", "R", "G", "B", "std", "RminusG", "edge_mag"])
        df_img["label"] = np.repeat(row_labels_arr, args.n_cols)
        df_img["step"] = step
        df_img["sampler"] = sampler
        df_img.to_csv(os.path.join(args.out_dir, "per_image_features", f"{sampler}_{step}.csv"), index=False)

        df_row = df_img.groupby(np.repeat(np.arange(args.n_rows), args.n_cols)).mean(numeric_only=True)
        df_row["label"] = row_labels_arr
        df_row["step"] = step
        df_row["sampler"] = sampler
        df_row["rf_acc"] = acc
        per_rows.append(df_row)

        summaries.append({"step": step, "sampler": sampler, "rf_cv_acc": acc, "rf_cv_std": acc_std, "n_imgs": len(feats), "chance": chance})
        print(f"  {sampler}@{step:>6}: rf_cv_acc={acc:.3f} (chance {chance:.2f})")

    summary_df = pd.DataFrame(summaries).sort_values(["step", "sampler"])
    summary_df.to_csv(os.path.join(args.out_dir, "per_step_summary.csv"), index=False)
    pd.concat(per_rows).to_csv(os.path.join(args.out_dir, "per_row_features.csv"), index=False)

    # real reference (same features, balanced per grade)
    real_ref = None
    if args.real_h5:
        try:
            import h5py

            with h5py.File(args.real_h5, "r") as hf:
                imgs, labs = hf["images"][:], hf["labels"][:]
        except Exception as e:
            print(f"  [warn] could not load real_h5 reference: {e}")
        else:
            real_feats, real_labs = [], []
            rng = np.random.RandomState(args.seed)
            for g in np.unique(labs):
                idx = np.where(labs == g)[0]
                if len(idx) > args.cap_per_grade:
                    idx = rng.choice(idx, size=args.cap_per_grade, replace=False)
                for i in idx:
                    im = imgs[i].transpose(1, 2, 0).astype(np.float32)
                    real_feats.append(per_image_features(im))
                    real_labs.append(g)
            real_ref, real_std = rf_separability(np.array(real_feats), np.array(real_labs), random_state=args.seed)
            print(f"  real train reference (balanced {args.cap_per_grade}/grade): rf_cv_acc={real_ref:.3f}")

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for sampler, grp in summary_df.groupby("sampler"):
        ax.plot(grp["step"], grp["rf_cv_acc"], marker="o", label=sampler)
    ax.axhline(chance, color="gray", ls="--", lw=1, label=f"chance ({chance:.2f})")
    if real_ref is not None:
        ax.axhline(real_ref, color="green", ls="--", lw=1.5, label=f"real ref ({real_ref:.2f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("RF 5-grade separability (CV acc)")
    ax.set_title("Grade-conditioning strength vs training step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "trace_plot.png"), dpi=150)
    plt.close(fig)

    late = summary_df.groupby("sampler").last().reset_index()
    lines = [
        "## Conditioning trace report",
        "",
        f"grids: {args.grid_dir} ({len(png_files)} grids, 10x10)",
        f"chance: {chance:.2f}",
        "| step | sampler | rf_cv_acc |",
        "|------|---------|-----------|",
    ]
    for _, r in summary_df.iterrows():
        lines.append(f"| {int(r['step'])} | {r['sampler']} | {r['rf_cv_acc']:.3f} |")
    if real_ref is not None:
        lines.append("")
        lines.append(f"**Real train reference: {real_ref:.3f}** (same script/features)")
    lines.append("")
    lines.append("## Verdict")
    max_acc = float(late["rf_cv_acc"].max())
    target = (real_ref or 0.55) * 0.8
    if real_ref is not None and max_acc >= target:
        lines.append(f"PASS (late-step separability {max_acc:.2f} >= 80% of real reference {real_ref:.2f})")
    elif real_ref is not None and max_acc >= 0.45:
        lines.append(f"IMPROVED but below target (late {max_acc:.2f} vs real {real_ref:.2f})")
    else:
        lines.append(f"FAIL (late-step separability {max_acc:.2f}; old broken run plateaued at ~0.17-0.44, ideally approach the real reference)")
    with open(os.path.join(args.out_dir, "trace_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n wrote: {args.out_dir}/per_step_summary.csv, per_row_features.csv, trace_report.md, trace_plot.png")


if __name__ == "__main__":
    main()