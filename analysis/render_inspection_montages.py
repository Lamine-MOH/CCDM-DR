#!/usr/bin/env python3
"""
render_inspection_montages.py (recreation of the A3 human-inspection montages)

Builds per-grade montages showing REAL images (top) vs CCDM SYNTHETIC images
(bottom) so a human ophthalmologist / reviewer can judge whether lesion
structure (MAs, hemorrhages, exudates) is present, plausible, and progresses
correctly with grade.

Outputs (per grade 0-4):
  real_vs_synthetic_grade_{g}.png        (2 rows x 12 cols, real top / synth bottom)
  all_grades_compare.png                 (10 rows x 12 cols)
  README_viewing_guide.md
"""

import argparse
import os

import h5py
import numpy as np
from PIL import Image


def load_h5(path):
    with h5py.File(path, "r") as hf:
        return hf["images"][:], hf["labels"][:]


def pick_per_grade(images, labels, grades, n_per_grade, seed):
    rng = np.random.RandomState(seed)
    out = {}
    for g in grades:
        idx = np.where(labels == g)[0]
        if len(idx) > n_per_grade:
            idx = rng.choice(idx, size=n_per_grade, replace=False)
        imgs = images[idx].transpose(0, 2, 3, 1)  # N,H,W,3 uint8
        out[g] = imgs
    return out


def paste_montage(canvas, imgs, start_row, cell, pad):
    for j, img in enumerate(imgs):
        y0 = start_row * (cell + pad) + pad
        x0 = j * (cell + pad) + pad
        canvas[y0 : y0 + cell, x0 : x0 + cell] = img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_h5", required=True)
    ap.add_argument("--synth_h5", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_grade", type=int, default=12)
    ap.add_argument("--cell", type=int, default=128)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--seed", type=int, default=111)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grades = [0, 1, 2, 3, 4]
    imsize = args.cell
    pad = args.pad
    stride = imsize + pad

    real_images, real_labels = load_h5(args.real_h5)
    syn_images, syn_labels = load_h5(args.synth_h5)
    real = pick_per_grade(real_images, real_labels, grades, args.n_per_grade, args.seed)
    syn = pick_per_grade(syn_images, syn_labels, grades, args.n_per_grade, args.seed)

    counts_real = {g: int((real_labels == g).sum()) for g in grades}
    counts_syn = {g: int((syn_labels == g).sum()) for g in grades}

    for g in grades:
        n_real = len(real[g])
        n_syn = len(syn[g])
        cols = max(n_real, n_syn, 1)
        canvas = np.full((2 * stride + 1, cols * stride + 1, 3), 0, dtype=np.uint8)
        paste_montage(canvas, real[g], 0, imsize, pad)
        paste_montage(canvas, syn[g], 1, imsize, pad)
        Image.fromarray(canvas).save(os.path.join(args.out_dir, f"real_vs_synthetic_grade_{g}.png"))
        print(f"  grade {g}: real {n_real} / synth {n_syn}")

    canvas = np.full((10 * stride + 1, 12 * stride + 1, 3), 0, dtype=np.uint8)
    for g in grades:
        paste_montage(canvas, real[g], g * 2, imsize, pad)
        paste_montage(canvas, syn[g], g * 2 + 1, imsize, pad)
    Image.fromarray(canvas).save(os.path.join(args.out_dir, "all_grades_compare.png"))

    guide = [
        "# Visual inspection guide",
        "",
        "Each montage shows REAL images (top row) vs CCDM SYNTHETIC images (bottom row).",
        "",
        "## Check items per image",
        "- Are lesions visible at all (microaneurysms = tiny dark-red dots, hemorrhages = blotches, exudates = bright yellowish)?",
        "- Does grade increase correspond to MORE / LARGER lesions in synthetic?",
        "- Are synthetic lesions anatomically plausible (inside retina, near vessels) or random blobs / smudges?",
        "- Background smoothness: synthetic images often look 'airbrushed'; note if vessel detail vanishes.",
        "- Are grade-0 synthetic images clean (no lesions) or do they show spurious dots?",
        "",
        "## Per-grade sample counts used",
    ]
    guide.append("| grade | real n | synthetic n |")
    guide.append("|-------|--------|-------------|")
    for g in grades:
        guide.append(f"| {g} | {counts_real[g]} | {counts_syn[g]} |")
    guide.append("")
    guide.append(f"- real h5: {args.real_h5}")
    guide.append(f"- synth h5: {args.synth_h5}")
    guide.append(f"- montage cell: {imsize}x{imsize}px, n_per_grade={args.n_per_grade}")
    with open(os.path.join(args.out_dir, "README_viewing_guide.md"), "w") as f:
        f.write("\n".join(guide) + "\n")

    print(f"\n wrote montages to {args.out_dir}")


if __name__ == "__main__":
    main()