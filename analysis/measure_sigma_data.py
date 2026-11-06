#!/usr/bin/env python3
"""Measure EDM sigma_data on a training h5 (no training required).

Matches the normalization used by main.py's `global` / `local` schemes:
    x_norm = (x / 255 - 0.5) / 0.5
and reports the global pixel std plus per-grade stds.

Example:
    python analysis/measure_sigma_data.py \
        --h5 data/DRGrading/Aptos/DRGrading_128x128_train.h5
"""

import argparse
import json
import os

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, help="training h5 path")
    p.add_argument("--images_key", default="images")
    p.add_argument("--labels_key", default="labels")
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--out", default=None, help="optional JSON output path")
    return p.parse_args()


def _finalize(total_sum, total_sumsq, count):
    if count == 0:
        return None
    mean = total_sum / count
    var = max(total_sumsq / count - mean * mean, 0.0)
    return float(np.sqrt(var))


def main():
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")

    g_sum = g_sumsq = 0.0
    g_count = 0
    grade_stats = {}

    with h5py.File(args.h5, "r") as f:
        images = f[args.images_key]
        labels = f[args.labels_key]
        n = images.shape[0]
        if labels.shape[0] != n:
            raise ValueError("images and labels have different lengths: {} vs {}".format(
                n, labels.shape[0]))

        for start in range(0, n, args.chunk_size):
            end = min(start + args.chunk_size, n)
            x = (images[start:end].astype(np.float64) / 255.0 - 0.5) / 0.5
            y = np.rint(labels[start:end].astype(np.float64)).astype(int)

            g_sum += float(x.sum())
            g_sumsq += float(np.square(x).sum())
            g_count += int(x.size)

            for grade in np.unique(y):
                mask = y == grade
                xg = x[mask]
                s = grade_stats.setdefault(int(grade), [0.0, 0.0, 0])
                s[0] += float(xg.sum())
                s[1] += float(np.square(xg).sum())
                s[2] += int(xg.size)

    global_std = _finalize(g_sum, g_sumsq, g_count)
    if global_std is None:
        raise RuntimeError("no pixels found in {}".format(args.h5))
    result = {
        "h5": os.path.abspath(args.h5),
        "n_images": int(n),
        "global": global_std,
        "per_grade": {},
    }
    for grade in sorted(grade_stats):
        total_sum, total_sumsq, count = grade_stats[grade]
        result["per_grade"][str(grade)] = {
            "std": _finalize(total_sum, total_sumsq, count),
            "pixels": int(count),
        }

    print("\n sigma_data measurement (normalized x in [-1, 1])")
    print("   global: {:.6f}  (EDM default 0.5)".format(global_std))
    print("   grade |    std    | pixels")
    for grade, stats in result["per_grade"].items():
        print("   {:>5s} | {:.6f} | {}".format(grade, stats["std"], stats["pixels"]))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("\n Saved measurement to {}".format(os.path.abspath(args.out)))

    print("\n Use --edm_sigma_data_type global (or local) for the retrain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
