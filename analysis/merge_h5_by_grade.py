"""merge_h5_by_grade.py — per-grade assembly of synthetic h5 files (blend step).

generate_from_ckpt.py writes one h5 per cond_scale (all grades). This script
re-assembles those files so each grade is sourced from a different scale, e.g.
grades 0,1,4 @ CFG4 and grades 2,3 @ CFG6, matching the per-grade CFG that
maximized downstream per-class recall. Schema is identical to
generate_from_ckpt.py: images uint8 N x 3 x H x W, labels float64 raw grades 0-4.

Usage:
    python analysis/merge_h5_by_grade.py --sources \
        "0=output/generated_cfg4/generated.h5 1=output/generated_cfg4/generated.h5 \
         2=output/generated_cfg6/generated.h5 3=output/generated_cfg6/generated.h5 \
         4=output/generated_cfg4/generated.h5" \
        --out output/generated_blend/generated.h5 --cap 1000
"""

import argparse
import os

import h5py
import numpy as np


def parse_spec(spec):
    mapping = {}
    for token in spec.split():
        if "=" not in token:
            raise ValueError("each source token must be grade=path, got '{}'".format(token))
        grade_s, path = token.split("=", 1)
        grade = int(grade_s)
        if grade in mapping:
            raise ValueError("grade {} mapped twice".format(grade))
        mapping[grade] = path
    return mapping


def main():
    p = argparse.ArgumentParser(description="Per-grade blend of generated h5 files")
    p.add_argument("--sources", required=True,
                   help="whitespace-separated grade=path pairs, e.g. '0=a.h5 1=a.h5 2=b.h5'")
    p.add_argument("--out", default="output/generated_blend/generated.h5")
    p.add_argument("--cap", type=int, default=1000, help="max images per grade in output")
    p.add_argument("--caps_override", default=None,
                   help="optional whitespace-separated grade=cap overrides, e.g. '4=500'")
    p.add_argument("--seed", type=int, default=111)
    args = p.parse_args()

    mapping = parse_spec(args.sources)
    cap_override = {}
    if args.caps_override:
        for g, v in parse_spec(args.caps_override).items():
            cap_override[g] = int(v)

    cache = {}
    images_out, labels_out = [], []
    for grade in sorted(mapping):
        path = mapping[grade]
        if path not in cache:
            with h5py.File(path, "r") as f:
                cache[path] = (f["images"][:], f["labels"][:])
        images, labels = cache[path]
        idx = np.where(np.round(labels) == grade)[0]
        if len(idx) == 0:
            raise ValueError("grade {} not present in {}".format(grade, path))
        cap = cap_override.get(grade, args.cap)
        if len(idx) > cap:
            rng = np.random.RandomState(args.seed + grade)
            idx = idx[rng.permutation(len(idx))[:cap]]
        images_out.append(images[idx])
        n_taken = len(idx)
        labels_out.append(np.full(n_taken, grade, dtype=np.float64))
        print("  grade {}: took {} from {}".format(grade, n_taken, path))

    images_all = np.concatenate(images_out, axis=0)
    labels_all = np.concatenate(labels_out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("images", data=images_all, dtype="uint8",
                         compression="gzip", compression_opts=6)
        f.create_dataset("labels", data=labels_all, dtype="float64")
    print("\n Wrote {} images (shape {}) -> {}".format(len(images_all), images_all.shape, args.out))
    for g in range(5):
        print("   grade {}: {}".format(g, int(np.sum(labels_all == g))))


if __name__ == "__main__":
    main()