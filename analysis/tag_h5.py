"""tag_h5.py — stamp generation attrs onto a legacy generated h5 file.

h5 files written by generate_from_ckpt.py after the self-describing-attr change
carry `cond_scale`/`model_ckpt`/`max_label`/`sampler`/`num_sample_steps` as
file attributes; older files don't. This script stamps those attrs onto legacy
files in place (non-destructive, never overwrites existing attrs) so that
analysis/merge_h5_by_grade.py prints correct provenance for them too.

Usage:
    python analysis/tag_h5.py --h5 output/generated/generated.h5 --cond_scale 1.5
    python analysis/tag_h5.py --h5 a.h5 b.h5 --cond_scale 4 --max_label 4 \
        --model_ckpt output/DRGrading_128/setup1_dr/results/model-100000.pt \
        --sampler sde --num_sample_steps 32
"""

import argparse

import h5py


def main():
    p = argparse.ArgumentParser(description="Stamp generation attrs onto legacy generated h5 files")
    p.add_argument("--h5", type=str, nargs="+", required=True)
    p.add_argument("--cond_scale", type=float, default=None)
    p.add_argument("--max_label", type=float, default=None)
    p.add_argument("--model_ckpt", type=str, default=None)
    p.add_argument("--sampler", type=str, default=None)
    p.add_argument("--num_sample_steps", type=int, default=None)
    args = p.parse_args()

    desired = {
        "cond_scale": args.cond_scale,
        "max_label": args.max_label,
        "model_ckpt": args.model_ckpt,
        "sampler": args.sampler,
        "num_sample_steps": args.num_sample_steps,
    }

    for path in args.h5:
        with h5py.File(path, "r+") as f:
            for key, val in desired.items():
                if val is None:
                    continue
                if key in f.attrs:
                    print("  {}: {} already set (skip)".format(path, key))
                    continue
                f.attrs[key] = val
                print("  {}: set {} = {}".format(path, key, val))
        with h5py.File(path, "r") as f:
            print("  {} attrs now: {}".format(path, dict(f.attrs)))


if __name__ == "__main__":
    main()