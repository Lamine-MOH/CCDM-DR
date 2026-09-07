#!/usr/bin/env python3
"""
sweep_cond_scale.py (E2) — is conditioning expressible at higher CFG?

Generates, from the CURRENT diffusion checkpoint, grade-0 and grade-4 image
batches at several classifier-free-guidance strengths (cond_scale). For each
scale it computes the same 7-feature schema used by trace_conditioning.py and
reports how separable grade-0 vs grade-4 become.

Decisive interpretation:
  - separation RISES with cond_scale  -> conditioning was learned but under-
    amplified at sampling: no retrain needed, just sample at higher CFG.
  - FLAT (~chance binary RF ~0.5, tiny Cohen's d) at all scales -> the model
    never encoded grade; retrain with aux_reg_loss is required.

Usage (mirrors generate_from_ckpt.py args):
  python analysis/sweep_cond_scale.py \
      --model_ckpt output/DRGrading_128/setup1_dr/results/model-100000.pt \
      --model_config config/model_cfg/unet_edm_128_v1.yaml \
      --root_path . --image_size 128 \
      --out_dir output/sweep_cond_scale --grades 0 4 \
      --nfake 64 --cond_scales 1.5 3 6
"""

import argparse
import os
import sys

import numpy as np
import torch
import torchvision

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from generate_from_ckpt import (  # noqa: E402
    build_diffusion,
    build_sigma_data_fn,
    load_ema_model,
    make_embed_fns,
    sample_for_grade,
)
import trace_conditioning as tc  # same 7-feature schema (analysis/ is on sys.path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--root_path", default=".")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--num_channels", type=int, default=3)
    p.add_argument("--max_label", type=float, default=4.0)

    p.add_argument("--path_y2h", default=None)
    p.add_argument("--path_y2cov", default=None)
    p.add_argument("--y2h_ckpt_name", default="ckpt_mlp_y2h_epoch_500.pth")
    p.add_argument("--y2cov_ckpt_name", default="ckpt_cnn_y2cov_epoch_500.pth")
    p.add_argument("--dim_embed", type=int, default=128)
    p.add_argument("--use_y2cov", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--y2cov_hy_weight_train", type=float, default=0.05)
    p.add_argument("--y2cov_hy_weight_test", type=float, default=0.05)

    p.add_argument("--edm_sigma_data_default", type=float, default=0.5)
    p.add_argument("--edm_sigma_min", type=float, default=0.002)
    p.add_argument("--edm_sigma_max", type=float, default=80)
    p.add_argument("--edm_rho", type=float, default=7)
    p.add_argument("--edm_P_mean", type=float, default=-1.2)
    p.add_argument("--edm_P_std", type=float, default=1.2)
    p.add_argument("--edm_S_churn", type=float, default=80)
    p.add_argument("--edm_S_tmin", type=float, default=0.05)
    p.add_argument("--edm_S_tmax", type=float, default=50)
    p.add_argument("--edm_S_noise", type=float, default=1.003)

    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_update_after_step", type=int, default=0)
    p.add_argument("--ema_update_every", type=int, default=10)

    p.add_argument("--sampler", default="sde", choices=["sde", "ode", "dpmpp"])
    p.add_argument("--num_sample_steps", type=int, default=32)
    p.add_argument("--rescaled_phi", type=float, default=0.7)

    p.add_argument("--grades", type=int, nargs="+", default=[0, 4])
    p.add_argument("--nfake", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cond_scales", type=float, nargs="+", default=[1.5, 3.0, 6.0])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2) + 1e-12)
    return (b.mean() - a.mean()) / sp


def main():
    args = parse_args()
    torch.manual_seed(111)
    np.random.seed(111)
    os.makedirs(args.out_dir, exist_ok=True)

    fn_y2h, fn_y2cov = make_embed_fns(args, args.device)
    diffusion = build_diffusion(args, build_sigma_data_fn(args.edm_sigma_data_default),
                                fn_y2cov if args.use_y2cov else None)
    gen_model = load_ema_model(diffusion, args)

    FEAT_NAMES = ["brightness", "R", "G", "B", "std", "RminusG", "edge_mag"]
    rows = []

    for cs in args.cond_scales:
        args.cond_scale = cs
        feats_by_grade, imgs_by_grade = {}, {}
        for g in args.grades:
            imgs = sample_for_grade(gen_model, fn_y2h, grade=g, nfake=args.nfake, args=args)
            imgs_np = imgs.numpy().transpose(0, 2, 3, 1).astype(np.float32)  # N,H,W,3 in [0,255]
            feats = np.array([tc.per_image_features(img) for img in imgs_np])
            feats_by_grade[g] = feats
            imgs_by_grade[g] = imgs

            preview = imgs[:36].float() / 255.0
            torchvision.utils.save_image(
                preview, os.path.join(args.out_dir, "cs{}_grade{}.png".format(cs, g)),
                nrow=6, normalize=False, padding=1)

        g0, g1 = args.grades
        a, b = feats_by_grade[g0], feats_by_grade[g1]

        ds = {f: cohen_d(a[:, i], b[:, i]) for i, f in enumerate(FEAT_NAMES)}
        rf_acc = binary_rf_separability(a, b, seed=111)

        print("\n == cond_scale {:.2f}: grade {} vs {} (chance {:.2f}) ==".format(cs, g0, g1, 0.5))
        print("    mean(g{})  /  mean(g{})".format(g0, g1))
        for i, f in enumerate(FEAT_NAMES):
            print("    {:8s}: {:7.2f} / {:7.2f}   d={:+.3f}".format(
                f, a[:, i].mean(), b[:, i].mean(), ds[f]))
        print("    interpreted: {:.2f} vs {:.2f} (raw), binary RF acc = {:.1%}".format(
            a[:, 0].mean(), b[:, 0].mean(), rf_acc))

        rows.append({
            "cond_scale": cs, "grade": str(g0), "rf_binary_acc": rf_acc,
            **{"d_" + f: ds[f] for f in FEAT_NAMES},
            **{f + "_g0": a[:, i].mean() for i, f in enumerate(FEAT_NAMES)},
            **{f + "_g1": b[:, i].mean() for i, f in enumerate(FEAT_NAMES)},
        })

        np.save(os.path.join(args.out_dir, "features_cs{}_{}v{}.npy".format(cs, g0, g1)),
                np.concatenate([a, b], axis=0))
        print("    saved previews cs{}_grade*.png + features_cs{}.npy".format(cs, cs))

    import csv

    with open(os.path.join(args.out_dir, "cond_scale_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: r["rf_binary_acc"])
    print("\n BEST cond_scale: {:.2f} (binary RF acc {:.1%})".format(best["cond_scale"], best["rf_binary_acc"]))
    if best["rf_binary_acc"] >= 0.8:
        print(" -> conditioning IS expressible at higher CFG. Sample final fake data at this cond_scale.")
    elif best["rf_binary_acc"] <= 0.6:
        print(" -> flat across scales. The model did not encode grade. Enable aux_reg_loss and resume retraining.")
    else:
        print(" -> weak trend. Try even higher cond_scale (e.g. 8-10) or proceed to aux_reg_loss.")
    print("\n summary: {}/cond_scale_summary.csv".format(args.out_dir))


def binary_rf_separability(a, b, seed=111):
    try:
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return float("nan")
    X = np.concatenate([a, b], axis=0)
    y = np.array([0] * len(a) + [1] * len(b))
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, X, y, cv=cv, n_jobs=-1)))


if __name__ == "__main__":
    main()