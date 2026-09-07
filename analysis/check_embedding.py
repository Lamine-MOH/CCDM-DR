#!/usr/bin/env python3
"""
check_embedding.py (E1) — is the label embedding net actually informative?

Loads the trained y2h nets from output/DRGrading_128/model_y2h/ and answers:
  "do the 5 DR grades map to well-separated 128-d embeddings that invert back
   to the correct label?"

If embeddings collapse (high pairwise cosine / bad h2y inversion), the
encoder is still the bottleneck. If they are clean and separated, the problem
is downstream (diffusion conditioning strength) — see sweep_cond_scale.py.

Usage:
  python analysis/check_embedding.py --y2h_dir output/DRGrading_128/model_y2h
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from models.resnet_y2h import ResNet34_embed_y2h, model_y2h


def strip_module_prefix(sd):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def load_state(path, module, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["net_state_dict"] if isinstance(ckpt, dict) and "net_state_dict" in ckpt else ckpt
    sd = strip_module_prefix(sd)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError("{}: missing keys {}".format(path, missing))
    if unexpected:
        raise RuntimeError("{}: unexpected keys {}".format(path, unexpected))
    return module.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y2h_dir", required=True, help="dir with ckpt_mlp_y2h_epoch_500.pth + ckpt_resnet_y2h_epoch_200.pth")
    ap.add_argument("--mlp_ckpt", default="ckpt_mlp_y2h_epoch_500.pth")
    ap.add_argument("--resnet_ckpt", default="ckpt_resnet_y2h_epoch_200.pth")
    ap.add_argument("--dim_embed", type=int, default=128)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--max_label", type=float, default=4.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    mlp = load_state(os.path.join(args.y2h_dir, args.mlp_ckpt), model_y2h(dim_embed=args.dim_embed), device)
    resnet = load_state(os.path.join(args.y2h_dir, args.resnet_ckpt),
                        ResNet34_embed_y2h(dim_embed=args.dim_embed, nc=args.nc), device)
    h2y = resnet.h2y
    print(" loaded mlp_y2h + resnet h2y head from {}".format(args.y2h_dir))

    labels_norm = torch.arange(5, dtype=torch.float32, device=device).view(-1, 1) / args.max_label
    grades = torch.arange(5).numpy()

    with torch.no_grad():
        h = mlp(labels_norm)                       # (5, dim_embed)
        recon_norm = h2y(h).cpu().numpy().reshape(-1)
    h_np = h.cpu().numpy()

    # pairwise cosine similarity of the 5 grade embeddings
    norms = np.linalg.norm(h_np, axis=1, keepdims=True)
    cos = (h_np @ h_np.T) / (norms @ norms.T + 1e-8)
    off = cos - np.eye(5)
    min_cos = off[~np.eye(5, dtype=bool)].max()  # worst off-diagonal cosine

    # pairwise L2 distance
    d2 = np.linalg.norm(h_np[:, None, :] - h_np[None, :, :], axis=-1)
    off_d2 = d2[~np.eye(5, dtype=bool)]
    min_l2 = off_d2.min()

    recon_grade = recon_norm * args.max_label
    err = np.abs(recon_grade - grades)

    print("\n label -> embedding -> h2y reconstruction")
    print("  grade | recon_grade | |err| | ||h||")
    for g, rc, e, nh in zip(grades, recon_grade, err, np.linalg.norm(h_np, axis=1)):
        print("   {:4d}  |    {:5.3f}   | {:4.3f} | {:6.3f}".format(g, rc, e, nh))

    print("\n pairwise cosine similarity matrix (off-diagonal)")
    np.set_printoptions(precision=3, suppress=True)
    print(cos)
    print("\n min off-diagonal cosine: {:.3f}  | min pairwise L2: {:.3f}".format(min_cos, min_l2))

    cos_pass = min_cos < 0.8
    recon_pass = err.max() < 0.15 * args.max_label
    print("\n VERDICT: {}".format(
        "PASS — embeddings well-separated & invertible (encoder healthy)"
        if (cos_pass and recon_pass) else
        "FAIL — embeddings collapsed / not invertible (encoder still the bottleneck)")
    )


if __name__ == "__main__":
    main()