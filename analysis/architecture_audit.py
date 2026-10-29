#!/usr/bin/env python3
"""
architecture_audit.py — the quantitative architecture analysis for CCDM-DR.

Three questions, answered from the actual model code and checkpoints (not from
estimates):

1. WHERE DO THE PARAMETERS GO?
   Per-group counts for the 128 px EDM U-Net: time path, label path
   (`cond_map` + per-block `affine_cond`), attention, residual convs, norms.
   This is what separates a *re-parameterisation* (`affine_cond`) from a real
   capacity change.

2. WHAT DOES A FORWARD PASS COST?
   FLOPs for one 128 px forward (batch 1) and the full-matrix attention budget
   per resolution. The latter is why the fork needs a chunked/SDPA backend:
   at 64 px one attention op materialises a 4096 x 4096 fp32 score matrix per
   head, and the legacy op saves it for backward.

3. IS THE CONDITIONING PATH INFORMATIVE?
   Loads the frozen `y2h` / `y2cov` checkpoints and reports the 5x5 grade
   embedding geometry (cosine, L2, h2y reconstruction) plus the label-dependent
   covariance map `tilde_h(y) = exp(-y2cov(y))` per grade.

Everything runs on CPU. The attention numbers are the analytical full-matrix
budget of the legacy `AttentionOp`; measured GPU peaks per backend can be added
on the training machine without changing the script's outputs.

Outputs (into --out_dir):
  architecture_audit.json       totals + metadata
  architecture_params.csv       parameter budget per group
  architecture_attention.csv    full-matrix attention cost per resolution
  architecture_flops.json       forward FLOPs (if --flops)
  conditioning_geometry.csv     per-grade y2h reconstruction + y2cov stats
  y2h_cosine.png                grade-embedding cosine matrix
  y2cov_grade_maps.png          exp(-y2cov) map per grade

Usage:
  python analysis/architecture_audit.py \
      --out_dir experiments_outputs/architecture_audit
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "models"))

# Import the modules directly: `models/__init__.py` also pulls DiT, which needs
# timm. The audit does not need DiT and should run in a minimal environment.
import unet_edm            # noqa: E402
import resnet_y2h          # noqa: E402
import resnet_y2cov        # noqa: E402

PARAM_GROUPS = [
    ("time_embedding", "time map (noise level)"),
    ("label_cond_map", "cond_map (label projection + LayerNorm)"),
    ("time_affine_per_block", "affine (time FiLM per block)"),
    ("label_affine_per_block", "affine_cond (label FiLM per block)"),
    ("null_label_vector", "null_cond_emb (CFG)"),
    ("attention", "self-attention qkv/proj"),
    ("res_conv", "residual conv0/conv1"),
    ("skip_proj", "skip projections"),
    ("io_conv", "input/output conv"),
    ("norm", "GroupNorm/LayerNorm"),
    ("other", "other"),
]


def classify(name):
    if name.startswith("time_map"):
        return "time_embedding"
    if name.startswith("cond_map"):
        return "label_cond_map"
    if "affine_cond" in name:
        return "label_affine_per_block"
    if ".affine." in name:
        return "time_affine_per_block"
    if name.startswith("null_cond_emb"):
        return "null_label_vector"
    if "qkv" in name or name.endswith(".proj.weight") or name.endswith(".proj.bias"):
        return "attention"
    if ".skip." in name:
        return "skip_proj"
    if name.startswith("in_conv") or name.startswith("out_conv"):
        return "io_conv"
    if ".norm" in name or name.startswith("out_norm"):
        return "norm"
    if ".conv" in name:
        return "res_conv"
    return "other"


def count_params(module):
    rows = {}
    for name, p in module.named_parameters():
        g = classify(name)
        rows[g] = rows.get(g, 0) + int(p.numel())
    total = sum(rows.values())
    return total, rows


def load_state(path, module):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["net_state_dict"] if isinstance(ckpt, dict) and "net_state_dict" in ckpt else ckpt
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"{path}: missing keys {missing[:5]}")
    if unexpected:
        raise RuntimeError(f"{path}: unexpected keys {unexpected[:5]}")
    return module.eval()


def attention_table(model):
    """Full-matrix cost per resolution for the legacy AttentionOp.

    AttentionOp forms `softmax(Q K^T)` in fp32 (L x L per head), saves it for
    backward, and has a second live copy during the softmax. Numbers are for
    batch 1, fp32, one image.
    """
    rows = {}
    for name, block in model.named_modules():
        if not isinstance(block, unet_edm.UNetBlock) or block.num_heads == 0:
            continue
        # names look like enc.64x64_block0 / dec.16x16_block2
        tag = [p for p in name.split(".") if "x" in p and p.split("x")[0].isdigit()]
        res = int(tag[0].split("x")[0]) if tag else None
        if res is None:
            continue
        heads = int(block.num_heads)
        L = res * res
        per_op_mib = heads * L * L * 4 / 1024 ** 2          # saved weights
        peak_mib = 2 * per_op_mib                           # QK^T + softmax live
        r = rows.setdefault(res, {
            "res": res, "channels": int(block.out_channels), "heads": heads,
            "seq_len": L, "n_attention_blocks": 0,
            "per_op_saved_mib": per_op_mib, "per_op_peak_mib": peak_mib,
            "total_saved_mib": 0.0, "total_peak_mib": 0.0,
            "blocks": [],
        })
        r["n_attention_blocks"] += 1
        r["total_saved_mib"] += per_op_mib
        r["total_peak_mib"] += peak_mib
        r["blocks"].append(name)
    return [rows[k] for k in sorted(rows, reverse=True)]


def forward_flops(model, image_size, label_dim):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as exc:  # older torch
        return {"error": f"FlopCounterMode unavailable: {exc}"}
    model.eval()
    x = torch.randn(1, 3, image_size, image_size)
    t = torch.rand(1)
    y = torch.randn(1, label_dim)
    try:
        with torch.no_grad():
            counter = FlopCounterMode(display=False)
            with counter:
                model(x, t, y)
        total = int(counter.get_total_flops())
        return {"forward_flops": total, "forward_gflops": total / 1e9,
                "note": "batch 1, 128 px, eval; conv/matmul only"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def embedding_geometry(y2h_dir, y2cov_dir, image_size, label_dim, out_dir,
                       y2cov_weight=0.05, make_plots=True):
    """E1 in numbers: 5 grade embeddings, their cosine/L2 geometry, inversion error,
    and the label-dependent covariance map produced by y2cov."""
    labels_norm = torch.arange(5, dtype=torch.float32).view(-1, 1) / 4.0
    grades = np.arange(5)

    mlp = resnet_y2h.model_y2h(dim_embed=label_dim)
    mlp = load_state(os.path.join(y2h_dir, "ckpt_mlp_y2h_epoch_500.pth"), mlp)
    resnet = resnet_y2h.ResNet34_embed_y2h(dim_embed=label_dim, nc=3)
    resnet = load_state(os.path.join(y2h_dir, "ckpt_resnet_y2h_epoch_200.pth"), resnet)
    h2y = resnet.h2y

    with torch.no_grad():
        h = mlp(labels_norm).cpu().numpy()
        recon = (h2y(torch.from_numpy(h)).cpu().numpy().reshape(-1)) * 4.0
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    cos = (h @ h.T) / (norms @ norms.T + 1e-8)
    off = cos - np.eye(5)
    min_cos = float(off[~np.eye(5, dtype=bool)].max())
    adjacent_cos = float(np.mean([cos[g, g + 1] for g in range(4)]))
    d2 = np.linalg.norm(h[:, None, :] - h[None, :, :], axis=-1)
    min_l2 = float(d2[~np.eye(5, dtype=bool)].min())
    max_l2 = float(d2[~np.eye(5, dtype=bool)].max())
    err = np.abs(recon - grades)
    # The `check_embedding.py` gate was written for categorical labels: it asks
    # for five distinct clusters (off-diagonal cosine < 0.8). An ordinal label
    # is a one-dimensional axis, so its five embeddings are expected to be
    # nearly collinear. Report both gates and never hide the categorical one.
    categorical_gate = "PASS" if min_cos < 0.8 else "FAIL"
    inversion_gate = "PASS" if err.max() < 0.15 * 4.0 else "FAIL"
    sv = np.linalg.svd(h, compute_uv=False)
    pc1_share = float(sv[0] ** 2 / (sv ** 2).sum())
    if categorical_gate == "PASS" and inversion_gate == "PASS":
        verdict = "PASS"
    elif inversion_gate == "PASS":
        verdict = "ORDINAL (collinear axis, invertible)"
    else:
        verdict = "FAIL"

    # y2cov: exp(-net(y)) is the per-pixel, per-channel noise-scale map.
    cnn = resnet_y2cov.model_cnn_y2cov(img_size=image_size, nc=3)
    cnn = load_state(os.path.join(y2cov_dir, "ckpt_cnn_y2cov_epoch_500.pth"), cnn)
    with torch.no_grad():
        cov = cnn(labels_norm).cpu().numpy().reshape(5, 3, image_size, image_size)
    tilde = np.exp(-cov)

    rows = []
    for g in range(5):
        rows.append({
            "grade": g,
            "recon_grade": float(recon[g]),
            "abs_err": float(err[g]),
            "embedding_norm": float(norms[g, 0]),
            "tilde_h_mean": float(tilde[g].mean()),
            "tilde_h_std": float(tilde[g].std()),
        })
    with open(os.path.join(out_dir, "conditioning_geometry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "min_offdiagonal_cosine": min_cos,
        "mean_adjacent_cosine": adjacent_cos,
        "min_pairwise_l2": min_l2,
        "max_pairwise_l2": max_l2,
        "max_reconstruction_error": float(err.max()),
        "categorical_cosine_gate": categorical_gate,
        "ordinal_inversion_gate": inversion_gate,
        "verdict": verdict,
        "embedding_singular_values": sv.tolist(),
        "embedding_pc1_share": pc1_share,
        "cosine_matrix": cos.tolist(),
        "recon_grade": recon.tolist(),
        "tilde_h_mean_by_grade": [float(tilde[g].mean()) for g in range(5)],
        "tilde_h_std_by_grade": [float(tilde[g].std()) for g in range(5)],
        "tilde_h_mean_abs_diff_vs_g0": [
            float(np.abs(tilde[g] - tilde[0]).mean()) for g in range(5)
        ],
        "tilde_h_max_abs_diff_vs_g0": [
            float(np.abs(tilde[g] - tilde[0]).max()) for g in range(5)
        ],
        "tilde_h_per_pixel_std_across_grades": {
            "mean": float(tilde.std(0).mean()),
            "max": float(tilde.std(0).max()),
        },
        "tilde_h_range_by_grade": [
            [float(tilde[g].min()), float(tilde[g].max())] for g in range(5)
        ],
        # Sigma = sigma^2 + w * tilde_h * sigma. This is the share of the total
        # variance carried by the label-dependent term at three noise levels.
        "y2cov_weight": y2cov_weight,
        "label_cov_share_of_sigma": {
            str(s): float(
                (y2cov_weight * tilde.mean()) / (s + y2cov_weight * tilde.mean())
            ) for s in (0.1, 1.0, 10.0)
        },
    }

    if make_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            im = ax.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r")
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center",
                            fontsize=9, color="black")
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            ax.set_xlabel("grade"); ax.set_ylabel("grade")
            ax.set_title("y2h grade-embedding cosine (128-d)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "y2h_cosine.png"), dpi=180)
            plt.close(fig)

            fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
            vmin, vmax = float(tilde.min()), float(tilde.max())
            for g, ax in enumerate(axes):
                im = ax.imshow(tilde[g].mean(0), cmap="magma", vmin=vmin, vmax=vmax)
                ax.set_title(f"grade {g}", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01,
                         label=r"$\tilde h(y)=\exp(-y2cov(y))$")
            fig.suptitle("Label-dependent noise scale produced by y2cov (mean over RGB)",
                         fontsize=12)
            fig.savefig(os.path.join(out_dir, "y2cov_grade_maps.png"), dpi=180,
                        bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] plots skipped: {exc}")

    np.savez_compressed(os.path.join(out_dir, "conditioning_geometry.npz"),
                        h=h, cos=cos, tilde_h=tilde.astype(np.float32))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/model_cfg/unet_edm_128_v1.yaml")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--out_dir", default="experiments_outputs/architecture_audit")
    ap.add_argument("--y2h_dir", default="experiments_outputs/DRGrading_128/model_y2h")
    ap.add_argument("--y2cov_dir", default="experiments_outputs/DRGrading_128/model_y2cov")
    ap.add_argument("--label_dim", type=int, default=128)
    ap.add_argument("--y2cov_weight", type=float, default=0.05,
                    help="--y2cov_hy_weight_train used by the DR configs")
    ap.add_argument("--skip_flops", action="store_true")
    ap.add_argument("--skip_embeddings", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = dict(cfg["model"])
    model_cfg["img_resolution"] = args.image_size
    model = unet_edm.UNet_EDM(**model_cfg)
    model.eval()

    total, groups = count_params(model)
    blocks = [m for m in model.modules() if isinstance(m, unet_edm.UNetBlock)]
    attn_blocks = [b for b in blocks if b.num_heads]
    print(f"[unet] {total:,} params | {len(blocks)} UNetBlocks | "
          f"{len(attn_blocks)} with attention")

    rows = []
    for key, label in PARAM_GROUPS:
        n = groups.get(key, 0)
        if n:
            rows.append({"group": key, "label": label, "params": n,
                         "share_pct": 100.0 * n / total})
    with open(os.path.join(args.out_dir, "architecture_params.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "label", "params", "share_pct"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "share_pct": f"{r['share_pct']:.4f}"})

    # The upstream concat layer has the same weight matrix; the split only adds
    # a second bias of length 2*C per block.
    added_bias = sum(2 * b.out_channels for b in blocks if b.affine_cond is not None)
    print(f"[label] cond_map={groups.get('label_cond_map', 0):,} | "
          f"affine_cond={groups.get('label_affine_per_block', 0):,} | "
          f"added bias vs concat={added_bias:,}")

    attn = attention_table(model)
    with open(os.path.join(args.out_dir, "architecture_attention.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "res", "channels", "heads", "seq_len", "n_attention_blocks",
            "per_op_saved_mib", "per_op_peak_mib",
            "total_saved_mib", "total_peak_mib", "blocks"])
        w.writeheader()
        for r in attn:
            w.writerow({**r,
                        "per_op_saved_mib": f"{r['per_op_saved_mib']:.2f}",
                        "per_op_peak_mib": f"{r['per_op_peak_mib']:.2f}",
                        "total_saved_mib": f"{r['total_saved_mib']:.2f}",
                        "total_peak_mib": f"{r['total_peak_mib']:.2f}",
                        "blocks": ";".join(r["blocks"])})

    flops = {} if args.skip_flops else forward_flops(model, args.image_size, args.label_dim)
    if flops:
        with open(os.path.join(args.out_dir, "architecture_flops.json"), "w") as f:
            json.dump(flops, f, indent=2)
        print(f"[flops] {flops}")

    geo = {}
    if not args.skip_embeddings:
        geo = embedding_geometry(args.y2h_dir, args.y2cov_dir, args.image_size,
                                 args.label_dim, args.out_dir,
                                 y2cov_weight=args.y2cov_weight)
        print(f"[geometry] y2h {geo['verdict']} | categorical cosine gate "
              f"{geo['categorical_cosine_gate']} (min cos {geo['min_offdiagonal_cosine']:.3f}) | "
              f"inversion gate {geo['ordinal_inversion_gate']} "
              f"(max recon err {geo['max_reconstruction_error']:.3f}) | "
              f"embedding PC1 share {geo['embedding_pc1_share']:.3f}")

    # embedding-net parameter counts (frozen, separate checkpoints)
    emb = {}
    try:
        for name, module in (
            ("y2h_resnet34", resnet_y2h.ResNet34_embed_y2h(dim_embed=args.label_dim, nc=3)),
            ("y2h_mlp", resnet_y2h.model_y2h(dim_embed=args.label_dim)),
            ("y2cov_cnn", resnet_y2cov.model_cnn_y2cov(img_size=args.image_size, nc=3)),
        ):
            emb[name] = int(sum(p.numel() for p in module.parameters()))
    except Exception as exc:
        emb["error"] = str(exc)

    summary = {
        "config": args.config,
        "image_size": args.image_size,
        "unet_total_params": total,
        "n_unet_blocks": len(blocks),
        "n_attention_blocks": len(attn_blocks),
        "param_groups": groups,
        "label_path_params": {
            "cond_map": groups.get("label_cond_map", 0),
            "affine_cond": groups.get("label_affine_per_block", 0),
            "added_bias_vs_upstream_concat": added_bias,
            "share_of_unet_pct": 100.0 * (
                groups.get("label_cond_map", 0) + groups.get("label_affine_per_block", 0)
            ) / total,
        },
        "attention_full_matrix": attn,
        "flops": flops,
        "embedding_nets_params": emb,
        "conditioning_geometry": geo,
    }
    with open(os.path.join(args.out_dir, "architecture_audit.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {os.path.join(args.out_dir, 'architecture_audit.json')}")


if __name__ == "__main__":
    main()
