"""lesion_audit.py — quantitative realism audit of synthetic vs real fundus (Batch A2).

Replaces eyeball montage comparison with numbers, per grade (0-4):

Per-image content stats (pure numpy, on the green channel unless noted):
    highfreq_ratio  fraction of 2D-FFT energy above 1/16 cycles/pixel
                    (low = smooth/"airbrushed", the CCDM artifact we look for)
    edge_energy     mean |gradient| magnitude of green channel
    local_noise     mean of per-8x8-block green std (background texture)
    redness         mean(R) - mean(G)
    green_mean/std  green-channel intensity stats

Optional lesion-presence via true-class Grad-CAM (needs a trained CNN ckpt):
    cam_trueclass   mean Grad-CAM of the predicted class over the retina mask
                    (green > 20th percentile), i.e. "where the model looks"

Input: --real h5 + one or more --sets h5 (same schema as generate_from_ckpt.py).
Output into --out_dir:
    realism_stats.csv        group,grade,mean/std per feature
    lesion_cam.csv           group,grade,mean_trueclass_cam
    realism_by_grade.png     real vs each synthetic set
    lesion_presence_by_grade.png

Usage:
    python analysis/lesion_audit.py \
        --real data/DRGrading/Aptos/DRGrading_128x128_train.h5 \
        --sets output/generated_cfg4/generated.h5 output/generated_cs1.5/generated.h5 \
               output/generated_blendA/generated.h5 \
        --ckpt downstream_results/real_only_s111_best.pth --backbone densenet121 \
        --out_dir output/audit --img_size 128 --batch_size 32
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from downstream_eval.train_dr_classifier import build_model
from downstream_eval.train_dr_classifier import DRDataset

from torch.utils.data import DataLoader

GRADES = (0, 1, 2, 3, 4)


def content_stats(img):
    """img: uint8 CHW. Returns per-image realism features."""
    g = img[1].astype(np.float64)
    r = img[0].astype(np.float64)

    n = g.shape[0]
    G = g - g.mean()
    F = np.fft.rfft2(G)
    E = np.abs(F) ** 2
    fx = np.fft.fftfreq(n)
    fy = np.fft.rfftfreq(n)
    R = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2)
    high = E[R > 1.0 / 16.0].sum()
    total = E.sum()
    if total <= 0:
        high_ratio = 0.0
    else:
        high_ratio = high / total

    gy, gx = np.gradient(g)
    edge = np.sqrt(gx ** 2 + gy ** 2).mean()

    b = 8
    sh, sw = g.shape[0] // b, g.shape[1] // b
    blocks = g[: sh * b, : sw * b].reshape(sh, b, sw, b)
    local_noise = blocks.std(axis=(1, 3)).mean()

    green_mean = float(g.mean())
    green_std = float(g.std())
    redness = float(r.mean() - g.mean())
    return {
        "highfreq_ratio": high_ratio,
        "edge_energy": float(edge),
        "local_noise": float(local_noise),
        "redness": redness,
        "green_mean": green_mean,
        "green_std": green_std,
    }


def cam_trueclass(model, loader, device):
    """Mean true-class Grad-CAM over the retina mask per image (order-matched)."""
    import torch
    import torch.nn.functional as F

    acts, grads = {}, {}

    def hook_fwd(m, _in, out):
        acts[m] = out

    def hook_bwd(m, _gin, gout):
        grads[m] = gout[0]

    target = None
    for name, mod in model.named_modules():
        if name == "features":
            target = mod
            break
    if target is None:
        raise ValueError("no 'features' module for Grad-CAM (CNN backbones only)")

    fwd_h = target.register_forward_hook(hook_fwd)
    bwd_h = target.register_full_backward_hook(hook_bwd)

    cam_means = []
    model.eval()
    for imgs, _labels in loader:
        x = imgs.to(device)
        x.requires_grad_(True)
        model.zero_grad()
        logits = model(x)
        pred = logits.argmax(dim=1)
        score = logits.gather(1, pred.view(-1, 1)).mean()
        score.backward()
        A = acts[target]
        w = grads[target].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.detach().cpu().numpy()
        xn = x.detach().cpu().numpy()
        # retina mask: green channel > 20th percentile within the image
        for i in range(len(xn)):
            g = xn[i, 1]
            mask = g > np.percentile(g, 20)
            cam_i = cam[i, 0]
            cam_means.append(float(cam_i[mask].mean()))
        acts.clear()
        grads.clear()

    fwd_h.remove()
    bwd_h.remove()
    return np.array(cam_means)


def group_stats(features_by_grade, key):
    rows = []
    for g in GRADES:
        vals = features_by_grade.get(g, [])
        if vals:
            rows.append((g, float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0, len(vals)))
    return rows


def main():
    p = argparse.ArgumentParser(description="Quantitative realism audit of synthetic fundus")
    p.add_argument("--real", required=True, help="real training h5 (reference)")
    p.add_argument("--sets", nargs="+", required=True, help="synthetic h5 sets to audit")
    p.add_argument("--ckpt", default=None, help="classifier state_dict for Grad-CAM")
    p.add_argument("--backbone", default="densenet121")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", default="output/audit")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def load(path):
        with h5py.File(path, "r") as f:
            return f["images"][:], np.round(f["labels"][:]).astype(np.int64)

    groups = [("real", args.real)]
    for s in args.sets:
        groups.append((os.path.splitext(os.path.basename(s))[0], s))

    device = None
    model = None
    if args.ckpt is not None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(args.backbone, pretrained=False).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
        print("[audit] Grad-CAM enabled with {} / {}".format(args.backbone, args.ckpt))

    stat_rows = []
    cam_rows = []
    cam_plot = {}
    for name, path in groups:
        images, labels = load(path)
        feats = {}
        for g in GRADES:
            feats[g] = []
        chunk = 256
        for st in range(0, len(images), chunk):
            en = min(st + chunk, len(images))
            labs = labels[st:en]
            for img, lab in zip(images[st:en], labs):
                feats[int(lab)].append(content_stats(img))
        print("[audit] {}: {} images".format(name, len(images)))
        for g in GRADES:
            n_g = len(feats[g])
            if n_g == 0:
                continue
            keys = list(feats[g][0].keys())
            means = {k: float(np.mean([f[k] for f in feats[g]])) for k in keys}
            stds = {k: float(np.std([f[k] for f in feats[g]])) for k in keys}
            row = [name, g, n_g]
            for k in keys:
                row += [means[k], stds[k]]
            stat_rows.append(row)
            for k in keys:
                cam_plot.setdefault(k, {}).setdefault(name, {}).setdefault(g, means[k])

        if model is not None:
            ds = DRDataset(images, labels, train=False, img_size=args.img_size)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            cam = cam_trueclass(model, loader, device)
            for g in GRADES:
                vals = cam[labels == g]
                if len(vals):
                    cam_rows.append([name, g, len(vals), float(vals.mean()), float(vals.std())])
                    cam_plot.setdefault("cam_trueclass", {}).setdefault(name, {})[g] = float(vals.mean())

    feat_keys = list(content_stats(np.zeros((3, args.img_size, args.img_size), dtype=np.uint8)).keys())
    with open(os.path.join(args.out_dir, "realism_stats.csv"), "w") as f:
        f.write("group,grade,n," + ",".join("{}_{}".format(k, s) for k in feat_keys for s in ("mean", "std")) + "\n")
        for row in stat_rows:
            f.write(",".join(str(v) for v in row) + "\n")

    if model is not None:
        with open(os.path.join(args.out_dir, "lesion_cam.csv"), "w") as f:
            f.write("group,grade,n,mean,std\n")
            for row in cam_rows:
                f.write(",".join(str(v) for v in row) + "\n")

    if not args.no_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            selected = ["highfreq_ratio", "local_noise", "edge_energy"]
            fig, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected), 3.6), squeeze=False)
            names = [name for name, _ in groups]
            for ax, k in zip(axes[0], selected):
                for name in names:
                    data = cam_plot.get(k, {}).get(name, {})
                    if data:
                        gs = sorted(data)
                        ax.plot(gs, [data[g] for g in gs], marker="o", label=name)
                ax.set_xlabel("grade"); ax.set_ylabel(k); ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "realism_by_grade.png"), dpi=150)
            if model is not None:
                fig2, ax = plt.subplots(figsize=(4.5, 3.6))
                for name in names:
                    data = cam_plot.get("cam_trueclass", {}).get(name, {})
                    if data:
                        gs = sorted(data)
                        ax.plot(gs, [data[g] for g in gs], marker="o", label=name)
                ax.set_xlabel("grade"); ax.set_ylabel("mean true-class Grad-CAM (retina)")
                ax.legend(fontsize=8)
                fig2.tight_layout()
                fig2.savefig(os.path.join(args.out_dir, "lesion_presence_by_grade.png"), dpi=150)
            print("[audit] plots -> {}".format(args.out_dir))
        except Exception as exc:  # pragma: no cover
            print("[audit] matplotlib plot failed (CSV still written): {}".format(exc))

    print("[audit] done -> {}".format(os.path.join(args.out_dir, "realism_stats.csv")))


if __name__ == "__main__":
    main()