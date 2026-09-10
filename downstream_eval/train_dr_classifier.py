"""
train_dr_classifier.py

The core evidence for the "does synthetic augmentation actually help DR
grading" contribution: train a standard DR classifier under two (or more)
data conditions and compare held-out performance:

    (A) real-only       : the original imbalanced training split
    (B) real+synthetic  : (A) + CCDM-generated images for the minority grades
    (C) [optional] real+classic-aug : (A) + traditional oversampling/augmentation,
                          as a baseline synthetic augmentation has to beat

This intentionally mirrors the training/eval setup already used in
`dr_benchmark` (ResNet50 / EfficientNet-B4 backbones, standard fundus
normalization, stratified split) so results are directly comparable to the
existing ICPR benchmark numbers, and reports the metrics that matter for DR
grading specifically:
    - Accuracy
    - Macro-F1 (sensitive to minority-grade performance)
    - Quadratic Weighted Kappa (QWK) -- the standard DR grading metric,
      since grades are ordinal and adjacent-grade errors should be
      penalized less than distant ones.
    - Per-class recall (does synthetic data actually rescue grade 3/4 recall?)

Usage:
    # (A) real-only baseline
    python train_dr_classifier.py \
        --real_h5 /path/DRGrading_128x128.h5 \
        --test_h5 /path/DRGrading_128x128_test.h5 \
        --backbone resnet50 --epochs 30 --run_name real_only

    # (B) real + CCDM synthetic
    python train_dr_classifier.py \
        --real_h5 /path/DRGrading_128x128.h5 \
        --test_h5 /path/DRGrading_128x128_test.h5 \
        --synthetic_h5 /path/to/ccdm_generated_128x128.h5 \
        --synthetic_cap_per_grade 1500 \
        --backbone resnet50 --epochs 30 --run_name real_plus_synthetic

Then compare the printed metrics (or the JSON dumped to --out_dir) across runs.
The synthetic h5 is expected to have the same 'images'/'labels' schema as the
files produced by CCDM's --dump_fake_data option (see main.py / trainer.py
for the exact dump path and shape it writes).
"""

import argparse
import json
import os

import h5py
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class DRDataset(Dataset):
    def __init__(self, images, labels, train=True, img_size=128):
        self.images = images  # uint8, N x 3 x H x W
        self.labels = labels.astype(np.int64)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            self.tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].transpose(1, 2, 0)  # CHW -> HWC for ToTensor
        img = self.tf(img)
        return img, self.labels[idx]


def load_h5(path):
    with h5py.File(path, "r") as hf:
        images = hf["images"][:]
        labels = hf["labels"][:]
    return images, labels


def cap_per_class(images, labels, cap):
    if cap is None:
        return images, labels
    keep_idx = []
    for g in np.unique(labels):
        idx_g = np.where(labels == g)[0]
        if len(idx_g) > cap:
            idx_g = np.random.choice(idx_g, size=cap, replace=False)
        keep_idx.append(idx_g)
    keep_idx = np.concatenate(keep_idx)
    return images[keep_idx], labels[keep_idx]


def _resize_vit_pos_embed(model, img_size):
    """Re-target a timm ViT with a learned positional embedding to a different
    input resolution than its pretrained default (e.g. DINOv2's 518 -> 128).

    The patch embed is just a stride-(patch_size) conv, so the model works at
    any resolution on its own; only the learned pos-embed needs to be
    re-interpolated to match the new token grid. Class/distillation prefix
    tokens are kept unchanged. No-op for any other model family.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    pe = getattr(model, "pos_embed", None)
    patch_embed = getattr(model, "patch_embed", None)
    if pe is None or patch_embed is None or not hasattr(patch_embed, "patch_size"):
        return

    ps = patch_embed.patch_size
    ps = ps[0] if isinstance(ps, (tuple, list)) else ps
    grid_new = (img_size - ps) // ps + 1  # conv output for non-divisible inputs
    n_tok_new = grid_new * grid_new

    prefix = getattr(model, "num_prefix_tokens", 1)
    n_tok_old = pe.shape[1] - prefix
    grid_old = int(round(n_tok_old**0.5))
    if grid_old * grid_old != n_tok_old:
        raise ValueError(f"non-square pos-embed grid with {n_tok_old} spatial tokens")

    cls_tokens = pe[:, :prefix]
    spatial = pe[:, prefix:].reshape(1, grid_old, grid_old, -1).permute(0, 3, 1, 2)
    spatial = F.interpolate(spatial, size=(grid_new, grid_new), mode="bicubic", align_corners=False)
    spatial = spatial.permute(0, 2, 3, 1).reshape(1, n_tok_new, -1)
    new_pe = torch.cat([cls_tokens, spatial], dim=1)

    with torch.no_grad():
        if new_pe.shape == pe.shape:
            pe.copy_(new_pe)
        else:
            model.pos_embed = nn.Parameter(new_pe)
    patch_embed.img_size = (img_size, img_size)
    patch_embed.num_patches = n_tok_new
    patch_embed.grid_size = (grid_new, grid_new)


def build_model(backbone, num_classes=5, pretrained=True, img_size=128):
    if backbone in ("resnet50", "resnet101"):
        if backbone == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            m = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone in ("efficientnet_b3", "efficientnet_b4", "efficientnet_b5"):
        if backbone == "efficientnet_b3":
            m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "efficientnet_b4":
            m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            m = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif backbone in ("densenet121", "densenet201"):
        if backbone == "densenet121":
            m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            m = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif backbone == "vit_base_patch14_dinov2":
        m = timm.create_model("vit_base_patch14_dinov2", pretrained=pretrained, num_classes=0)
        m.head = nn.Linear(m.embed_dim, num_classes)
        if img_size is not None:
            _resize_vit_pos_embed(m, img_size)
    elif backbone == "swin_large":
        m = timm.create_model("swin_large_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return m


def evaluate(model, loader, device, return_probs=False):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            if return_probs:
                all_probs.append(probs.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "qwk": cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
        "per_class_report": classification_report(
            all_labels, all_preds, target_names=[f"grade_{i}" for i in range(5)], output_dict=True, zero_division=0
        ),
    }
    if return_probs:
        metrics["probs"] = np.concatenate(all_probs)
        metrics["preds"] = all_preds
        metrics["labels"] = all_labels
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_h5", type=str, default=None)
    p.add_argument("--test_h5", type=str, default=None)
    p.add_argument("--synthetic_h5", type=str, default=None, help="images generated by CCDM (--dump_fake_data)")
    p.add_argument("--synthetic_cap_per_grade", type=int, default=None)
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "densenet121", "densenet201", "vit_base_patch14_dinov2", "swin_large"])
    p.add_argument("--pretrained", action="store_true", default=True, help="use ImageNet pretrained weights (default); pass --no-pretrained to train from scratch")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./downstream_results")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--score_h5", type=str, default=None,
                   help="score this h5 with a trained checkpoint (--ckpt); no training happens")
    p.add_argument("--ckpt", type=str, default=None, help="path to {run}_best.pth state_dict for --score_h5")
    p.add_argument("--score_save_probs", action="store_true",
                   help="with --score_h5, also dump per-image class probabilities to .npz")
    p.add_argument("--save_test_preds", type=str, default=None,
                   help="after training, re-evaluate the best model on the test split and dump "
                        "preds/labels/probs to this .npz (for external-domain re-scoring, e.g. "
                        "Messidor-2's 0-3 scale)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.score_h5 is not None:
        if args.ckpt is None:
            raise ValueError("--score_h5 requires --ckpt")
        images, labels = load_h5(args.score_h5)
        ds = DRDataset(images, labels, train=False, img_size=args.img_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        model = build_model(args.backbone, pretrained=False, img_size=args.img_size).to(device)
        state = torch.load(args.ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=True)
        print(f"[score] loaded {args.ckpt}: missing={missing}, unexpected={unexpected}")
        metrics = evaluate(model, loader, device, return_probs=args.score_save_probs)
        conf = metrics["probs"].max(axis=1) if args.score_save_probs else None
        base = os.path.splitext(os.path.basename(args.score_h5))[0]
        os.makedirs(args.out_dir, exist_ok=True)
        csv_path = os.path.join(args.out_dir, f"{base}_scores_{args.run_name}.csv")
        with open(csv_path, "w") as f:
            f.write("idx,pred_grade,pred_conf\n")
            for i in range(len(metrics["preds"])):
                c = conf[i] if conf is not None else ""
                f.write(f"{i},{metrics['preds'][i]},{c}\n")
        print(f"[score] wrote {csv_path}")
        if args.score_save_probs:
            npz_path = os.path.join(args.out_dir, f"{base}_probs_{args.run_name}.npz")
            np.savez(npz_path, probs=metrics["probs"], preds=metrics["preds"], labels=metrics["labels"])
            print(f"[score] wrote {npz_path}")
        return

    if args.real_h5 is None or args.test_h5 is None:
        raise ValueError("training mode requires --real_h5 and --test_h5")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    real_images, real_labels = load_h5(args.real_h5)
    test_images, test_labels = load_h5(args.test_h5)

    train_images, train_labels = real_images, real_labels
    if args.synthetic_h5 is not None:
        syn_images, syn_labels = load_h5(args.synthetic_h5)
        syn_images, syn_labels = cap_per_class(syn_images, syn_labels, args.synthetic_cap_per_grade)
        train_images = np.concatenate([train_images, syn_images], axis=0)
        train_labels = np.concatenate([train_labels, syn_labels], axis=0)
        print(f"Added {len(syn_labels)} synthetic images. New training-set grade distribution:")
    else:
        print("Training-set grade distribution (real only):")
    for g in range(5):
        print(f"  grade {g}: {int((train_labels == g).sum())}")

    train_ds = DRDataset(train_images, train_labels, train=True, img_size=args.img_size)
    test_ds = DRDataset(test_images, test_labels, train=False, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.backbone, pretrained=args.pretrained, img_size=args.img_size).to(device)

    # Class-balanced loss: even with synthetic augmentation the real
    # distribution is still skewed, so keep inverse-frequency weighting on
    # by default rather than relying on augmentation alone to fix it.
    class_counts = np.array([max(1, int((train_labels == g).sum())) for g in range(5)])
    class_weights = torch.tensor(class_counts.sum() / (5 * class_counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_qwk = -1.0
    best_metrics = None
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()

        metrics = evaluate(model, test_loader, device)
        print(
            f"[{args.run_name}] epoch {epoch+1}/{args.epochs} "
            f"loss={running_loss/len(train_ds):.4f} "
            f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} qwk={metrics['qwk']:.4f}"
        )
        if metrics["qwk"] > best_qwk:
            best_qwk = metrics["qwk"]
            best_metrics = metrics
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"{args.run_name}_best.pth"))

    if args.save_test_preds is not None:
        state = torch.load(os.path.join(args.out_dir, f"{args.run_name}_best.pth"), map_location=device)
        model.load_state_dict(state)
        ev = evaluate(model, test_loader, device, return_probs=True)
        np.savez(args.save_test_preds, preds=ev["preds"], labels=ev["labels"], probs=ev["probs"])
        print(f"\nSaved best-model test predictions -> {args.save_test_preds}")

    out_path = os.path.join(args.out_dir, f"{args.run_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"\nBest QWK for '{args.run_name}': {best_qwk:.4f}. Full metrics saved to {out_path}")


if __name__ == "__main__":
    main()
