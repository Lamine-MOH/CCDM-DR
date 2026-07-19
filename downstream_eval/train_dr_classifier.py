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


def build_model(backbone, num_classes=5, pretrained=True):
    if backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone == "efficientnet_b4":
        m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return m


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
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
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_h5", type=str, required=True)
    p.add_argument("--test_h5", type=str, required=True)
    p.add_argument("--synthetic_h5", type=str, default=None, help="images generated by CCDM (--dump_fake_data)")
    p.add_argument("--synthetic_cap_per_grade", type=int, default=None)
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "efficientnet_b4"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./downstream_results")
    p.add_argument("--seed", type=int, default=111)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(args.backbone).to(device)

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

    out_path = os.path.join(args.out_dir, f"{args.run_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"\nBest QWK for '{args.run_name}': {best_qwk:.4f}. Full metrics saved to {out_path}")


if __name__ == "__main__":
    main()
