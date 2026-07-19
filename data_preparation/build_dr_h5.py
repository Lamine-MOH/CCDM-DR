"""
build_dr_h5.py

Converts a diabetic retinopathy dataset (image folder + CSV of grades) into the
h5 format that CCDM-DR/dataset.py expects for data_name="DRGrading":

    {data_path}/DRGrading_{img_size}x{img_size}.h5
        - 'images': uint8 array, shape (N, 3, img_size, img_size), RGB, CHW
        - 'labels': float array, shape (N,), values in {0,1,2,3,4} (ICDR grades)

Works out of the box with any dataset whose labels follow the International
Clinical Diabetic Retinopathy (ICDR) severity scale (0=No DR, 1=Mild NPDR,
2=Moderate NPDR, 3=Severe NPDR, 4=PDR) -- e.g. APTOS 2019, EyePACS/Diabetic
Retinopathy Detection, IDRiD Disease Grading, Messidor-2, or a local/regional
dataset labelled the same way.

Expected CSV format (column names configurable via --id_col / --label_col):
    id_code,diagnosis
    000c1434d8d7,2
    001639a390f0,4
    ...

Usage:
    python build_dr_h5.py \
        --image_dir /path/to/train_images \
        --csv_path  /path/to/train.csv \
        --out_dir   /path/to/output/DRGrading \
        --img_size  128 \
        --id_col id_code --label_col diagnosis --ext .png

Preprocessing performed per image (standard for fundus photographs):
    1. Load RGB.
    2. Detect and crop to the circular fundus field of view (removes black
       borders), falling back to a center square crop if detection fails.
    3. Resize (with high-quality resampling) to img_size x img_size.
    4. (optional) CLAHE contrast enhancement on the green/luminance channel,
       which is common practice for retinal image pipelines -- disabled by
       default so raw pixel statistics stay closer to what a diffusion model
       will need to invert; enable with --clahe if you want it baked in.

A held-out test split is written to a *second* h5 file
    {out_dir}/DRGrading_{img_size}x{img_size}_test.h5
so the diffusion model only ever trains on the train split, and the same
test split can be reused later for downstream classifier evaluation
(see downstream_eval/train_dr_classifier.py).
"""

import argparse
import os

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def crop_to_fundus(img_bgr, pad_frac=0.02):
    """Crop a fundus photo to its circular field of view.

    Falls back to a centered square crop if the fundus circle can't be
    reliably detected (this happens for some already-cropped datasets).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return center_square_crop(img_bgr)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Reject degenerate detections (e.g. thresholding grabbed a tiny artifact,
    # or almost the whole frame, which usually means the fundus already
    # fills the image and cropping would just re-crop noise).
    area_frac = (w * h) / (img_bgr.shape[0] * img_bgr.shape[1])
    if area_frac < 0.05:
        return center_square_crop(img_bgr)

    side = max(w, h)
    pad = int(side * pad_frac)
    cx, cy = x + w // 2, y + h // 2
    half = side // 2 + pad

    y0, y1 = max(0, cy - half), min(img_bgr.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(img_bgr.shape[1], cx + half)
    crop = img_bgr[y0:y1, x0:x1]

    if crop.size == 0:
        return center_square_crop(img_bgr)
    return crop


def center_square_crop(img_bgr):
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img_bgr[y0 : y0 + side, x0 : x0 + side]


def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def process_image(path, img_size, use_clahe):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_bgr = crop_to_fundus(img_bgr)
    if use_clahe:
        img_bgr = apply_clahe(img_bgr)
    img_bgr = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.transpose(2, 0, 1)  # HWC -> CHW


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--id_col", type=str, default="id_code")
    p.add_argument("--label_col", type=str, default="diagnosis")
    p.add_argument("--ext", type=str, default=".png", help="image extension if not already in id_col")
    p.add_argument("--clahe", action="store_true", default=False)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=111)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    assert args.id_col in df.columns, f"'{args.id_col}' not found in CSV columns: {list(df.columns)}"
    assert args.label_col in df.columns, f"'{args.label_col}' not found in CSV columns: {list(df.columns)}"

    labels = df[args.label_col].astype(int).values
    assert set(np.unique(labels)).issubset({0, 1, 2, 3, 4}), (
        f"Found label values outside the expected ICDR 0-4 range: {sorted(set(labels))}. "
        "Remap your labels to the 0-4 scale before building the h5 file."
    )

    images, kept_labels = [], []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing fundus images"):
        img_id = str(row[args.id_col])
        fname = img_id if os.path.splitext(img_id)[1] else img_id + args.ext
        fpath = os.path.join(args.image_dir, fname)
        if not os.path.exists(fpath):
            skipped += 1
            continue
        arr = process_image(fpath, args.img_size, args.clahe)
        if arr is None:
            skipped += 1
            continue
        images.append(arr)
        kept_labels.append(int(row[args.label_col]))

    images = np.stack(images, axis=0).astype(np.uint8)
    kept_labels = np.array(kept_labels, dtype=np.float64)
    print(f"\nProcessed {len(images)} images ({skipped} skipped/missing).")
    print("Grade distribution:")
    for g in range(5):
        n = int((kept_labels == g).sum())
        print(f"  grade {g}: {n} ({100*n/len(kept_labels):.1f}%)")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(images))
    n_test = int(len(images) * args.test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    train_path = os.path.join(args.out_dir, f"DRGrading_{args.img_size}x{args.img_size}.h5")
    test_path = os.path.join(args.out_dir, f"DRGrading_{args.img_size}x{args.img_size}_test.h5")

    with h5py.File(train_path, "w") as hf:
        hf.create_dataset("images", data=images[train_idx], dtype="uint8")
        hf.create_dataset("labels", data=kept_labels[train_idx], dtype="float64")
    with h5py.File(test_path, "w") as hf:
        hf.create_dataset("images", data=images[test_idx], dtype="uint8")
        hf.create_dataset("labels", data=kept_labels[test_idx], dtype="float64")

    print(f"\nWrote {len(train_idx)} training images -> {train_path}")
    print(f"Wrote {len(test_idx)} held-out test images -> {test_path}")
    print(
        "\nNext: point --data_path in your run_train.sh at "
        f"'{args.out_dir}' and set --data_name DRGrading --min_label 0 --max_label 4."
    )


if __name__ == "__main__":
    main()
