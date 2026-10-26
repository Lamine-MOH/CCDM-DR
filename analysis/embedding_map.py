#!/usr/bin/env python3
"""embedding_map.py — 2D map of real vs classically-augmented vs synthetic DR data.

No classifier and no pretrained network: each image is represented by its own
content, so the map describes the data itself rather than a model's view of it.

Representation (per image):
    1. optional fundus-mask crop  -> removes black border / framing differences
    2. square pad + resize to N x N RGB (default 48)
    3. flatten to N*N*3 pixels in [0, 1]
    4. optional handcrafted colour/texture statistics (--extra_feats)
Then PCA on the union of ALL sources, and a 2D projection of the PCA space
with UMAP (default) and/or t-SNE.  All sources share one embedding, so the
positions are directly comparable.

Compared sources:
    real            real training h5 (per-grade capped)
    real_test       optional real test h5 (natural-variation reference)
    aug_train       exactly the downstream training augmentation:
                    horizontal flip p=.5, rotation +/-8 deg, fill=0
    aug_popular     heavy classical augmentation: h/v flip, +/-15 deg rotation,
                    random resized crop (scale .85-1.0), colour jitter, blur,
                    cutout
    <name>          one or more synthetic pools (--synthetic NAME=PATH)

Outputs (into --out_dir):
    {method}_by_source.png     one panel per source + an all-source overlay
    {method}_by_grade.png      one panel per grade, colour = source
    {method}_pair_a_vs_b.png   1-vs-1 map: global panel + one panel per grade
    {method}_thumbnails.png    actual image thumbnails at their 2D coordinates
    embedding_projection.csv   x, y, source, split, grade, h5 index, method
    embedding_metrics.csv      per source x grade, in PCA space:
                                 nn_real_ratio   distance to nearest same-grade
                                                 real / median real-real NN dist
                                 knn_real_frac   fraction of 10 same-grade
                                                 neighbours that are real
                                 energy_distance distribution distance to
                                                 same-grade real
                                 silhouette_grade  (grade='all' row only)
    pairwise_metrics.csv       per pair: B nearest-A ratio, fraction of B's
                               neighbours in A, and energy distance, overall
                               and per grade (--pairs real|all)
    embedding_features.npz     feature/meta cache for --reuse_features
    run_config.json            parameters + per-source counts

Usage:
    python analysis/embedding_map.py \
        --real_h5   data/DRGrading/Aptos/DRGrading_128x128_train.h5 \
        --real_test_h5 data/DRGrading/Aptos/DRGrading_128x128_test.h5 \
        --synthetic blendA_g3cs4=output/blends/blendA_g3cs4.h5 \
        --synthetic cs4.0=output/generated_cs4.0/generated.h5 \
        --aug train,popular --max_per_grade 500 --method both \
        --out_dir experiments_outputs/embedding_map

Notes / caveats:
    * --pairs real (default) writes one dedicated 1-vs-1 map per source against
      the real training set; --pairs all compares every unordered pair and
      --pairs none disables them.
    * Pixel space is dominated by global colour, illumination and framing.
      That is a property of the data, not a bug -- but read the per-grade
      panels and the PCA-space metrics, not just the global blobs.
    * UMAP/t-SNE distort distances; never read absolute inter-cluster
      distances off the 2D plots.  The CSV metrics are computed in the full
      PCA space.
    * blendA_g3cs4 keeps grade 4 real-only by construction, so that pool has
      no synthetic grade 4.  The uniform cs4.0 pool covers all five grades.
"""

import argparse
import csv
import inspect
import json
import os
import random
import time

import h5py
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

GRADES = (0, 1, 2, 3, 4)
GRADE_COLORS = ["#2f6db3", "#3f9e4d", "#d9a21b", "#d05252", "#8a5ab5"]
GRADE_NAMES = {g: f"grade {g}" for g in GRADES}

DISPLAY = {
    "real": "Real (train)",
    "real_test": "Real (test)",
    "aug_train": "Classical aug: training pipeline",
    "aug_popular": "Classical aug: popular set",
}

# Default synthetic-pool candidates, discovered only when --synthetic is omitted.
DEFAULT_SYNTHETIC = [
    ("blendA_g3cs4", [
        "output/blends/blendA_g3cs4.h5",
        "output/generated_blendA_g3cs4/generated.h5",
    ]),
    ("cs4.0", [
        "output/generated_cs4.0/generated.h5",
        "output/generated_cs4/generated.h5",
        "output/generated_cfg4/generated.h5",
    ]),
]

AUG_FNS = {}


# ---------------------------------------------------------------------------
# h5 / selection helpers
# ---------------------------------------------------------------------------
def read_labels(path):
    with h5py.File(path, "r") as hf:
        if "labels" not in hf:
            raise ValueError(f"{path}: missing 'labels' dataset")
        labels = hf["labels"][:]
    labels = np.rint(np.asarray(labels, dtype=np.float64)).astype(np.int64)
    bad = np.unique(labels[(labels < 0) | (labels > 4)])
    if len(bad):
        raise ValueError(f"{path}: labels outside [0,4]: {bad}")
    if labels.max() <= 1:
        print(f"  [warn] {path}: labels max={labels.max()} — are these "
              f"normalized to [0,1] instead of raw ICDR grades 0-4?")
    return labels


def read_images(path, idx):
    """Read selected images (sorted h5 indices). Returns uint8 NCHW."""
    idx = np.asarray(sorted(int(i) for i in idx), dtype=np.int64)
    with h5py.File(path, "r") as hf:
        if "images" not in hf:
            raise ValueError(f"{path}: missing 'images' dataset")
        ds = hf["images"]
        if ds.ndim != 4 or ds.shape[1] != 3:
            raise ValueError(f"{path}: expected (N,3,H,W), got {ds.shape}")
        if ds.dtype != np.uint8:
            raise ValueError(f"{path}: images dtype {ds.dtype} != uint8")
        images = ds[idx]
    return images


def select_indices(labels, max_per_grade, rng):
    """Stratified per-grade selection, returned sorted (h5-friendly)."""
    chosen = []
    for g in GRADES:
        gi = np.flatnonzero(labels == g)
        if max_per_grade and len(gi) > max_per_grade:
            gi = rng.choice(gi, size=max_per_grade, replace=False)
        chosen.append(np.sort(gi))
    chosen = [c for c in chosen if len(c)]
    if not chosen:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(chosen))


def grade_counts(labels):
    return {g: int(np.sum(labels == g)) for g in GRADES}


# ---------------------------------------------------------------------------
# image -> feature vector (pure numpy / PIL, no network)
# ---------------------------------------------------------------------------
def _square(arr):
    h, w = arr.shape[:2]
    if h == w:
        return arr
    s = max(h, w)
    out = np.zeros((s, s, arr.shape[2]), dtype=arr.dtype)
    y = (s - h) // 2
    x = (s - w) // 2
    out[y:y + h, x:x + w] = arr
    return out


def fundus_crop(img):
    """Crop to the fundus bounding box, then pad to a square.

    A cheap luminance threshold is used (fundus datasets and CCDM output both
    place the retina on a near-black background).  If the mask is implausible
    (empty or the full frame) the image is returned unchanged.
    """
    gray = img.astype(np.float32).mean(axis=2)
    mask = gray > 8.0
    if mask.sum() < 100 or mask.sum() > 0.995 * mask.size:
        return img
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    if h < 8 or w < 8:
        return img
    side = int(round(max(h, w) * 1.02))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, cy - side // 2)
    y1 = min(img.shape[0], y0 + side)
    y0 = max(0, y1 - side)
    x0 = max(0, cx - side // 2)
    x1 = min(img.shape[1], x0 + side)
    x0 = max(0, x1 - side)
    return _square(np.ascontiguousarray(img[y0:y1, x0:x1]))


def content_stats(img):
    """Model-free colour/texture descriptors (16 dims)."""
    x = img.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mask = gray > 8.0
    if mask.sum() < 100:
        mask = np.ones_like(gray, dtype=bool)

    feats = []
    for c in range(3):
        ch = x[..., c][mask]
        feats += [float(ch.mean()) / 255.0, float(ch.std()) / 255.0]

    hsv = np.asarray(
        Image.fromarray(np.ascontiguousarray(img)).convert("HSV"), dtype=np.float32
    )
    for c in range(3):
        ch = hsv[..., c][mask]
        feats += [float(ch.mean()) / 255.0, float(ch.std()) / 255.0]

    # high-frequency energy ratio (smoothness / "airbrushed" look)
    n = g.shape[0]
    Gc = g - g.mean()
    E = np.abs(np.fft.rfft2(Gc)) ** 2
    fx = np.fft.fftfreq(n)
    fy = np.fft.rfftfreq(n)
    fxg, fyg = np.meshgrid(fx, fy, indexing="ij")
    rad = np.sqrt(fxg ** 2 + fyg ** 2)
    hi = float(E[rad > 0.25].sum() / (E.sum() + 1e-12))

    # edge energy
    gy, gx = np.gradient(g)
    edge = float(np.sqrt(gx * gx + gy * gy).mean()) / 255.0

    # local noise: mean std of 8x8 blocks
    h, w = g.shape
    bh, bw = h // 8, w // 8
    if bh > 0 and bw > 0:
        blocks = (g[:bh * 8, :bw * 8]
                  .reshape(bh, 8, bw, 8)
                  .transpose(0, 2, 1, 3)
                  .reshape(bh, bw, 64))
        noise = float(blocks.std(axis=2).mean()) / 255.0
    else:
        noise = 0.0

    redness = float((r - g).mean()) / 255.0
    feats += [hi, edge, noise, redness]
    return np.asarray(feats, dtype=np.float32)


def image_features(img_hwc, pixel_size, crop, extra_feats):
    """Return (feature vector, 48x48 uint8 thumbnail)."""
    img_hwc = np.ascontiguousarray(img_hwc)
    proc = fundus_crop(img_hwc) if crop else img_hwc
    proc = np.ascontiguousarray(_square(proc))
    pil = Image.fromarray(proc).resize((pixel_size, pixel_size), Image.BICUBIC)
    pix = np.asarray(pil, dtype=np.float32).reshape(-1) / 255.0

    parts = [pix]
    if extra_feats:
        parts.append(content_stats(proc))
    vec = np.concatenate(parts).astype(np.float32)
    thumb = np.asarray(
        Image.fromarray(proc).resize((48, 48), Image.BILINEAR), dtype=np.uint8
    )
    return vec, thumb


# ---------------------------------------------------------------------------
# classical augmentation (PIL + numpy, no torch)
# ---------------------------------------------------------------------------
def _to_pil(img_hwc):
    return Image.fromarray(np.ascontiguousarray(img_hwc))


def _to_np(im):
    return np.asarray(im).copy()


def _random_resized_crop(im, rng, scale=(0.85, 1.0), ratio=(0.9, 1.1)):
    w, h = im.size
    area = float(w * h)
    for _ in range(10):
        target = area * rng.uniform(*scale)
        ar = rng.uniform(*ratio)
        cw = int(round(np.sqrt(target * ar)))
        ch = int(round(np.sqrt(target / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            x = rng.randint(0, w - cw)
            y = rng.randint(0, h - ch)
            return im.crop((x, y, x + cw, y + ch)).resize((w, h), Image.BICUBIC)
    return im


def _hue_shift(im, delta):
    hsv = np.asarray(im.convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(round(delta * 255.0))) % 256
    chans = [Image.fromarray(np.ascontiguousarray(hsv[..., i]).astype(np.uint8))
             for i in range(3)]
    return Image.merge("HSV", chans).convert("RGB")


def aug_train(img_hwc, rng):
    """Exactly the downstream training pipeline."""
    im = _to_pil(img_hwc)
    if rng.random() < 0.5:
        im = ImageOps.mirror(im)
    im = im.rotate(rng.uniform(-8.0, 8.0), resample=Image.BICUBIC,
                   expand=False, fillcolor=0)
    return _to_np(im)


def aug_popular(img_hwc, rng):
    """A heavier, still label-preserving classical augmentation set."""
    im = _to_pil(img_hwc)
    if rng.random() < 0.5:
        im = ImageOps.mirror(im)
    if rng.random() < 0.5:
        im = ImageOps.flip(im)
    im = im.rotate(rng.uniform(-15.0, 15.0), resample=Image.BICUBIC,
                   expand=False, fillcolor=0)
    im = _random_resized_crop(im, rng)

    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.8, 1.2))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.8, 1.2))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:
        im = _hue_shift(im, rng.uniform(-0.02, 0.02))
    if rng.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 1.2)))

    arr = _to_np(im)
    if rng.random() < 0.25 and arr.shape[0] >= 16 and arr.shape[1] >= 16:
        side = max(8, arr.shape[0] // 8)
        y = rng.randint(0, arr.shape[0] - side)
        x = rng.randint(0, arr.shape[1] - side)
        arr[y:y + side, x:x + side] = 0
    return arr


AUG_FNS["train"] = aug_train
AUG_FNS["popular"] = aug_popular


# ---------------------------------------------------------------------------
# feature extraction over a source
# ---------------------------------------------------------------------------
def build_block(name, split, path, images, indices, labels, transform, reps,
                args, seed):
    """Extract features for one source (optionally with augmentation replicas)."""
    n = len(images)
    feats, thumbs, out_idx, out_grade = [], [], [], []
    t0 = time.time()
    for i in range(n):
        img = np.ascontiguousarray(np.transpose(images[i], (1, 2, 0)))
        if transform is None:
            variants = [img]
        else:
            variants = [
                transform(img, random.Random(seed + 1000003 * int(indices[i]) + r))
                for r in range(reps)
            ]
        for v in variants:
            vec, thumb = image_features(
                v, args.pixel_size, not args.no_fundus_crop, args.extra_feats
            )
            feats.append(vec)
            thumbs.append(thumb)
            out_idx.append(int(indices[i]))
            out_grade.append(int(labels[i]))
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"  [{name}] {i + 1}/{n} images  ({time.time() - t0:.1f}s)")

    return {
        "source": name,
        "split": split,
        "path": path,
        "features": np.stack(feats).astype(np.float32),
        "thumbs": np.stack(thumbs).astype(np.uint8),
        "index": np.asarray(out_idx, dtype=np.int64),
        "grade": np.asarray(out_grade, dtype=np.int8),
    }


# ---------------------------------------------------------------------------
# projections
# ---------------------------------------------------------------------------
def run_umap(z, args):
    try:
        import umap  # noqa: WPS433 (optional dependency)
    except ImportError as exc:
        raise SystemExit(
            "UMAP requested but umap-learn is not installed.\n"
            "  pip install umap-learn\n"
            "or use --method tsne / --method pca."
        ) from exc

    n_neighbors = max(2, min(args.umap_neighbors, len(z) - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
        transform_seed=args.seed,
        n_jobs=1,
        verbose=False,
    )
    print(f"  UMAP: n={len(z)}, n_neighbors={n_neighbors}, "
          f"min_dist={args.umap_min_dist}")
    return reducer.fit_transform(z)


def run_tsne(z, args):
    n = len(z)
    perplexity = min(args.tsne_perplexity, max(5.0, (n - 1) / 3.0))
    params = inspect.signature(TSNE).parameters
    iter_kw = {"max_iter": 1000} if "max_iter" in params else {"n_iter": 1000}
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
        **iter_kw,
    )
    print(f"  t-SNE: n={n}, perplexity={perplexity:.1f}")
    return tsne.fit_transform(z)


# ---------------------------------------------------------------------------
# metrics (PCA space, not the 2D projection)
# ---------------------------------------------------------------------------
def _nn_distances(query, reference):
    tree = cKDTree(reference)
    d, _ = tree.query(query, k=1)
    return d


def _real_self_nn(real):
    if len(real) < 2:
        return None
    d, _ = cKDTree(real).query(real, k=2)
    return float(np.median(d[:, 1]))


def _knn_real_fraction(src, real, k=10):
    if len(src) == 0 or len(real) == 0:
        return float("nan")
    pool = np.vstack([src, real])
    is_real = np.concatenate([np.zeros(len(src), bool), np.ones(len(real), bool)])
    kk = min(k, len(pool) - 1)
    if kk < 1:
        return float("nan")
    _, idx = cKDTree(pool).query(src, k=kk + 1)
    fracs = []
    for i, row in enumerate(np.atleast_2d(idx)):
        nbr = [j for j in row if j != i][:kk]
        if nbr:
            fracs.append(float(is_real[nbr].mean()))
    return float(np.mean(fracs)) if fracs else float("nan")


def _energy_distance(a, b, seed, max_n=600):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    d_ab = cdist(a, b).mean()
    d_aa = pdist(a).mean() if len(a) > 1 else 0.0
    d_bb = pdist(b).mean() if len(b) > 1 else 0.0
    return float(np.sqrt(max(0.0, 2.0 * d_ab - d_aa - d_bb)))


def compute_metrics(z, meta, out_csv, seed):
    sources = list(dict.fromkeys(meta["source"].tolist()))
    grades = meta["grade"]
    rows = []

    for src in sources:
        m_src = meta["source"] == src
        # source-level silhouette over grades
        labels_src = grades[m_src]
        sil = float("nan")
        if len(labels_src) > 2 and len(np.unique(labels_src)) >= 2:
            try:
                sil = float(silhouette_score(
                    z[m_src], labels_src,
                    sample_size=min(3000, len(labels_src)),
                    random_state=seed,
                ))
            except Exception:
                sil = float("nan")
        rows.append({
            "source": src, "grade": "all", "n": int(m_src.sum()),
            "nn_real_ratio": "", "knn_real_frac": "",
            "energy_distance": "", "silhouette_grade": f"{sil:.4f}"
            if np.isfinite(sil) else "",
        })

        if src == "real":
            for g in GRADES:
                n = int(np.sum(m_src & (grades == g)))
                if n:
                    defined = n > 1
                    rows.append({
                        "source": src, "grade": g, "n": n,
                        "nn_real_ratio": "1.0000" if defined else "",
                        "knn_real_frac": "1.0000" if defined else "",
                        "energy_distance": "0.0000" if defined else "",
                        "silhouette_grade": "",
                    })
            continue

        for g in GRADES:
            m_s = m_src & (grades == g)
            m_r = (meta["source"] == "real") & (grades == g)
            n = int(m_s.sum())
            if n == 0:
                continue
            zs, zr = z[m_s], z[m_r]
            if len(zr) == 0:
                rows.append({
                    "source": src, "grade": g, "n": n,
                    "nn_real_ratio": "", "knn_real_frac": "",
                    "energy_distance": "", "silhouette_grade": "",
                })
                continue
            base = _real_self_nn(zr)
            nn = _nn_distances(zs, zr)
            ratio = float(np.median(nn) / base) if base else float("nan")
            rows.append({
                "source": src, "grade": g, "n": n,
                "nn_real_ratio": f"{ratio:.4f}" if np.isfinite(ratio) else "",
                "knn_real_frac": f"{_knn_real_fraction(zs, zr):.4f}",
                "energy_distance": f"{_energy_distance(zs, zr, seed):.4f}",
                "silhouette_grade": "",
            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "grade", "n", "nn_real_ratio",
                           "knn_real_frac", "energy_distance", "silhouette_grade"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nEmbedding metrics (PCA space)")
    print(f"{'source':>18} {'grade':>5} {'n':>6} {'nn/r':>7} {'knnR':>7} "
          f"{'energy':>8} {'sil':>7}")
    for r in rows:
        print(f"{r['source']:>18} {str(r['grade']):>5} {r['n']:>6} "
              f"{str(r['nn_real_ratio']):>7} {str(r['knn_real_frac']):>7} "
              f"{str(r['energy_distance']):>8} {str(r['silhouette_grade']):>7}")
    print(f"\nwrote {out_csv}")
    return rows


def _self_nn_within_grade(z, grades):
    """Median nearest-neighbour distance inside each grade (pooled)."""
    parts = []
    for g in np.unique(grades):
        m = grades == g
        if int(m.sum()) > 1:
            d, _ = cKDTree(z[m]).query(z[m], k=2)
            parts.append(d[:, 1])
    if not parts:
        return None
    return float(np.median(np.concatenate(parts)))


def compute_pairwise_metrics(z, meta, pairs, out_csv, seed):
    """Per-pair distribution numbers: B relative to A, overall and per grade."""
    sources = meta["source"]
    grades = meta["grade"]
    rows = []

    for a, b in pairs:
        m_a = sources == a
        m_b = sources == b
        for g in ("all",) + GRADES:
            if g == "all":
                za, zb, ga = z[m_a], z[m_b], grades[m_a]
            else:
                sel_a = m_a & (grades == g)
                sel_b = m_b & (grades == g)
                za, zb, ga = z[sel_a], z[sel_b], grades[sel_a]
            if len(za) == 0 or len(zb) == 0:
                continue
            base = _self_nn_within_grade(za, ga)
            nn = _nn_distances(zb, za)
            ratio = float(np.median(nn) / base) if base else float("nan")
            rows.append({
                "a": a,
                "b": b,
                "grade": g,
                "n_a": int(len(za)),
                "n_b": int(len(zb)),
                "nn_a_ratio": f"{ratio:.4f}" if np.isfinite(ratio) else "",
                "knn_a_frac": f"{_knn_real_fraction(zb, za):.4f}",
                "energy_distance": f"{_energy_distance(za, zb, seed):.4f}",
            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["a", "b", "grade", "n_a", "n_b",
                           "nn_a_ratio", "knn_a_frac", "energy_distance"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nPairwise metrics (PCA space; B measured against A)")
    print(f"{'A':>16} {'B':>16} {'grade':>5} {'nA':>5} {'nB':>5} "
          f"{'nn/r':>7} {'knnA':>7} {'energy':>8}")
    for r in rows:
        print(f"{r['a']:>16} {r['b']:>16} {str(r['grade']):>5} "
              f"{r['n_a']:>5} {r['n_b']:>5} {r['nn_a_ratio']:>7} "
              f"{r['knn_a_frac']:>7} {r['energy_distance']:>8}")
    print(f"\nwrote {out_csv}")
    return rows


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _display(name):
    return DISPLAY.get(name, name)


def _source_palette(sources):
    """Grey is reserved for the real references; other sources get vivid colours."""
    grays = {"real": "#b5b5b5", "real_test": "#6f6f6f"}
    vivid = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    out, k = {}, 0
    for s in sources:
        if s in grays:
            out[s] = grays[s]
        else:
            out[s] = vivid[k % len(vivid)]
            k += 1
    return out


def _robust_limits(xy):
    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1e-9)
    return lo - pad, hi + pad


def _safe_name(name):
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def make_pairs(sources, mode):
    """Pair list for the 1-vs-1 figures.

    real : every non-real source against the real training set (recommended)
    all  : every unordered pair of sources
    none : no pairwise figures
    """
    if mode == "none" or len(sources) < 2:
        return []
    if mode == "real":
        if "real" not in sources:
            return []
        return [("real", s) for s in sources if s != "real"]
    return [(sources[i], sources[j])
            for i in range(len(sources))
            for j in range(i + 1, len(sources))]



def _grade_legend(fig):
    handles = [Line2D([0], [0], marker="o", ls="", color=GRADE_COLORS[g],
                      markersize=7, label=GRADE_NAMES[g]) for g in GRADES]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               frameon=False, fontsize=10)


def plot_by_source(xy, meta, out_path, method):
    sources = list(dict.fromkeys(meta["source"].tolist()))
    real_m = meta["source"] == "real"
    lo, hi = _robust_limits(xy)

    ncols = 3
    nrows = int(np.ceil((len(sources) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.2 * ncols, 4.6 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    for ax, src in zip(axes, sources):
        m = meta["source"] == src
        if src != "real" and real_m.any():
            ax.scatter(xy[real_m, 0], xy[real_m, 1], c="#cfcfcf", s=2,
                       alpha=0.35, linewidths=0, rasterized=True)
        for g in GRADES:
            mm = m & (meta["grade"] == g)
            if mm.any():
                ax.scatter(xy[mm, 0], xy[mm, 1], c=GRADE_COLORS[g], s=7,
                           alpha=0.70, linewidths=0, rasterized=True)
        ax.set_title(f"{_display(src)}  (n={int(m.sum())})", fontsize=10)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([])
        ax.set_yticks([])

    overlay_ax = axes[len(sources)]
    palette = _source_palette(sources)
    for src in sources:
        m = meta["source"] == src
        overlay_ax.scatter(xy[m, 0], xy[m, 1], c=palette[src], s=3,
                           alpha=0.45, linewidths=0, rasterized=True)
    overlay_ax.set_title("All sources", fontsize=10)
    overlay_ax.set_xlim(lo[0], hi[0])
    overlay_ax.set_ylim(lo[1], hi[1])
    overlay_ax.set_xticks([])
    overlay_ax.set_yticks([])
    handles = [Line2D([0], [0], marker="o", ls="", color=palette[s],
                      markersize=6, label=_display(s)) for s in sources]
    overlay_ax.legend(handles=handles, loc="best", fontsize=7, frameon=True)

    for ax in axes[len(sources) + 1:]:
        ax.axis("off")

    for ax in axes[:len(sources) + 1]:
        ax.set_xlabel(f"{method.upper()}-1", fontsize=8)
        ax.set_ylabel(f"{method.upper()}-2", fontsize=8)

    _grade_legend(fig)
    fig.suptitle(
        f"{method.upper()} of image-space features — colour = DR grade "
        f"(n={len(xy)}; grey = real background in non-real panels)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_by_grade(xy, meta, out_path, method):
    sources = list(dict.fromkeys(meta["source"].tolist()))
    palette = _source_palette(sources)
    lo, hi = _robust_limits(xy)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.6), squeeze=False)
    axes = axes.ravel()
    for g, ax in zip(GRADES, axes):
        for src in sources:
            m = (meta["source"] == src) & (meta["grade"] == g)
            if not m.any():
                continue
            ax.scatter(xy[m, 0], xy[m, 1], c=palette[src], s=6, alpha=0.55,
                       linewidths=0, rasterized=True)
        n_real = int(np.sum((meta["source"] == "real") & (meta["grade"] == g)))
        n_other = int(np.sum((meta["source"] != "real") & (meta["grade"] == g)))
        ax.set_title(f"{GRADE_NAMES[g]}  (real n={n_real}, other n={n_other})",
                     fontsize=10)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{method.upper()}-1", fontsize=8)
        ax.set_ylabel(f"{method.upper()}-2", fontsize=8)

    handles = [Line2D([0], [0], marker="o", ls="", color=palette[s],
                      markersize=7, label=_display(s)) for s in sources]
    axes[-1].axis("off")
    axes[-1].legend(handles=handles, loc="center", fontsize=9, frameon=False)

    fig.suptitle(f"{method.upper()} — per-grade view of real vs augmented vs "
                 f"synthetic", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_pair(xy, meta, out_path, method, a, b, palette):
    """One dedicated 1-vs-1 map: global panel + one panel per grade."""
    lo, hi = _robust_limits(xy)
    m_a = meta["source"] == a
    m_b = meta["source"] == b

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.6), squeeze=False)
    axes = axes.ravel()

    ax = axes[0]
    if m_a.any():
        ax.scatter(xy[m_a, 0], xy[m_a, 1], c=palette[a], s=5, alpha=0.55,
                   linewidths=0, rasterized=True, label=_display(a))
    if m_b.any():
        ax.scatter(xy[m_b, 0], xy[m_b, 1], c=palette[b], s=5, alpha=0.65,
                   linewidths=0, rasterized=True, label=_display(b))
    ax.set_title(
        f"All grades — {_display(a)} vs {_display(b)}  "
        f"(n={int(m_a.sum())} / {int(m_b.sum())})", fontsize=10,
    )
    ax.legend(loc="best", fontsize=8, frameon=True)

    for g, ax in zip(GRADES, axes[1:]):
        ma_g = m_a & (meta["grade"] == g)
        mb_g = m_b & (meta["grade"] == g)
        if ma_g.any():
            ax.scatter(xy[ma_g, 0], xy[ma_g, 1], c=palette[a], s=6,
                       alpha=0.55, linewidths=0, rasterized=True)
        if mb_g.any():
            ax.scatter(xy[mb_g, 0], xy[mb_g, 1], c=palette[b], s=6,
                       alpha=0.70, linewidths=0, rasterized=True)
        ax.set_title(
            f"{GRADE_NAMES[g]}  (n={int(ma_g.sum())} / {int(mb_g.sum())})",
            fontsize=10,
        )

    for ax in axes:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{method.upper()}-1", fontsize=8)
        ax.set_ylabel(f"{method.upper()}-2", fontsize=8)

    fig.suptitle(
        f"{method.upper()} — 1-vs-1 comparison: {_display(a)} vs {_display(b)}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_thumbnails(xy, meta, thumbs, out_path, method, n_max, seed):
    sources = list(dict.fromkeys(meta["source"].tolist()))
    palette = _source_palette(sources)
    rng = np.random.default_rng(seed)
    per = max(1, n_max // max(1, len(sources)))
    chosen = []
    for src in sources:
        ii = np.flatnonzero(meta["source"] == src)
        if len(ii):
            chosen.extend(rng.choice(ii, size=min(per, len(ii)),
                                     replace=False).tolist())
    chosen = np.asarray(chosen, dtype=np.int64)

    lo, hi = _robust_limits(xy)
    span = float(min(hi - lo))
    radius = 0.022 * span

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(xy[:, 0], xy[:, 1], c="#dddddd", s=1, alpha=0.25,
               linewidths=0, rasterized=True)
    ax.set_autoscale_on(False)
    for i in chosen:
        x, y = xy[i]
        ax.imshow(
            thumbs[i],
            extent=(x - radius, x + radius, y - radius, y + radius),
            interpolation="bilinear",
            zorder=3,
        )
        ax.add_patch(Rectangle(
            (x - radius, y - radius), 2 * radius, 2 * radius,
            fill=False, ec=palette[meta["source"][i]], lw=0.7, zorder=4,
        ))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [Line2D([0], [0], marker="s", ls="", color=palette[s],
                      markersize=7, label=_display(s)) for s in sources]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    ax.set_title(f"Actual images at their {method.upper()} coordinates "
                 f"(n={len(chosen)} sampled)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def save_cache(path, blocks, config):
    sources, grades, indices, splits, feats, thumbs = [], [], [], [], [], []
    for b in blocks:
        sources.append(np.full(len(b["grade"]), b["source"], dtype="U40"))
        splits.append(np.full(len(b["grade"]), b["split"], dtype="U10"))
        grades.append(b["grade"])
        indices.append(b["index"])
        feats.append(b["features"])
        thumbs.append(b["thumbs"])
    np.savez(
        path,
        features=np.concatenate(feats),
        source=np.concatenate(sources),
        split=np.concatenate(splits),
        grade=np.concatenate(grades),
        index=np.concatenate(indices),
        thumbs=np.concatenate(thumbs),
        config=np.array(json.dumps(config)),
    )
    print(f"  wrote {path}")


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        block = {
            "features": data["features"],
            "thumbs": data["thumbs"],
        }
        meta = {
            "source": data["source"].copy(),
            "split": data["split"].copy(),
            "grade": data["grade"].astype(np.int64),
            "index": data["index"].astype(np.int64),
        }
        cfg = json.loads(str(data["config"]))
    return block, meta, cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="2D image-space map of real vs classical augmentation vs "
                    "CCDM synthetic DR data (no classifier / no pretrained net).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real_h5", required=True,
                   help="real training h5 (reference source)")
    p.add_argument("--real_test_h5", default=None,
                   help="optional real test h5 for a natural-variation reference")
    p.add_argument("--synthetic", action="append", default=[], metavar="NAME=PATH",
                   help="synthetic pool to plot; repeatable. If omitted, known "
                        "output/ paths are auto-discovered.")
    p.add_argument("--aug", default="train,popular",
                   help="comma-separated augmentation arms: train, popular, none")
    p.add_argument("--aug_reps", type=int, default=1,
                   help="augmented replicas per real image, per arm")
    p.add_argument("--max_per_grade", type=int, default=500,
                   help="max images per (source, grade); 0 = all")
    p.add_argument("--pixel_size", type=int, default=48,
                   help="thumbnail resolution used for the pixel features")
    p.add_argument("--extra_feats", action="store_true",
                   help="append model-free colour/texture statistics")
    p.add_argument("--extra_feats_weight", type=float, default=0.5,
                   help="total variance of the statistics block relative to the "
                        "pixel block (only used with --extra_feats)")
    p.add_argument("--no_fundus_crop", action="store_true",
                   help="do not crop to the fundus bounding box")
    p.add_argument("--pca_components", type=int, default=64)
    p.add_argument("--method", default="umap",
                   choices=["umap", "tsne", "both", "pca"])
    p.add_argument("--umap_neighbors", type=int, default=30)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--out_dir", default="experiments_outputs/embedding_map")
    p.add_argument("--reuse_features", action="store_true",
                   help="load embedding_features.npz from --out_dir")
    p.add_argument("--no_metrics", action="store_true")
    p.add_argument("--no_thumbnails", action="store_true")
    p.add_argument("--thumbnails_max", type=int, default=240)
    p.add_argument("--pairs", default="real", choices=["real", "all", "none"],
                   help="1-vs-1 maps: 'real' = every source against real_train, "
                        "'all' = every unordered pair, 'none' = disable")
    return p.parse_args(argv)


def resolve_synthetics(specs):
    out = []
    seen = set()
    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
            name = name.strip()
        else:
            path = spec
            name = os.path.basename(os.path.dirname(os.path.abspath(path))) or "synthetic"
        path = path.strip()
        name = name.replace(" ", "_")
        if not name:
            raise SystemExit(f"bad --synthetic spec: {spec!r}")
        if not os.path.isfile(path):
            raise SystemExit(f"synthetic pool not found: {path}")
        if name in seen:
            raise SystemExit(f"duplicate synthetic name: {name}")
        seen.add(name)
        out.append((name, path))
    return out


def auto_discover_synthetics():
    found = []
    for name, candidates in DEFAULT_SYNTHETIC:
        for path in candidates:
            if os.path.isfile(path):
                found.append((name, path))
                break
    return found


def source_path(src, cfg):
    if src == "real":
        return cfg.get("real_h5")
    if src == "real_test":
        return cfg.get("real_test_h5")
    if src.startswith("aug_"):
        return cfg.get("real_h5")
    return dict(cfg.get("synthetic") or []).get(src)


def parse_augs(value):
    if value is None:
        return []
    items = [v.strip().lower() for v in value.split(",") if v.strip()]
    if "none" in items:
        return []
    bad = [v for v in items if v not in AUG_FNS]
    if bad:
        raise SystemExit(f"unknown --aug entries {bad}; choose from {sorted(AUG_FNS)} or none")
    # deterministic order
    return [a for a in ("train", "popular") if a in items]


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    augs = parse_augs(args.aug)
    synth = resolve_synthetics(args.synthetic)
    if not synth:
        synth = auto_discover_synthetics()
        if synth:
            print("[synthetic] --synthetic not given; auto-discovered:")
            for name, path in synth:
                print(f"  {name}: {path}")
        else:
            print("[synthetic] WARNING: no synthetic pool found; the map will "
                  "contain only real + augmented data.")

    cfg = {
        "real_h5": os.path.abspath(args.real_h5),
        "real_test_h5": os.path.abspath(args.real_test_h5) if args.real_test_h5 else None,
        "synthetic": [[n, os.path.abspath(p)] for n, p in synth],
        "aug": augs,
        "aug_reps": args.aug_reps,
        "max_per_grade": args.max_per_grade,
        "pixel_size": args.pixel_size,
        "extra_feats": args.extra_feats,
        "extra_feats_weight": args.extra_feats_weight,
        "fundus_crop": not args.no_fundus_crop,
        "pca_components": args.pca_components,
        "method": args.method,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "tsne_perplexity": args.tsne_perplexity,
        "seed": args.seed,
        "pairs": args.pairs,
    }

    cache_path = os.path.join(args.out_dir, "embedding_features.npz")
    blocks = []
    meta = None

    if args.reuse_features:
        if not os.path.isfile(cache_path):
            raise SystemExit(f"--reuse_features: cache not found: {cache_path}")
        print(f"[cache] loading {cache_path}")
        cached, meta, old_cfg = load_cache(cache_path)
        for key in ("pixel_size", "extra_feats", "extra_feats_weight",
                    "fundus_crop"):
            if old_cfg.get(key) != cfg.get(key):
                raise SystemExit(
                    f"--reuse_features: cache is incompatible ({key}: "
                    f"cached={old_cfg.get(key)!r}, now={cfg.get(key)!r}). "
                    "Re-run without --reuse_features."
                )
        for key in ("aug", "aug_reps", "max_per_grade", "seed"):
            if old_cfg.get(key) != cfg.get(key):
                print(f"[cache] WARNING: config differs for {key!r}: "
                      f"cached={old_cfg.get(key)!r} now={cfg.get(key)!r}")
        # features / meta / thumbs come from the cache below; no h5 is read
        blocks = []
    else:
        rng = np.random.default_rng(args.seed)

        # ---- real train (+ classical augmentation arms) -------------------
        print(f"[real] reading {args.real_h5}")
        real_all = read_labels(args.real_h5)
        sel = select_indices(real_all, args.max_per_grade, rng)
        if len(sel) == 0:
            raise SystemExit("no real images selected")
        real_labels = real_all[sel]
        print(f"[real] selected {len(sel)} / {len(real_all)}  "
              f"counts={grade_counts(real_labels)}")
        real_images = read_images(args.real_h5, sel)
        blocks.append(build_block(
            "real", "train", os.path.abspath(args.real_h5),
            real_images, sel, real_labels, None, 1, args, args.seed,
        ))
        for aug_name in augs:
            blocks.append(build_block(
                f"aug_{aug_name}", "train", os.path.abspath(args.real_h5),
                real_images, sel, real_labels, AUG_FNS[aug_name],
                max(1, args.aug_reps), args, args.seed,
            ))
        del real_images

        # ---- optional real test ------------------------------------------
        if args.real_test_h5:
            print(f"[real_test] reading {args.real_test_h5}")
            test_all = read_labels(args.real_test_h5)
            tsel = select_indices(test_all, args.max_per_grade, rng)
            test_labels = test_all[tsel]
            print(f"[real_test] selected {len(tsel)} / {len(test_all)}  "
                  f"counts={grade_counts(test_labels)}")
            test_images = read_images(args.real_test_h5, tsel)
            blocks.append(build_block(
                "real_test", "test", os.path.abspath(args.real_test_h5),
                test_images, tsel, test_labels, None, 1, args, args.seed,
            ))
            del test_images

        # ---- synthetic pools ---------------------------------------------
        for name, path in synth:
            print(f"[synthetic:{name}] reading {path}")
            all_labels = read_labels(path)
            ssel = select_indices(all_labels, args.max_per_grade, rng)
            slabels = all_labels[ssel]
            print(f"[synthetic:{name}] selected {len(ssel)} / {len(all_labels)}  "
                  f"counts={grade_counts(slabels)}")
            if len(ssel) == 0:
                print(f"[synthetic:{name}] WARNING: empty after selection; skipping")
                continue
            simages = read_images(path, ssel)
            blocks.append(build_block(
                name, "train", os.path.abspath(path),
                simages, ssel, slabels, None, 1, args, args.seed,
            ))
            del simages

        save_cache(cache_path, blocks, cfg)

    # ---- assemble -----------------------------------------------------------
    if args.reuse_features:
        feats = cached["features"]
        thumbs = cached["thumbs"]
        # `meta` already comes from load_cache
    else:
        feats = np.concatenate([b["features"] for b in blocks])
        meta = {
            "source": np.concatenate([np.full(len(b["grade"]), b["source"], dtype="U40")
                                      for b in blocks]),
            "split": np.concatenate([np.full(len(b["grade"]), b["split"], dtype="U10")
                                     for b in blocks]),
            "grade": np.concatenate([b["grade"] for b in blocks]).astype(np.int64),
            "index": np.concatenate([b["index"] for b in blocks]).astype(np.int64),
        }
        thumbs = np.concatenate([b["thumbs"] for b in blocks])

    sources = list(dict.fromkeys(meta["source"].tolist()))
    pairs = make_pairs(sources, args.pairs)
    print(f"\n[done] {len(feats)} points, {feats.shape[1]}-d raw features, "
          f"sources={sources}")
    if pairs:
        print(f"[pairs] {len(pairs)} 1-vs-1 map(s): "
              f"{', '.join(f'{a} vs {b}' for a, b in pairs)}")

    # Give the handcrafted statistics a meaningful share of the variance;
    # without this they are swamped by the ~7k pixel dimensions.
    if args.extra_feats:
        pix_dim = args.pixel_size * args.pixel_size * 3
        pix = feats[:, :pix_dim]
        stats = feats[:, pix_dim:]
        stats = (stats - stats.mean(axis=0)) / (stats.std(axis=0) + 1e-6)
        pix_ss = float(np.einsum("ij,ij->", pix, pix))
        stats_ss = float(np.einsum("ij,ij->", stats, stats))
        if stats_ss > 0:
            scale = np.sqrt(max(args.extra_feats_weight, 0.0) * pix_ss / stats_ss)
            feats[:, pix_dim:] = (stats * scale).astype(np.float32)
            print(f"[features] stats block scaled to "
                  f"{args.extra_feats_weight:.2f}x the pixel-block energy "
                  f"(scale={scale:.3f})")

    # ---- PCA ----------------------------------------------------------------
    k = min(args.pca_components, feats.shape[1], feats.shape[0] - 1)
    print(f"[pca] {feats.shape[0]} x {feats.shape[1]} -> {k} components")
    pca = PCA(n_components=k, svd_solver="randomized", random_state=args.seed)
    z = pca.fit_transform(feats).astype(np.float32)
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"[pca] explained variance (top {k}): {evr:.3f}")

    # ---- projections --------------------------------------------------------
    methods = []
    if args.method == "both":
        methods = ["umap", "tsne"]
    else:
        methods = [args.method]

    projections = {}
    for method in methods:
        print(f"[project] {method}")
        if method == "umap":
            xy = run_umap(z, args)
        elif method == "tsne":
            xy = run_tsne(z, args)
        else:
            xy = z[:, :2].copy()
        projections[method] = np.asarray(xy, dtype=np.float32)

    # ---- projection CSV -----------------------------------------------------
    proj_csv = os.path.join(args.out_dir, "embedding_projection.csv")
    with open(proj_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "source", "split", "grade", "index", "x", "y"])
        for method, xy in projections.items():
            for i in range(len(xy)):
                writer.writerow([
                    method, meta["source"][i], meta["split"][i],
                    int(meta["grade"][i]), int(meta["index"][i]),
                    f"{xy[i, 0]:.6f}", f"{xy[i, 1]:.6f}",
                ])
    print(f"  wrote {proj_csv}")

    # ---- metrics ------------------------------------------------------------
    if not args.no_metrics:
        compute_metrics(
            z, meta,
            os.path.join(args.out_dir, "embedding_metrics.csv"),
            args.seed,
        )
        if pairs:
            compute_pairwise_metrics(
                z, meta, pairs,
                os.path.join(args.out_dir, "pairwise_metrics.csv"),
                args.seed,
            )

    # ---- figures ------------------------------------------------------------
    palette = _source_palette(sources)
    for method, xy in projections.items():
        print(f"[plot] {method}")
        plot_by_source(
            xy, meta,
            os.path.join(args.out_dir, f"{method}_by_source.png"), method,
        )
        plot_by_grade(
            xy, meta,
            os.path.join(args.out_dir, f"{method}_by_grade.png"), method,
        )
        for a, b in pairs:
            plot_pair(
                xy, meta,
                os.path.join(
                    args.out_dir,
                    f"{method}_pair_{_safe_name(a)}_vs_{_safe_name(b)}.png",
                ),
                method, a, b, palette,
            )
        if not args.no_thumbnails and len(thumbs):
            plot_thumbnails(
                xy, meta, thumbs,
                os.path.join(args.out_dir, f"{method}_thumbnails.png"),
                method, args.thumbnails_max, args.seed,
            )

    # ---- config / counts ----------------------------------------------------
    counts = {}
    for src in sources:
        m = meta["source"] == src
        counts[src] = {
            "n": int(m.sum()),
            "per_grade": {str(g): int(np.sum(m & (meta["grade"] == g)))
                          for g in GRADES},
            "path": source_path(src, cfg),
        }
    out_cfg = dict(cfg)
    out_cfg["counts"] = counts
    out_cfg["pca_explained_variance"] = evr
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
    print(f"  wrote {os.path.join(args.out_dir, 'run_config.json')}")

    print(f"\nAll outputs in {os.path.abspath(args.out_dir)}  "
          f"({time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
