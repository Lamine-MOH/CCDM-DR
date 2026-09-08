"""filter_synthetic.py — semantic filtering of CCDM synthetic samples (Batch A1).

Applies the MICCAI 2025 (ECC_DM_for_DR) idea to our pool: score every generated
image with a trained real-data classifier ensemble and keep only samples whose
max-likelihood member predicts the target grade, ranked by that likelihood.
Optional dHash near-duplicate removal (2026 synthetic-medical-data survey:
~10% near-duplicates in some pools; dedup can lift accuracy).

Input:  one or more generated h5 files (generate_from_ckpt.py schema:
        images uint8 N x 3 x H x W, labels float64 grades 0-4).
Output per source (into --out_dir):
    {base}_filtered.h5     images/labels + an extra float64 'rank' dataset
                           (the winning softmax likelihood; used by
                           merge_h5_by_grade.py to take top-k by rank instead
                           of a random subsample).
    {base}_scores.npz      labels, per-member preds/probs for ALL samples
                           (diagnostics).
    filter_pass_rate.csv   scale,grade,produced,passed,dedup_removed,kept,
                           pass_rate (the "airbrushed tail" table).
                           Appended to across sources.

Usage:
    python analysis/filter_synthetic.py \
        --sources output/generated_cfg4/generated.h5 output/generated_cs1.5/generated.h5 \
        --ckpts densenet121=downstream_results/real_only_s112_best.pth \
                resnet50=downstream_results/real_only_r50_s111_best.pth \
        --out_dir output/filtered --img_size 128 --batch_size 32

The --ckpts tokens are backbone=path pairs; the backbone name is passed to the
classifier's build_model, so any supported backbone name works.
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from downstream_eval.train_dr_classifier import DRDataset, build_model, evaluate, load_h5

from torch.utils.data import DataLoader

EARLY_GRADES = (0, 1, 2, 3, 4)


def parse_tokens(specs):
    ckpts = {}
    for token in specs:
        if "=" not in token:
            raise ValueError("each --ckpts token must be backbone=path, got '{}'".format(token))
        backbone, path = token.split("=", 1)
        ckpts[backbone] = path
    return ckpts


def to_gray_blocks(img, rows=8, cols=9):
    """CHW uint8 -> grayscale, block-mean 'resize' to rows x cols (pure numpy)."""
    h, w = img.shape[1], img.shape[2]
    gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).astype(np.float64)
    sh, sw = h // rows, w // cols
    gray = gray[: sh * rows, : sw * cols]
    return gray.reshape(rows, sh, cols, sw).mean(axis=(1, 3))


def dhash64(gray_blocks):
    """Difference hash over block grid -> 64-bit int (adjacent-column diffs)."""
    bits = (gray_blocks[:, :, 1:] > gray_blocks[:, :, :-1]).astype(np.uint8)
    bits = bits.reshape(len(bits), -1)
    out = np.zeros(len(bits), dtype=np.uint64)
    for j in range(bits.shape[1]):
        out |= (bits[:, j].astype(np.uint64) << np.uint64(j))
    return out


def popcount64(x):
    return int(x).bit_count()


def near_dup_mask(hashes, confs, hamming=6):
    """Greedy near-dup removal (order = confidence descending): keep a sample
    only if no earlier (higher-confidence) kept sample is within hamming bits."""
    order = np.argsort(-confs, kind="stable")
    kept = np.zeros(len(hashes), dtype=bool)
    kept_list = []
    for pos in order:
        h = hashes[pos]
        if any(popcount64(h ^ prev) <= hamming for prev in kept_list):
            continue
        kept[pos] = True
        kept_list.append(h)
    return kept


def main():
    p = argparse.ArgumentParser(description="Semantic filter of generated DR h5 files")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--ckpts", nargs="+", required=True, help="backbone=path pairs, e.g. densenet121=..._best.pth")
    p.add_argument("--out_dir", default="output/filtered")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dedup_hamming", type=int, default=6,
                   help="0 disables near-duplicate removal")
    args = p.parse_args()

    ckpts = parse_tokens(args.ckpts)
    if not ckpts:
        raise ValueError("need at least one --ckpts entry")
    os.makedirs(args.out_dir, exist_ok=True)

    device = None
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pytorch unavailable: {}".format(exc))

    models_cache = {}
    summary_rows = []
    with open(os.path.join(args.out_dir, "filter_pass_rate.csv"), "w") as summary:
        summary.write("scale,grade,produced,passed,dedup_removed,kept,pass_rate\n")
        for src in args.sources:
            base = os.path.basename(os.path.dirname(src))
            scale = base if base else src
            images, labels = load_h5(src)
            labels_int = np.round(labels).astype(np.int64)

            all_probs, all_preds = [], []
            for backbone, ckpt in ckpts.items():
                if backbone not in models_cache:
                    model = build_model(backbone, pretrained=False).to(device)
                    model.load_state_dict(
                        torch.load(ckpt, map_location=device), strict=True
                    )
                    models_cache[backbone] = model
                model = models_cache[backbone]
                ds = DRDataset(images, labels, train=False, img_size=args.img_size)
                loader = DataLoader(
                    ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
                )
                metrics = evaluate(model, loader, device, return_probs=True)
                all_probs.append(metrics["probs"])
                all_preds.append(metrics["preds"])
            probs = np.stack(all_probs, axis=0)  # K x N x 5
            preds = np.stack(all_preds, axis=0)  # K x N
            member_best = probs.max(axis=2)      # K x N, best class likelihood per member
            winner = member_best.argmax(axis=0)  # N, max-likelihood member index
            ens_pred = preds[winner, np.arange(len(winner))]
            ens_conf = member_best[winner, np.arange(len(winner))]

            np.savez(
                os.path.join(args.out_dir, "{}_scores.npz".format(base)),
                labels=labels_int, preds=preds, probs=probs,
            )

            keep = ens_pred == labels_int
            kept_images, kept_labels, kept_conf = images[keep], labels[keep], ens_conf[keep]

            hamming = args.dedup_hamming
            if hamming > 0:
                dedup_removed = 0
                per_grade = []
                for g in EARLY_GRADES:
                    idx = np.where(np.round(kept_labels) == g)[0]
                    if len(idx) == 0:
                        continue
                    confs = kept_conf[idx]
                    gray = np.stack([to_gray_blocks(im) for im in kept_images[idx]])
                    hashes = dhash64(gray)
                    m = near_dup_mask(hashes, confs, hamming=hamming)
                    per_grade.append(idx[m])
                    dedup_removed += int((~m).sum())
                if per_grade:
                    final_idx = np.concatenate(per_grade)
                    final_idx = final_idx[np.argsort(kept_conf[final_idx], kind="stable")[::-1]]
                    kept_images, kept_labels = kept_images[final_idx], kept_labels[final_idx]
                    kept_conf = kept_conf[final_idx]
            else:
                dedup_removed = 0

            out_path = os.path.join(args.out_dir, "{}_filtered.h5".format(base))
            with h5py.File(out_path, "w") as f:
                f.create_dataset("images", data=kept_images, dtype="uint8",
                                 compression="gzip", compression_opts=6)
                f.create_dataset("labels", data=kept_labels, dtype="float64")
                f.create_dataset("rank", data=kept_conf, dtype="float64")

            n_total = len(images)
            print("  {}: kept {} / {} samples -> {}".format(base, len(kept_images), n_total, out_path))
            for g in EARLY_GRADES:
                produced = int((labels_int == g).sum())
                passed = int((keep & (labels_int == g)).sum())
                final = int((np.round(kept_labels) == g).sum())
                rate = (passed / produced) if produced else 0.0
                print(
                    "    grade {}: produced {} passed {} dedup {} kept {} (pass {:.1%})".format(
                        g, produced, passed, passed - final if produced else 0, final, rate
                    )
                )
                summary.write(
                    "{},{},{},{},{},{},{:.4f}\n".format(
                        scale, g, produced, passed,
                        (passed - final) if produced else 0, final, rate
                    )
                )

    print("\nPass-rate table: {}".format(os.path.join(args.out_dir, "filter_pass_rate.csv")))


if __name__ == "__main__":
    main()