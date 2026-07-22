"""
download_h5.py

Downloads pre-built DRGrading h5 files from a public Google Drive folder.

File IDs are read from a .env.h5_links config file (one KEY=VALUE per line).
Each dataset (Aptos, IDRiD, DDR, Messidor2) needs a TRAIN and TEST entry.

Usage:
    # Download all datasets at 128x128
    python data_preparation/download_h5.py --resolution 128

    # Download only Aptos and IDRiD at 256x256
    python data_preparation/download_h5.py --dataset Aptos IDRiD --resolution 256

    # Custom output directory
    python data_preparation/download_h5.py --resolution 128 --out_dir /data/DRGrading

Output structure:
    {out_dir}/
    ├── Aptos/
    │   ├── DRGrading_128x128_train.h5
    │   └── DRGrading_128x128_test.h5
    ├── IDRiD/
    │   └── ...
    ...
"""

import argparse
import os
import sys

import gdown
import h5py


DATASETS = ["Aptos", "IDRiD", "DDR", "Messidor2"]
RESOLUTIONS = [64, 128, 256]

# Mapping from (dataset, split, resolution) to env file key
# e.g. APTOS_128_TRAIN, IDRiD_64_TEST, etc.
KEY_MAP = {
    ("Aptos", "train", 64): "APTOS_64_TRAIN",
    ("Aptos", "test", 64): "APTOS_64_TEST",
    ("Aptos", "train", 128): "APTOS_128_TRAIN",
    ("Aptos", "test", 128): "APTOS_128_TEST",
    ("Aptos", "train", 256): "APTOS_256_TRAIN",
    ("Aptos", "test", 256): "APTOS_256_TEST",
    ("IDRiD", "train", 64): "IDRID_64_TRAIN",
    ("IDRiD", "test", 64): "IDRID_64_TEST",
    ("IDRiD", "train", 128): "IDRID_128_TRAIN",
    ("IDRiD", "test", 128): "IDRID_128_TEST",
    ("IDRiD", "train", 256): "IDRID_256_TRAIN",
    ("IDRiD", "test", 256): "IDRID_256_TEST",
    ("DDR", "train", 64): "DDR_64_TRAIN",
    ("DDR", "test", 64): "DDR_64_TEST",
    ("DDR", "train", 128): "DDR_128_TRAIN",
    ("DDR", "test", 128): "DDR_128_TEST",
    ("DDR", "train", 256): "DDR_256_TRAIN",
    ("DDR", "test", 256): "DDR_256_TEST",
    ("Messidor2", "train", 64): "MESSIDOR2_64_TRAIN",
    ("Messidor2", "test", 64): "MESSIDOR2_64_TEST",
    ("Messidor2", "train", 128): "MESSIDOR2_128_TRAIN",
    ("Messidor2", "test", 128): "MESSIDOR2_128_TEST",
    ("Messidor2", "train", 256): "MESSIDOR2_256_TRAIN",
    ("Messidor2", "test", 256): "MESSIDOR2_256_TEST",
}


def parse_env_file(path):
    """Parse a KEY=VALUE env file, ignoring comments and blank lines."""
    links = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            links[key.strip()] = val.strip()
    return links


def download_file(file_id, output_path):
    """Download a file from Google Drive using gdown."""
    if os.path.exists(output_path):
        print(f"  Already exists, skipping: {output_path}")
        return True
    try:
        gdown.download(id=file_id, output=output_path, quiet=False, fuzzy=True)
        return True
    except TypeError:
        gdown.download(id=file_id, output=output_path, quiet=False)
        return True
    except Exception as e:
        print(f"  Download failed: {e}", file=sys.stderr)
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def verify_h5(path):
    """Quick sanity check: open the h5 and print dataset shapes."""
    try:
        with h5py.File(path, "r") as hf:
            images = hf["images"]
            labels = hf["labels"]
            print(f"    images: {images.shape}, dtype={images.dtype}")
            print(f"    labels: {labels.shape}, dtype={labels.dtype}")
            print(f"    label range: [{labels[:].min()}, {labels[:].max()}]")
        return True
    except Exception as e:
        print(f"    Verification failed: {e}", file=sys.stderr)
        return False


def main():
    p = argparse.ArgumentParser(
        description="Download pre-built DRGrading h5 files from Google Drive."
    )
    p.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASETS + ["all"],
        default=["all"],
        help="Which datasets to download (default: all)",
    )
    p.add_argument(
        "--resolution",
        type=int,
        choices=RESOLUTIONS,
        default=128,
        help="Image resolution (default: 128)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="./data/DRGrading",
        help="Output directory (default: ./data/DRGrading)",
    )
    p.add_argument(
        "--env_file",
        type=str,
        default=".env.h5_links",
        help="Path to env file with Google Drive IDs (default: .env.h5_links)",
    )
    p.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip h5 integrity verification after download",
    )
    args = p.parse_args()

    # Resolve datasets
    datasets = DATASETS if "all" in args.dataset else args.dataset

    # Load env file
    env_path = os.path.join(os.path.dirname(__file__), "..", args.env_file)
    if not os.path.exists(env_path):
        print(f"Error: env file not found: {env_path}", file=sys.stderr)
        print(f"Create it with Google Drive file IDs. See .env.h5_links.", file=sys.stderr)
        sys.exit(1)

    links = parse_env_file(env_path)
    print(f"Loaded {len(links)} entries from {env_path}")

    res = args.resolution
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} | Resolution: {res}x{res}")
        print(f"{'='*60}")

        ds_dir = os.path.join(args.out_dir, dataset)
        os.makedirs(ds_dir, exist_ok=True)

        for split in ["train", "test"]:
            key = KEY_MAP[(dataset, split, res)]
            if key not in links:
                print(f"  WARNING: {key} not found in env file, skipping {split} split")
                results.append((dataset, split, "skipped"))
                continue

            file_id = links[key]
            filename = f"DRGrading_{res}x{res}_{split}.h5"
            output_path = os.path.join(ds_dir, filename)

            print(f"\n  Downloading {split} split...")
            print(f"  File ID: {file_id}")
            print(f"  Output:  {output_path}")

            success = download_file(file_id, output_path)
            if success and not args.no_verify:
                print(f"  Verifying...")
                verify_h5(output_path)

            status = "ok" if success else "failed"
            results.append((dataset, split, status))

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    for dataset, split, status in results:
        print(f"  {dataset:12s} {split:5s}  {status}")

    failed = [r for r in results if r[2] == "failed"]
    if failed:
        print(f"\n{len(failed)} download(s) failed.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll downloads completed successfully.")
        print(f"\nTo train on a specific dataset:")
        print(f"  export DATA_PATH={args.out_dir}/<Dataset>")
        print(f"  bash config/DR128/run_train.sh")


if __name__ == "__main__":
    main()
