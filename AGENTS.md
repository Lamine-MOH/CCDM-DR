# AGENTS.md

## What this is

Fork of UBCDingXin/CCDM (iCCDM) adapted to generate grade-conditioned diabetic retinopathy fundus images. `DRGrading` is added as a fifth `--data_name` option; upstream configs for RC-49/UTKFace/Cell200/SteeringAngle remain unmodified.

## Key commands

### Data preparation

```bash
# Option A: Download pre-built h5 files from Google Drive
# 1. Fill in your Google Drive file IDs in .env.h5_links
# 2. Download datasets (selectable by name and resolution)
python data_preparation/download_h5.py --resolution 128
python data_preparation/download_h5.py --dataset Aptos IDRiD --resolution 256

# Option B: Build h5 files from raw datasets
# 1. Download and prepare a dataset (APTOS, IDRiD, DDR, or Messidor-2)
python data_preparation/get_dataset.py --dataset Aptos
python data_preparation/get_dataset.py --dataset IDRiD --save_path /data

# 2. Build the h5 dataset from the prepared output
python data_preparation/build_dr_h5.py \
    --image_dir ./data/Aptos/Images \
    --csv_path  ./data/Aptos/labels.csv \
    --out_dir   ./data/DRGrading \
    --img_size  128
```

`download_h5.py` reads Google Drive file IDs from `.env.h5_links` and downloads pre-built h5 files into per-dataset subdirectories. `get_dataset.py` downloads the raw dataset and normalizes it into `{save_path}/{dataset_name}/Images/` + `labels.csv` (columns: `id_code, diagnosis`). `build_dr_h5.py` then converts that into the h5 format CCDM-DR expects.

Produces `{out_dir}/{dataset}/DRGrading_{size}x{size}_train.h5` (train) and `DRGrading_{size}x{size}_test.h5` (held-out test). The h5 schema is `images` (uint8, N×3×H×W, CHW) and `labels` (float64, 0-4 ICDR grades).

### Training

```bash
bash config/DR128/run_train.sh    # main 128×128 config
bash config/DR64/run_train.sh     # fast-iteration 64×64 debug config
```

Both scripts call `python main.py` with DR-appropriate flags. Pass `ROOT_PATH` and `DATA_PATH` as positional arguments:

```bash
bash config/DR128/run_train.sh /path/to/CCDM-DR /path/to/DRGrading/Aptos
```

Optional flags to override VRAM defaults:

```bash
# Low-VRAM GPU (e.g. 8GB T4)
bash config/DR64/run_train.sh /path/to/CCDM-DR /path/to/data --batch_size 8 --grad_accum 16

# High-VRAM GPU (e.g. 80GB A100)
bash config/DR128/run_train.sh /path/to/CCDM-DR /path/to/data --batch_size 128 --grad_accum 1
```

### Downstream evaluation (the primary evidence)

```bash
# Real-only baseline
python downstream_eval/train_dr_classifier.py \
    --real_h5 /path/DRGrading_128x128.h5 \
    --test_h5 /path/DRGrading_128x128_test.h5 \
    --backbone resnet50 --epochs 30 --run_name real_only

# Real + CCDM synthetic
python downstream_eval/train_dr_classifier.py \
    --real_h5 /path/DRGrading_128x128.h5 \
    --test_h5 /path/DRGrading_128x128_test.h5 \
    --synthetic_h5 /path/to/generated.h5 \
    --synthetic_cap_per_grade 1500 \
    --backbone resnet50 --epochs 30 --run_name real_plus_synthetic

# Compare runs
python downstream_eval/compare_runs.py --results_dir ./downstream_results
```

## Critical gotchas

- **`--do_eval` is broken for DRGrading.** Built-in SFID/label-score/NIQE require dataset-specific pretrained networks (AE + ResNet34) that ship for upstream datasets but not DRGrading. `evaluation/eval_ckpts/DRGrading/` is empty until you train them yourself (recipe in CcGAN-AVAR repo). Leave `--do_eval` off; both DR configs do this by default.
- **Image data is stored as uint8 (0-255), not float.** The diffusion model expects `[0, 1]` float tensors. `trainer.py` normalizes at load time: `normalize_images(batch_images, to_neg_one_to_one=False)`. Never double-normalize.
- **Labels are normalized to [0, 1] by dividing by `--max_label`.** For DRGrading, `max_label=4`, so grades 0-4 become 0.0-1.0. Denormalization multiplies back by `max_label`.
- **Config YAML `image_size`/`num_channels` must match CLI `--image_size`/`--num_channels`.** `main.py` asserts this at startup.
- **64×64 config is for debugging only.** Fine lesion detail is invisible at 64×64; use 128×128 (or 256) for reported results.

## Architecture notes

- `main.py` — entrypoint. Parses args, loads dataset, trains diffusion model, then samples and evaluates.
- `dataset.py` — `LoadDataSet` loads h5 files. DRGrading uses the UTKFace code path (small integer label set with minority replication).
- `diffusion.py` — EDM-based conditional diffusion model (`ElucidatedDiffusion`).
- `trainer.py` — Training loop using HuggingFace `Accelerator`. Handles vicinity sampling, EMA, and mixed precision.
- `label_embedding.py` — Learns a mapping from labels to embedding vectors.
- `models/` — UNet variants (`unet_edm.py` used by DR configs), DiT, auxiliary ResNet heads.
- `evaluation/evaluator.py` — Computes SFID/LS/IS using pretrained networks. For DRGrading, these checkpoints must be trained separately.
- `config/model_cfg/` — YAML configs for each model+resolution combination.
- `data_preparation/get_dataset.py` — Downloads and normalizes DR datasets into a common structure.
- `data_preparation/build_dr_h5.py` — Converts normalized dataset into h5 format for training.
- `data_preparation/download_h5.py` — Downloads pre-built h5 files from Google Drive using `gdown`. Reads file IDs from `.env.h5_links` (gitignored).

## Dependencies

```
pip install -r requirements.txt
```

Additional DR-specific packages: `opencv-python-headless>=4.8`, `pandas>=1.5`, `scikit-learn>=1.2`. No linting, type-checking, or test framework is configured.

## Output structure

All outputs go to `{root_path}/output/{data_name}_{image_size}/{setting_name}/`:
- `results/model-{step}.pt` — checkpoints (loaded via `--resume_step`)
- `results/log_loss_steps*.txt` — training loss log
- `results/fake_data/` — generated images and h5 dumps
