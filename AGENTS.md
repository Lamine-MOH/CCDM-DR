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
    --img_size  128 \
    --test_frac 0.20 --seed 111
```

`download_h5.py` reads Google Drive file IDs from `.env.h5_links` and downloads pre-built h5 files into per-dataset subdirectories. `get_dataset.py` downloads the raw dataset and normalizes it into `{save_path}/{dataset_name}/Images/` + `labels.csv` (columns: `id_code, diagnosis`). `build_dr_h5.py` then converts that into the h5 format CCDM-DR expects.

Produces `{out_dir}/{dataset}/DRGrading_{size}x{size}_train.h5` (train) and `DRGrading_{size}x{size}_test.h5` (held-out test). The h5 schema is `images` (uint8, N×3×H×W, CHW) and `labels` (float64, 0-4 ICDR grades).

The train/test split is **STRATIFIED BY GRADE** at `--test_frac` (default `0.20` = 80/20), seeded by `--seed` (default 111). If any grade has fewer than `--min_test_per_grade` test samples (default 10) the script errors and tells you to raise `--test_frac`. Note: for DRGrading the source pool is the merged original train+test+val (per `get_dataset.py`), so this is a within-pool holdout, not the competition's official split.

### Training

```bash
bash config/DR128/run_train.sh    # main 128×128 config
bash config/DR256/run_train.sh    # high-res 256×256 config
bash config/DR64/run_train.sh     # fast-iteration 64×64 debug config
```

All three scripts call `python main.py` with DR-appropriate flags. Pass `ROOT_PATH` and `DATA_PATH` as positional arguments:

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

### Phased / resumed training across machines

Training is checkpointed by **step** (not epoch). `main.py` writes `results/model-{step}.pt` every `--save_every` steps containing `{step, model, opt, ema, scaler}` — a self-contained, machine-portable snapshot. Resume with `--resume_step`, and remember **`--train_num_steps` is the TOTAL step count, not the remaining ones**.

```bash
# Phase 1 (machine A): train 0 -> 50k steps. Saves model-10000 ... model-50000.pt
bash config/DR128/run_train.sh /path/to/CCDM-DR /path/to/DRGrading/Aptos \
    --num_steps 50000 --skip_final_sampling

# Phase 2 (machine B): continue 50k -> 100k.
# 1. Copy to machine B at the SAME relative root_path:
#      output/DRGrading_128/setup1_dr/results/model-*.pt   (diffusion checkpoints)
#      output/DRGrading_128/model_y2h/  model_y2cov/  aux_reg_model/  (embedding nets,
#      loaded-if-present; copy so labels stay byte-identical and aren't retrained)
# 2. Resume with the same hyperparameters as phase 1 (lr, batch, vicinity, etc.):
bash config/DR128/run_train.sh /path/to/CCDM-DR /path/to/DRGrading/Aptos \
    --num_steps 100000 --resume_step 50000
```

Notes:

- `--skip_final_sampling` ends the phase right after training, skipping the post-training sampling/`--dump_fake_data` block (wasteful + overwrites `fake_data/` on intermediate phases). Leave it off on the final phase so fake data is dumped.
- `--resume_step` must be a multiple of `--save_every` (only those checkpoints exist).
- Checkpoints don't store hyperparameters or RNG state; pass the same flags each phase (batch sampling is with replacement, so continuation is stochastic but correct). The resume path is single-GPU safe (scripts use `CUDA_VISIBLE_DEVICES=0`); multi-GPU resume has a DDP-wrap quirk in `trainer.load()`.
- The loss log is named `log_loss_steps{train_num_steps}.txt`, so a changed total starts a fresh log file.

> **Retrain-from-scratch guide (post-conditioning-fix):** `docs_private/RETRAIN_GUIDE.md` (private, gitignored — working memo for the owner; may be stale). Public reproducibility relies on `docs/DOWNSTREAM_RESULTS.md` (protocol, provenance, regeneration). Breaks the UNet architecture (dedicated per-block `affine_cond` for label embeddings, decoupled from time) and wires the previously-dead `--epoch_cnn_embed`/`--epoch_cnn_embed_y2cov`/`--epoch_net_y2h`/`--epoch_net_y2cov`/`--batch_size_embed*` flags into `label_embedding.py` (y2h/y2cov encoders now train 200 epochs, not 10). Old `model-*.pt`/`model_y2h`/`model_y2cov` outputs are incompatible — delete before training. Validate with `bash analysis/run_diagnosis.sh ROOT DATA --synth_h5 ...` (A1 trace + A3 montages; optional A2 classifier), see `analysis/README.md`. **Next full retrain (Tier-3, see `docs_private/FIX_PLAN.md`):** also lands LayerNorm in `unet_edm.cond_map`, the EDM loss-weight fix in `diffusion.py`, and wires `--num_img_per_label_after_replica 1000` (embedding-net minority replication) into all three DR `run_train.sh` configs — invalidating the existing checkpoints again.

### Downstream evaluation (the primary evidence)

Supported backbones: `resnet50`, `resnet101`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5`, `densenet121`, `densenet201`. All use ImageNet-pretrained weights by default; pass `--no-pretrained` to train from scratch.

> **Planned protocol changes (audit fixes, not yet landed — see `docs_private/FIX_PLAN.md`):**
> - Best-epoch model selection will move **off the test set** onto a validation split carved from the
>   real train h5 (`--val_frac 0.12 --val_seed 999`, stratified, identical across arms/seeds), with a
>   single final test evaluation of the val-selected weights.
> - `--no-pretrained` is currently a **no-op** (`action="store_true"` + `default=True`); the flag fix
>   (`BooleanOptionalAction`) lands in Phase 1.
> - The multi-backbone headline protocol is `densenet121` + `resnet50` + `efficientnet_b4`.
> - Generated synthetic sets self-describe `edm_sigma_data_type` once the sigma-data metadata lands.
> Until those land, results below reflect best-epoch-on-test selection (optimistic).

`--synthetic_cap_per_grade N` limits synthetic samples per DR grade (0-4) before concatenation with real data. Useful for sweeping synthetic-to-real ratios.

```bash
# Real-only baseline
python downstream_eval/train_dr_classifier.py \
    --real_h5 /path/DRGrading_128x128_train.h5 \
    --test_h5 /path/DRGrading_128x128_test.h5 \
    --backbone resnet50 --epochs 30 --run_name real_only

# Real + CCDM synthetic (cap at 1500 synthetic samples per grade)
python downstream_eval/train_dr_classifier.py \
    --real_h5 /path/DRGrading_128x128_train.h5 \
    --test_h5 /path/DRGrading_128x128_test.h5 \
    --synthetic_h5 /path/to/generated.h5 \
    --synthetic_cap_per_grade 1500 \
    --backbone resnet50 --epochs 30 --run_name real_plus_synthetic

# Compare runs
python downstream_eval/compare_runs.py --results_dir ./downstream_results
```

Outputs per run: `{run_name}_best.pth` (model weights) and `{run_name}_metrics.json` (accuracy, macro_f1, QWK, per-grade recall). `compare_runs.py` aggregates all `*_metrics.json` into a comparison table saved as `comparison_table.csv`. **Classifier results are kept local**: `downstream_results/` is gitignored and must never be committed.

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
- `data_preparation/download_h5.py` — Downloads pre-built h5 files from Google Drive using `gdown`. Reads file IDs from `.env.h5_links` (committed).
- `generate_from_ckpt.py` — Samples a trained checkpoint without re-training (needs `model-*.pt` + embedding nets + training yaml); writes `generated.h5` for `train_dr_classifier.py --synthetic_h5`. It reads `edm_sigma_data.json` persisted next to the training checkpoints so the exact sigma-data configuration (`default`/`local`) is reproduced (planned — currently falls back to `default` 0.5). `--out_dir` defaults to `output/generated_cs{cond_scale}`, and the h5 stores generation attrs (`cond_scale`, `model_ckpt`, `sampler`, ...) so a generated set is self-describing.
- `docs/DOWNSTREAM_RESULTS.md` — **superseded** (status now `SUPERSEDED — pending regeneration`): historical 5-seed protocol + mean±std tables, blendA provenance, Batch-A/E16/C9 records. New numbers come from `docs_private/FIX_PLAN.md` + regeneration after the Tier-3 retrain.

## Dependencies

```
pip install -r requirements.txt
```

Additional DR-specific packages: `opencv-python-headless>=4.8`, `pandas>=1.5`, `scikit-learn>=1.2`, `gdown>=5.0`. `ema-pytorch` is required by `generate_from_ckpt.py` (included in requirements.txt). No linting, type-checking, or test framework is configured.

## Output structure

All outputs go to `{root_path}/output/{data_name}_{image_size}/{setting_name}/`:
- `results/model-{step}.pt` — checkpoints (loaded via `--resume_step`)
- `results/log_loss_steps*.txt` — training loss log
- `results/fake_data/` — generated images and h5 dumps
