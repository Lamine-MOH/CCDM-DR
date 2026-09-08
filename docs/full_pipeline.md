# CCDM-DR: Full Pipeline (Start to Finish)

Everything needed to go from a raw DR dataset to a trained grade-conditioned
diffusion model, generated synthetic images, and downstream evidence that the
synthetic data helps DR grading.

All paths are relative to the repo root:
`/home/lamine/Documents/PhD/Projects/CCDM DR/CCDM-DR`

---

## Table of contents

1. [Environment setup](#1-environment-setup)
2. [Data preparation](#2-data-preparation)
3. [Training](#3-training)
4. [Phased / resumed training](#4-phased--resumed-training)
5. [Generating images from a checkpoint (no re-training)](#5-generating-images-from-a-checkpoint-no-re-training)
6. [Downstream evaluation (the primary evidence)](#6-downstream-evaluation-the-primary-evidence)
7. [Output structure](#7-output-structure)
8. [Gotchas](#8-gotchas)

---

## 1. Environment setup

Create a dedicated conda environment (see `guide.md`):

```bash
conda create -n ccdm-dr python=3.10 -y
conda activate ccdm-dr
```

Install PyTorch with CUDA 12.1 support **before** the other packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install the repo requirements plus the DR-specific packages:

```bash
pip install -r requirements.txt
pip install opencv-python-headless>=4.8 pandas>=1.5 scikit-learn>=1.2
```

> Note: `requirements.txt` also pulls in `ema-pytorch`, which the EMA
> checkpoint loading in `generate_from_ckpt.py` depends on. No linting,
> type-checking, or test framework is configured.

---

## 2. Data preparation

There are two paths to get the h5 files CCDM-DR consumes.

### Option A: Download pre-built h5 files from Google Drive

1. Fill in your Google Drive file IDs in `.env.h5_links` (gitignored).
2. Download, selecting dataset(s) and resolution:

```bash
python data_preparation/download_h5.py --resolution 128
python data_preparation/download_h5.py --dataset Aptos IDRiD --resolution 256
```

### Option B: Build h5 files from raw datasets

1. Download and normalize a raw dataset. `get_dataset.py` supports
   `Aptos`, `IDRiD`, `DDR`, and `Messidor-2`:

```bash
python data_preparation/get_dataset.py --dataset Aptos
python data_preparation/get_dataset.py --dataset IDRiD --save_path /data
```

   This writes `{save_path}/{dataset_name}/Images/*.png` and
   `{save_path}/{dataset_name}/labels.csv` with columns `id_code, diagnosis`.

2. Convert the normalized dataset into the h5 format CCDM-DR expects:

```bash
python data_preparation/build_dr_h5.py \
    --image_dir ./data/Aptos/Images \
    --csv_path  ./data/Aptos/labels.csv \
    --out_dir   ./data/DRGrading \
    --img_size  128
```

   Options: `--test_frac 0.15` (default held-out split), `--clahe`, `--seed 111`.
   Produces:
   - `{out_dir}/{dataset}/DRGrading_{size}x{size}_train.h5`
   - `{out_dir}/{dataset}/DRGrading_{size}x{size}_test.h5`

   h5 schema: `images` (uint8, N×3×H×W, CHW) and `labels` (float64, 0-4 ICDR
   grades). The script prints per-grade class counts — use these to retune the
   vicinity hyperparameters (`--kappa`, `--min_n_per_vic`) if imbalance is
   severe.

---

## 3. Training

Two configs ship with the repo. Both call `python main.py` with
DR-appropriate flags; pass `ROOT_PATH` and `DATA_PATH` as positional args:

```bash
# Main 128x128 config (150k steps, batch 64 x grad-accum 2, ~24GB VRAM)
bash config/DR128/run_train.sh /path/to/CCDM-DR /path/to/DRGrading/Aptos

# Fast-iteration 64x64 debug config (60k steps) — pipeline debugging only,
# fine lesion detail is invisible at 64x64
bash config/DR64/run_train.sh /path/to/CCDM-DR /path/to/DRGrading/Aptos
```

### Optional flags

```bash
# Low-VRAM GPU (e.g. 8GB T4)
bash config/DR64/run_train.sh /path/to/CCDM-DR /path/to/data --batch_size 8 --grad_accum 16

# High-VRAM GPU (e.g. 80GB A100)
bash config/DR128/run_train.sh /path/to/CCDM-DR /path/to/data --batch_size 128 --grad_accum 1

# Stop early / resume / change checkpoint cadence
bash config/DR128/run_train.sh /path/to/CCDM-DR /path/to/data \
    --num_steps 100000 --save_every 5000
```

Recognized flags (both scripts): `--num_steps`, `--batch_size`, `--grad_accum`,
`samp_batch_size`, `--resume_step`, `--save_every`, `--skip_final_sampling`.

### What the training script actually runs

```bash
python main.py \
    --setting_name setup1_dr \
    --root_path $ROOT_PATH --data_name DRGrading --data_path $DATA_PATH \
    --num_channels 3 --image_size 128 \
    --min_label 0 --max_label 4 \
    --model_config "./config/model_cfg/unet_edm_128_v1.yaml" \
    --y2h_embed_type "resnet" \
    --use_y2cov --y2cov_hy_weight_train 0.05 --y2cov_hy_weight_test 0.05 \
    --y2cov_embed_type "resnet" --net_embed_y2cov_y2emb "cnn" \
    --train_num_steps 150000 --train_lr 1e-5 \
    --train_batch_size 64 --gradient_accumulate_every 2 \
    --train_amp --train_mixed_precision fp16 \
    --kernel_sigma -1.0 --threshold_type hard --kappa -1.0 \
    --use_ada_vic --ada_vic_type vanilla --min_n_per_vic 50 --use_symm_vic \
    --sample_every 2500 --save_every 10000 \
    --sampler sde --num_sample_steps 32 \
    --sample_cond_scale 1.5 --sample_cond_rescaled_phi 0.7 \
    --nfake_per_label 1000 --samp_batch_size 100 \
    --dump_fake_data \
    2>&1 | tee output_DRGrading_128_setup1_dr.txt
```

> `--do_eval` is broken for DRGrading (needs dataset-specific pretrained AE +
> ResNet34 eval checkpoints you must train yourself — see `evaluation/`). Both
> DR configs leave it off by default.

### Monitoring

Tail the log the script tees to:

```bash
tail -f output_DRGrading_128_setup1_dr.txt
```

Training is checkpointed **by step**, not epoch. Checkpoints land in
`{root}/output/DRGrading_128/setup1_dr/results/model-{step}.pt` every
`--save_every` steps; sample grids every `--sample_every` steps.

---

## 4. Phased / resumed training

`main.py` writes self-contained, machine-portable snapshots
(`{step, model, opt, ema, scaler}`). Resume with `--resume_step`.
**`--train_num_steps` is the TOTAL step count, not remaining.**

```bash
# Phase 1 (machine A): train 0 -> 50k steps, no sampling overhead
bash config/DR128/run_train.sh /path/to/CCDM-DR /path/to/DRGrading/Aptos \
    --num_steps 50000 --skip_final_sampling

# Phase 2 (machine B): continue 50k -> 100k
```

Before resuming, copy to machine B at the **same relative root_path**:
- `output/DRGrading_128/setup1_dr/results/model-*.pt` (diffusion checkpoints)
- `output/DRGrading_128/model_y2h/`, `output/DRGrading_128/model_y2cov/`,
  `output/DRGrading_128/aux_reg_model/` (embedding nets; loaded-if-present —
  copy so labels stay byte-identical and aren't retrained)

Then resume with the same hyperparameters as phase 1:

```bash
bash config/DR128/run_train.sh /path/to/CCDM-DR /path/to/DRGrading/Aptos \
    --num_steps 100000 --resume_step 50000
```

Rules:
- `--skip_final_sampling` ends a phase right after training (no sampling /
  `--dump_fake_data` block). Leave it off on the **final** phase so fake data
  is dumped.
- `--resume_step` must be a multiple of `--save_every`.
- Checkpoints don't store hyperparameters or RNG state — pass the same flags
  every phase. Single-GPU resume is safe; multi-GPU has a DDP-wrap quirk in
  `trainer.load()`.
- The loss log is named `log_loss_steps{train_num_steps}.txt`, so a changed
  total starts a fresh log file.

---

## 5. Generating images from a checkpoint (no re-training)

`generate_from_ckpt.py` samples a trained model **without** loading the
training set. It needs the diffusion checkpoint **plus** the embedding nets and
the training yaml (none of those live inside `model-*.pt`):

```bash
python generate_from_ckpt.py \
    --model_ckpt output/DRGrading_128/setup1_dr/results/model-100000.pt \
    --model_config config/model_cfg/unet_edm_128_v1.yaml \
    --root_path . \
    --image_size 128 \
    --grades 0 1 2 3 4 \
    --nfake_per_grade 1000 \
    --cond_scale 4
# out_dir defaults to output/generated_cs4; the h5 stores its generation attrs
```

It looks for the embedding checkpoints at the default locations:

- `{root_path}/output/DRGrading_{image_size}/model_y2h/ckpt_mlp_y2h_epoch_500.pth`
- `{root_path}/output/DRGrading_{image_size}/model_y2cov/ckpt_cnn_y2cov_epoch_500.pth`

Override with `--path_y2h` / `--path_y2cov` / `--y2h_ckpt_name` /
`--y2cov_ckpt_name` if you renamed them.

### Useful options

| Option | Default | Meaning |
|---|---|---|
| `--sampler` | `sde` | `sde` (stochastic, best quality), `ode` (deterministic), `dpmpp` (fastest) |
| `--num_sample_steps` | `32` | sampling steps (more = better, slower) |
| `--cond_scale` | `1.5` | classifier-free guidance strength |
| `--rescaled_phi` | `0.7` | CFG rescaling (EDM) |
| `--batch_size` | `100` | samples per forward pass |
| `--grades` | `0 1 2 3 4` | which grades to generate |
| `--nfake_per_grade` | `1000` | images per grade |
| `--out_dir` | `output/generated` | output folder |
| `--device` | auto | `cuda` or `cpu` |
| `--no-use_y2cov` | (on) | disable label-dependent covariance |

> The EDM hyperparameter flags (`--edm_sigma_min`, `--edm_sigma_max`,
> `--edm_rho`, `--edm_S_churn`, ...) and the EMA flags must match training.
> The script's defaults match the DR128 config; don't change them unless your
> training run did.

### Output

- `{out_dir}/generated.h5` — same schema as the training h5 (`images` uint8
  N×3×H×W, `labels` float64 **raw grades 0-4**), ready for
  `train_dr_classifier.py --synthetic_h5`.
- `{out_dir}/sample_grade_{0..4}.png` — per-grade preview grids.

---

## 6. Downstream evaluation (the primary evidence)

Train a standard DR classifier under different data conditions and compare
held-out performance. Metrics: Accuracy, Macro-F1, Quadratic Weighted Kappa
(QWK — the standard DR grading metric), and per-class recall.

```bash
# (A) Real-only baseline
python downstream_eval/train_dr_classifier.py \
    --real_h5 /path/DRGrading_128x128_train.h5 \
    --test_h5 /path/DRGrading_128x128_test.h5 \
    --backbone resnet50 --epochs 30 --run_name real_only

# (B) Real + CCDM synthetic
python downstream_eval/train_dr_classifier.py \
    --real_h5 /path/DRGrading_128x128_train.h5 \
    --test_h5 /path/DRGrading_128x128_test.h5 \
    --synthetic_h5 /path/to/generated.h5 \
    --synthetic_cap_per_grade 1500 \
    --backbone resnet50 --epochs 30 --run_name real_plus_synthetic

# (C) Optional: real + classic augmentation baseline (what synthetic must beat)
#     (run train_dr_classifier.py without --synthetic_h5, but note: the classic
#      augmentation baseline lives in dr_benchmark; see train_dr_classifier.py
#      docstring for the intended comparison)
```

Compare runs (reads the JSON metrics dumped to `--out_dir`):

```bash
python downstream_eval/compare_runs.py --results_dir ./downstream_results
```

Key args for `train_dr_classifier.py`:
`--backbone` (`resnet50` or `efficientnet_b4`), `--epochs`, `--batch_size`,
`--lr`, `--synthetic_cap_per_grade` (cap how many synthetic images per grade
are mixed in), `--out_dir` (default `./downstream_results`), `--seed`.

---

## 7. Output structure

All outputs go under `{root_path}/output/DRGrading_{image_size}/`:

```
output/DRGrading_128/
├── model_y2h/                       # label->h embedding net (trained, 500 ep)
│   ├── ckpt_mlp_y2h_epoch_500.pth
│   ├── ckpt_resnet_y2h_epoch_10.pth
│   └── resnet_y2h_ckpt_in_train/
├── model_y2cov/                     # label->covariance embedding net
│   ├── ckpt_cnn_y2cov_epoch_500.pth
│   ├── ckpt_resnet_y2cov_epoch_10.pth
│   └── resnet_y2cov_ckpt_in_train/
├── aux_reg_model/                   # auxiliary regression head (loaded-if-present)
└── setup1_dr/
    ├── setting_info.txt             # the exact CLI used
    └── results/
        ├── log_loss_steps150000.txt # training loss log
        ├── model-{step}.pt          # diffusion checkpoints (EMA inside)
        ├── sample_{sde,ode}_{step}.png   # sample grids (every --sample_every)
        ├── fake_data/               # final dump (last phase only)
        └── fake_data_steps{n}_nfake{n}_{sampler}_scale{c}_sampstep{s}_trw{w}_tew{w}/
            ├── {grade}.h5           # per-grade fake image dumps
            └── sample_{grade}.png
```

Downstream results go to `./downstream_results/*_metrics.json`.

---

## 8. Gotchas

- **`--do_eval` is broken for DRGrading.** SFID/label-score/NIQE need
  dataset-specific pretrained networks that don't ship for DRGrading
  (`evaluation/eval_ckpts/DRGrading/` is empty). Leave it off.
- **Images are uint8 (0-255) in h5**, but the diffusion model expects `[0,1]`
  float tensors. `trainer.py` normalizes at load time; never double-normalize.
- **Labels are normalized to `[0,1]` by dividing by `--max_label` (4 for DR)**,
  so grades 0-4 become 0.0-1.0. Denormalization multiplies back.
- **Config YAML `image_size`/`num_channels` must match CLI flags** — `main.py`
  asserts this at startup.
- **64x64 is debugging only.** Use 128 (or 256) for reported results.
- **Checkpoints don't store hyperparameters or the embedding nets.** Ship
  `model_y2h/`, `model_y2cov/`, `aux_reg_model/`, and the training yaml along
  with `model-*.pt` whenever you move machines or use
  `generate_from_ckpt.py`.
