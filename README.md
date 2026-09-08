# CCDM-DR — a diabetic retinopathy fork of UBCDingXin/CCDM (iCCDM)

This is a fork of the `iCCDM` folder from [UBCDingXin/CCDM](https://github.com/UBCDingXin/CCDM)
(EDM-based continuous conditional diffusion model), adapted to generate
grade-conditioned diabetic retinopathy (DR) fundus images. `DRGrading` is
added as a fifth `--data_name` option; the loading/training/sampling logic
for it lives alongside the original RC-49/UTKFace/SteeringAngle/Cell-200
code paths, which still work as documented upstream.

Unlike a typical fork, this one also had unused pieces of the upstream
tree removed — see "What was removed" below if something you expected
from the original repo isn't here.

## Quick start

```bash
pip install -r requirements.txt

# Option A: Download pre-built h5 files from Google Drive
# 1. Fill in your Google Drive file IDs in .env.h5_links (pre-filled for all datasets)
# 2. Download datasets (selectable by name and resolution)
python data_preparation/download_h5.py --resolution 128
python data_preparation/download_h5.py --dataset Aptos IDRiD --resolution 256

# Option B: Build h5 files from raw datasets
# 1. Download and prepare a dataset (APTOS, IDRiD, DDR, or Messidor-2)
python data_preparation/get_dataset.py --dataset Aptos

# 2. Build the h5 dataset from the prepared output
python data_preparation/build_dr_h5.py \
    --image_dir ./data/Aptos/Images \
    --csv_path  ./data/Aptos/labels.csv \
    --out_dir   ./data/DRGrading \
    --img_size  128

# 3. Train (pass ROOT_PATH and DATA_PATH as arguments)
bash config/DR128/run_train.sh /path/to/CCDM-DR /path/to/DRGrading/Aptos
bash config/DR256/run_train.sh /path/to/CCDM-DR /path/to/DRGrading/Aptos
bash config/DR64/run_train.sh /path/to/CCDM-DR /path/to/DRGrading/Aptos

# Override VRAM defaults if needed (--batch_size, --grad_accum, --samp_batch_size)
bash config/DR64/run_train.sh /path/to/CCDM-DR /path/to/data --batch_size 8 --grad_accum 16

# 4. Generate images from a trained checkpoint (no re-training)
python generate_from_ckpt.py \
    --model_ckpt output/DRGrading_128/setup1_dr/results/model-100000.pt \
    --model_config config/model_cfg/unet_edm_128_v1.yaml \
    --root_path . --image_size 128 \
    --grades 0 1 2 3 4 --nfake_per_grade 1000 --cond_scale 4
# (out_dir defaults to output/generated_cs4; the h5 stores its generation attrs)

# 5. Evaluate: does the synthetic data actually help a DR classifier?
python downstream_eval/train_dr_classifier.py \
    --real_h5 /path/DRGrading_128x128_train.h5 \
    --test_h5 /path/DRGrading_128x128_test.h5 \
    --backbone resnet50 --epochs 30 --run_name real_only

python downstream_eval/train_dr_classifier.py \
    --real_h5 /path/DRGrading_128x128_train.h5 \
    --test_h5 /path/DRGrading_128x128_test.h5 \
    --synthetic_h5 /path/to/generated.h5 --synthetic_cap_per_grade 1500 \
    --backbone resnet50 --epochs 30 --run_name real_plus_synthetic

python downstream_eval/compare_runs.py --results_dir ./downstream_results
# Frozen 5-seed protocol & results: docs/DOWNSTREAM_RESULTS.md
# (analysis/seed_report.py aggregates per-seed *_metrics.json into mean ± std)
```

The data-preparation scripts produce `{out_dir}/{dataset}/DRGrading_{size}x{size}_train.h5`
(train) and `DRGrading_{size}x{size}_test.h5` (held-out test) with the schema
`images` (uint8, N×3×H×W, CHW) and `labels` (float64, 0-4 ICDR grades) — pass
those exact `*_train.h5` / `*_test.h5` paths to the downstream-eval commands.

The frozen 5-seed downstream experiment (protocol, mean±std table, blendA
provenance, regeneration commands) is in `docs/DOWNSTREAM_RESULTS.md`.

See `AGENTS.md` for a denser command/gotcha reference, and the
"eval-checkpoint gap" section below before you touch `--do_eval`. A
start-to-finish walkthrough lives in `docs/full_pipeline.md`, and the private
retrain-from-scratch memo in `docs_private/RETRAIN_GUIDE.md` (local companion
docs, not needed for reproducing the published results).

## What was changed vs. upstream

| File | Change |
|---|---|
| `dataset.py` | Added a `DRGrading` branch to `LoadDataSet`. DR severity grades are discrete integers 0-4 (ICDR scale), so this reuses the same loading/minority-replication logic already written for `UTKFace` (also a small integer label set) rather than the continuous-label logic used for `SteeringAngle`/`RC-49`. Now expects `{data_path}/DRGrading_{size}x{size}_train.h5` (with `_train` suffix). |
| `opts.py` | Added `"DRGrading"` to the `--data_name` choices. |
| `evaluation/evaluator.py` | Added a `DRGrading` branch pointing to `./evaluation/eval_ckpts/DRGrading/...`. **These checkpoints don't exist yet** — see "The eval-checkpoint gap" below. |
| `main.py` | **Bug fix.** Upstream unconditionally imports `evaluation/eval_models/{data_name}/metrics_{size}x{size}` right after training, regardless of `--do_eval`. That path doesn't exist for `DRGrading`, so every DR run would otherwise crash right after sampling — after the GPU time for training was already spent. This is now gated behind `if args.do_eval:` (a no-op for the original datasets). Also: device handling refactored for CPU fallback. |
| `trainer.py` | Modified: added a visual-sample batch-size parameter, removed a batch-size-divisibility assertion, guarded covariance updates against empty label sets, and CPU-fallback device handling. |
| `label_embedding.py`, `utils.py` | Lightly modified (CPU fallback in embedding/prediction device handling). |
| `data_preparation/get_dataset.py` | **New.** Downloads and normalizes APTOS/IDRiD/DDR/Messidor-2 into a common `{dataset}/Images/` + `labels.csv` structure. |
| `data_preparation/build_dr_h5.py` | **New.** Converts a fundus image folder + CSV of grades into the h5 format `dataset.py` expects, with fundus-specific preprocessing (circular field-of-view crop, optional `--clahe`), a held-out `--test_frac` split, and per-grade class-count printouts. |
| `data_preparation/download_h5.py` | **New.** Downloads pre-built h5 files from Google Drive using `gdown`. Reads file IDs from `.env.h5_links` (committed). Supports `--dataset` and `--resolution` flags for selective downloads. |
| `config/DR128/run_train.sh`, `config/DR256/run_train.sh`, `config/DR64/run_train.sh` | **New.** Training configs for DR: 128×128 (main), 256×256 (high-res), 64×64 (fast debug). All accept `ROOT_PATH`/`DATA_PATH` positionally plus overridable `--num_steps`, `--batch_size`, `--grad_accum`, `--samp_batch_size`, `--resume_step`, `--save_every`, `--skip_final_sampling`. |
| `generate_from_ckpt.py` | **New.** Samples a trained checkpoint without re-training or loading the training set — the diffusion weights come from `model-{step}.pt`; the label-embedding nets and training yaml must be supplied separately. Writes `generated.h5` in the same schema as the training h5, ready for `train_dr_classifier.py --synthetic_h5`. `--out_dir` defaults to `output/generated_cs{cond_scale}` and the h5 stores its generation attrs (`cond_scale`, `model_ckpt`, ...) so a generated set is self-describing. |
| `downstream_eval/train_dr_classifier.py` | **New.** Trains a DR grading classifier (ResNet50/101, EfficientNet-B3..B5, DenseNet121/201) under real-only vs. real+synthetic conditions and reports accuracy/macro-F1/QWK. This is the primary evidence for the contribution. |
| `downstream_eval/compare_runs.py` | **New.** Summarizes multiple `train_dr_classifier.py` runs into one comparison table. |
| `analysis/` | **New.** Post-training diagnostics and downstream helpers: `trace_conditioning.py` (A1 grade-separability trace), `render_inspection_montages.py` (A3 per-grade montages), `run_diagnosis.sh`, `check_embedding.py` (E1), `sweep_cond_scale.py` (E2 CFG sweep, includes previews), `merge_h5_by_grade.py` (per-grade blend across CFG scales), `tag_h5.py` (stamp generation attrs on legacy h5s), `seed_report.py` (mean±std aggregation across classifier seeds). See `analysis/README.md`. |

## What was removed vs. upstream

Everything below was specific to the four original datasets or was dead
code, and isn't needed to train or evaluate on DR. Pull any of it back from
[UBCDingXin/CCDM](https://github.com/UBCDingXin/CCDM) (`iCCDM` folder) if
you need it later:

| Removed | Why |
|---|---|
| `config/RC64/`, `config/SA64/`, `config/SA128/`, `config/SA256/`, `config/UK64/`, `config/UK128/`, `config/UK192/`, `config/UK256/`, `config/Cell/` | Training configs for the other four datasets. |
| `evaluation/eval_models/RC49/`, `Cell200/`, `SteeringAngle/`, `UTKFace/` | Eval-network *architecture* code matched to those datasets' pretrained checkpoints. Use as an architecture template (pulled from upstream) when building DR's own eval nets — see below. |
| `config/model_cfg/*_192_v1.yaml`, `dit_b_4_192.yaml` | 192px model configs. 64px/128px/256px kept for all three backbones (UNet-EDM, UNet-CCDM, DiT). |
| `DiffAugment_pytorch.py` | Dead code — not imported anywhere in the repo. |
| `models/sngan.py` | A GAN architecture with no corresponding GAN training loop anywhere in `trainer.py` — dead weight for a diffusion-only pipeline. Its import was also removed from `models/__init__.py`. |

## The eval-checkpoint gap (read this before you run `--do_eval`)

CCDM's built-in `--do_eval` (SFID, label score, NIQE) depends on
**dataset-specific pretrained networks** — an autoencoder for the SFID
feature space, plus a ResNet34 classifier/regressor for the label score —
that ship pretrained for UTKFace/RC-49/SteeringAngle/Cell-200, but not for
a dataset that didn't exist when the repo was released. `evaluator.py`
looks for these at `./evaluation/eval_ckpts/DRGrading/metrics_{size}x{size}/...`,
but that directory is empty until you train those three networks yourself,
following the recipe in the acknowledged upstream repo,
[CcGAN-AVAR](https://github.com/UBCDingXin/CcGAN-AVAR) (same authors, same
checkpoint format).

Until then, **leave `--do_eval` off** (all three DR configs do this by default)
and use `downstream_eval/` instead. Arguably that's the more convincing
evaluation for a DR contribution anyway: SFID tells you the generated
images are distributionally plausible, but what actually matters is
whether the synthetic data improves a real DR classifier — which is what
`train_dr_classifier.py` measures directly.

## Sampling from a trained checkpoint (`generate_from_ckpt.py`)

`generate_from_ckpt.py` generates grade-conditioned images from a finished
run **without re-training and without loading the full training set** —
useful for squeezing more samples out of an existing model or for
generating on a machine that doesn't have the h5 data.

It needs three things (none of which live inside `model-{step}.pt`):

- the diffusion checkpoint: `output/DRGrading_{size}/setup1_dr/results/model-{step}.pt`
- the label-embedding nets, found by default at
  `{root_path}/output/DRGrading_{size}/model_y2h/ckpt_mlp_y2h_epoch_500.pth` and
  `{root_path}/output/DRGrading_{size}/model_y2cov/ckpt_cnn_y2cov_epoch_500.pth`
  (override with `--path_y2h` / `--path_y2cov` / `--y2h_ckpt_name` / `--y2cov_ckpt_name`)
- the exact `config/model_cfg/*.yaml` used for training

```bash
python generate_from_ckpt.py \
    --model_ckpt output/DRGrading_128/setup1_dr/results/model-100000.pt \
    --model_config config/model_cfg/unet_edm_128_v1.yaml \
    --root_path . --image_size 128 \
    --grades 0 1 2 3 4 --nfake_per_grade 1000 --cond_scale 4
```

All CLI defaults match `config/DR128/run_train.sh`; only `--grades`,
`--nfake_per_grade`, and `--cond_scale` normally need changing. `--out_dir`
defaults to `output/generated_cs{cond_scale}` (pass `--out_dir` to override),
and the output h5 stores its generation attrs (`cond_scale`, `model_ckpt`,
`sampler`, ...) so a generated set is self-describing. Output is
`{out_dir}/generated.h5` (same schema as the training h5, labels as raw
grades 0-4) plus `sample_grade_{g}.png` preview grids. `--sampler` can be
`sde` (default, best quality), `ode`, or `dpmpp` (fastest).

The canonical frozen pipeline picks different CFG strengths per grade
(grades 0/1/4 at cond_scale 4, grades 2/3 at 1.5, grade 4 real-only), assembled
with `analysis/merge_h5_by_grade.py` into a blend h5 — see
`docs/DOWNSTREAM_RESULTS.md` for the exact recipe and results.

## Directory map

```
CCDM-DR/
├── dataset.py                     # modified: + DRGrading branch (expects _train.h5)
├── opts.py                        # modified: + DRGrading choice
├── main.py                        # modified: eval-model import gated behind --do_eval (bug fix)
├── trainer.py                     # modified: visual sample batch size, robustness/device fixes
├── diffusion.py                   # unmodified
├── label_embedding.py, utils.py   # lightly modified (CPU fallback)
├── generate_from_ckpt.py          # NEW: sample a trained checkpoint without re-training
├── models/
│   ├── unet_edm.py, unet_ccdm.py, dit.py    # three usable backbones (unmodified)
│   ├── resnet_y2h.py, resnet_y2cov.py, resnet_aux_regre.py, attend.py  # unmodified
│   └── __init__.py                # modified: dropped the sngan import (file removed)
├── config/
│   ├── DR64/run_train.sh          # fast-iteration 64x64 debug config
│   ├── DR128/run_train.sh         # main 128x128 config
│   ├── DR256/run_train.sh         # high-res 256x256 config
│   └── model_cfg/                 # 64/128/256px x {unet_edm, unet_ccdm, dit}
├── evaluation/
│   ├── evaluator.py               # modified: + DRGrading branch (checkpoint gap above)
│   ├── eval_metrics.py            # FID/IS/MMD/label-score implementations (unmodified)
│   └── eval_models/                # empty except __init__.py -- see "What was removed"
├── data_preparation/
│   ├── get_dataset.py              # download + normalize APTOS/IDRiD/DDR/Messidor-2
│   ├── build_dr_h5.py             # image folder + CSV -> h5
│   └── download_h5.py             # download pre-built h5 from Google Drive
├── downstream_eval/
│   ├── train_dr_classifier.py     # the real vs real+synthetic experiment
│   └── compare_runs.py
├── analysis/
│   ├── trace_conditioning.py, render_inspection_montages.py, run_diagnosis.sh
│   ├── check_embedding.py, sweep_cond_scale.py   # E1/E2 conditioning probes
│   ├── merge_h5_by_grade.py, tag_h5.py, seed_report.py
│   └── README.md                                 # diagnostics usage + thresholds
├── notebooks/
│   └── dataset_prepare.ipynb      # end-to-end data-prep walkthrough (Colab-friendly)
├── docs/
│   ├── DOWNSTREAM_RESULTS.md      # frozen 5-seed downstream experiment (tracked)
│   └── full_pipeline.md           # start-to-finish walkthrough
├── .env.h5_links                  # Google Drive file IDs (committed)
├── requirements.txt
├── AGENTS.md                      # command/gotcha reference for coding agents
└── LICENSE
```

Note: `docs/` is tracked (`DOWNSTREAM_RESULTS.md` + `full_pipeline.md`). Private
working notes (e.g. `docs_private/RETRAIN_GUIDE.md`, a retrain-from-scratch
memo for the owner) live in `docs_private/`, which is gitignored.

## Citation

If you use the underlying CCDM/iCCDM method, cite the original papers (see
[UBCDingXin/CCDM](https://github.com/UBCDingXin/CCDM) for the current
BibTeX entries).
