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

# 1. Download and prepare a dataset (APTOS, IDRiD, DDR, or Messidor-2)
python data_preparation/get_dataset.py --dataset Aptos

# 2. Build the h5 dataset from the prepared output
python data_preparation/build_dr_h5.py \
    --image_dir ./data/Aptos/Images \
    --csv_path  ./data/Aptos/labels.csv \
    --out_dir   ./data/DRGrading \
    --img_size  128

# 3. Train (export ROOT_PATH and DATA_PATH first)
export ROOT_PATH=/path/to/CCDM-DR
export DATA_PATH=/path/to/DRGrading   # dir containing DRGrading_*.h5
bash config/DR128/run_train.sh          # main 128x128 config
bash config/DR64/run_train.sh           # fast-iteration 64x64 debug config

# 4. Evaluate: does the synthetic data actually help a DR classifier?
python downstream_eval/train_dr_classifier.py \
    --real_h5 /path/DRGrading_128x128.h5 \
    --test_h5 /path/DRGrading_128x128_test.h5 \
    --backbone resnet50 --epochs 30 --run_name real_only

python downstream_eval/train_dr_classifier.py \
    --real_h5 /path/DRGrading_128x128.h5 \
    --test_h5 /path/DRGrading_128x128_test.h5 \
    --synthetic_h5 /path/to/generated.h5 --synthetic_cap_per_grade 1500 \
    --backbone resnet50 --epochs 30 --run_name real_plus_synthetic

python downstream_eval/compare_runs.py --results_dir ./downstream_results
```

See `AGENTS.md` for a denser command/gotcha reference, and the "eval-checkpoint
gap" section below before you touch `--do_eval`.

## What was changed vs. upstream

| File | Change |
|---|---|
| `dataset.py` | Added a `DRGrading` branch to `LoadDataSet`. DR severity grades are discrete integers 0-4 (ICDR scale), so this reuses the same loading/minority-replication logic already written for `UTKFace` (also a small integer label set) rather than the continuous-label logic used for `SteeringAngle`/`RC-49`. |
| `opts.py` | Added `"DRGrading"` to the `--data_name` choices. |
| `evaluation/evaluator.py` | Added a `DRGrading` branch pointing to `./evaluation/eval_ckpts/DRGrading/...`. **These checkpoints don't exist yet** — see "The eval-checkpoint gap" below. |
| `main.py` | **Bug fix.** Upstream unconditionally imports `evaluation/eval_models/{data_name}/metrics_{size}x{size}` right after training, regardless of `--do_eval`. That path doesn't exist for `DRGrading`, so every DR run would otherwise crash right after sampling — after the GPU time for training was already spent. This is now gated behind `if args.do_eval:` (a no-op for the original datasets). |
| `data_preparation/get_dataset.py` | **New.** Downloads and normalizes APTOS/IDRiD/DDR/Messidor-2 into a common `{dataset}/Images/` + `labels.csv` structure. |
| `data_preparation/build_dr_h5.py` | **New.** Converts a fundus image folder + CSV of grades into the h5 format `dataset.py` expects, with fundus-specific preprocessing (circular field-of-view crop). |
| `config/DR128/run_train.sh`, `config/DR64/run_train.sh` | **New.** Training configs for DR. |
| `downstream_eval/train_dr_classifier.py` | **New.** Trains a DR grading classifier (ResNet50/EfficientNet-B4) under real-only vs. real+synthetic conditions and reports accuracy/macro-F1/QWK. This is the primary evidence for the contribution. |
| `downstream_eval/compare_runs.py` | **New.** Summarizes multiple `train_dr_classifier.py` runs into one comparison table. |

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

Until then, **leave `--do_eval` off** (both DR configs do this by default)
and use `downstream_eval/` instead. Arguably that's the more convincing
evaluation for a DR contribution anyway: SFID tells you the generated
images are distributionally plausible, but what actually matters is
whether the synthetic data improves a real DR classifier — which is what
`train_dr_classifier.py` measures directly.

## Directory map

```
CCDM-DR/
├── dataset.py                     # modified: + DRGrading branch
├── opts.py                        # modified: + DRGrading choice
├── main.py                        # modified: eval-model import gated behind --do_eval (bug fix)
├── trainer.py, diffusion.py, label_embedding.py, utils.py   # unmodified
├── models/
│   ├── unet_edm.py, unet_ccdm.py, dit.py    # three usable backbones (unmodified)
│   ├── resnet_y2h.py, resnet_y2cov.py, resnet_aux_regre.py, attend.py  # unmodified
│   └── __init__.py                # modified: dropped the sngan import (file removed)
├── config/
│   ├── DR64/run_train.sh          # fast-iteration 64x64 config
│   ├── DR128/run_train.sh         # main 128x128 config
│   └── model_cfg/                 # 64/128/256px x {unet_edm, unet_ccdm, dit}
├── evaluation/
│   ├── evaluator.py               # modified: + DRGrading branch (checkpoint gap above)
│   └── eval_models/                # empty except __init__.py -- see "What was removed"
├── data_preparation/
│   ├── get_dataset.py              # download + normalize APTOS/IDRiD/DDR/Messidor-2
│   └── build_dr_h5.py             # image folder + CSV -> h5
├── downstream_eval/
│   ├── train_dr_classifier.py     # the real vs real+synthetic experiment
│   └── compare_runs.py
├── requirements.txt
├── AGENTS.md                      # command/gotcha reference for coding agents
└── LICENSE
```

## Citation

If you use the underlying CCDM/iCCDM method, cite the original papers (see
[UBCDingXin/CCDM](https://github.com/UBCDingXin/CCDM) for the current
BibTeX entries).