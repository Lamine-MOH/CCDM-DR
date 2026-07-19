# CCDM-DR — a diabetic retinopathy fork of UBCDingXin/CCDM (iCCDM)

This is a fork of the `iCCDM` folder from [UBCDingXin/CCDM](https://github.com/UBCDingXin/CCDM)
(EDM-based continuous conditional diffusion model), adapted to generate
grade-conditioned diabetic retinopathy (DR) fundus images. Nothing about the
original datasets (RC-49, UTKFace, Cell-200, SteeringAngle) was removed —
`DRGrading` is added as a fifth `--data_name` option alongside them, so the
original configs still work unmodified.

## What was changed vs. upstream

| File | Change |
|---|---|
| `dataset.py` | Added a `DRGrading` branch to `LoadDataSet`. DR severity grades are discrete integers 0-4 (ICDR scale), so this reuses the same loading/minority-replication logic already written for `UTKFace` (also small integer label set) rather than the continuous-label logic used for `SteeringAngle`/`RC-49`. |
| `opts.py` | Added `"DRGrading"` to the `--data_name` choices. |
| `evaluation/evaluator.py` | Added a `DRGrading` branch pointing to `./evaluation/eval_ckpts/DRGrading/...`. **These checkpoints don't exist yet** — see "The eval-checkpoint gap" below. |
| `data_preparation/build_dr_h5.py` | **New.** Converts a fundus image folder + CSV of grades into the h5 format `dataset.py` expects, with fundus-specific preprocessing (circular field-of-view crop). |
| `config/DR128/run_train.sh`, `config/DR64/run_train.sh` | **New.** Training configs for DR, following the same pattern as `config/UK128`, `config/SA64`, etc. |
| `downstream_eval/train_dr_classifier.py` | **New.** Trains a DR grading classifier (ResNet50/EfficientNet-B4) under real-only vs. real+synthetic conditions and reports accuracy/macro-F1/QWK. This is the primary evidence for the contribution — see the experiment guide. |
| `downstream_eval/compare_runs.py` | **New.** Summarizes multiple `train_dr_classifier.py` runs into one comparison table. |

## The eval-checkpoint gap (read this before you run `--do_eval`)

CCDM's built-in `--do_eval` (SFID, label score, NIQE) depends on **dataset-specific
pretrained networks** — an autoencoder for the SFID feature space, plus a
ResNet34 classifier/regressor for the label score — that ship pretrained for
UTKFace/RC-49/SteeringAngle/Cell-200, but obviously not for a dataset that
didn't exist when the repo was released. `evaluator.py` now looks for these
at `./evaluation/eval_ckpts/DRGrading/metrics_{size}x{size}/...`, but that
directory is empty until you train those three networks yourself, following
the recipe in the acknowledged upstream repo,
[CcGAN-AVAR](https://github.com/UBCDingXin/CcGAN-AVAR) (same authors, same
checkpoint format).

Until you've done that, **leave `--do_eval` off** (both DR configs here do
this by default) and use `downstream_eval/` instead. Arguably this is the
more convincing evaluation for a DR contribution anyway: SFID tells you the
generated images are distributionally plausible, but a reviewer will care
much more about whether the synthetic data actually improves a real DR
classifier — which is what `train_dr_classifier.py` measures directly.

## Directory map

```
CCDM-DR/
├── dataset.py                     # modified: + DRGrading branch
├── opts.py                        # modified: + DRGrading choice
├── main.py, trainer.py, diffusion.py, label_embedding.py, utils.py   # unmodified
├── models/                        # unmodified (unet_edm.py is what config/DR* uses)
├── config/
│   ├── DR64/run_train.sh          # new: fast-iteration 64x64 config
│   ├── DR128/run_train.sh         # new: main 128x128 config
│   └── ... (RC64, SA64, UK64, etc. unmodified)
├── evaluation/
│   └── evaluator.py               # modified: + DRGrading branch (checkpoint gap noted above)
├── data_preparation/
│   ├── build_dr_h5.py             # new: image folder + CSV -> h5
│   └── requirements_dr.txt        # new
└── downstream_eval/
    ├── train_dr_classifier.py     # new: the real vs real+synthetic experiment
    └── compare_runs.py            # new
```

See `DR_CCDM_Experiment_Guide.md` (shipped alongside this code, not inside
the repo) for the full step-by-step experimental plan.
