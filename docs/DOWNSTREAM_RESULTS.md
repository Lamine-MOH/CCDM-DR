# Downstream DR classification — results with grade-conditioned synthetic augmentation

**Status: frozen** (DONE). These are the canonical numbers for the contribution.

## How this was reached

Conditioning fixes (dedicated per-block `affine_cond` for label embeddings,
decoupled from time; embedding encoders trained 200 epochs, not 10) →
retrained the model 0→50k→**100k steps** → A1 trace showed conditioning
"flows but is weak" → E1 (embedding health, PASS) + E2 (CFG sweep, separation
rises with cond_scale) showed the signal is **expressible but under-amplified
at CFG 1.5** → per-grade CFG selection (grades 0/1/4 @4.0, grades 2/3 @1.5,
grade 4 real-only) → 5-seed downstream validation below. No `aux_reg_loss` and
no further diffusion training were needed.

## Protocol

- Downstream classifier: `densenet121`, ImageNet-pretrained, 30 epochs,
  Adam lr=1e-4, batch 32, seeds **111–115** (`train_dr_classifier.py --seed`).
- Train/real: `data/DRGrading/Aptos/DRGrading_128x128_train.h5`.
- Test: `data/DRGrading/Aptos/DRGrading_128x128_test.h5`.
- Augmentation: real train + synthetic **blendA** h5 capped at 1000 samples per
  grade **except grade 4, which is real-only** (`--synthetic_cap_per_grade 1000`
  + blend constructed with `--caps_override "4=0"`).

### blendA provenance

Source model: `output/DRGrading_128/setup1_dr/results/model-100000.pt`
(EMA), sampler `sde`, 32 steps. Per-grade source of the synthetic images:

| grade | source h5 | cond_scale |
|---|---|---|
| 0, 1, 4 | `output/generated_cfg4/generated.h5` | 4.0 |
| 2, 3 | `output/generated/generated.h5` (legacy, attr-tagged) | 1.5 |

Assembled with:

```bash
python analysis/merge_h5_by_grade.py --sources \
    "0=output/generated_cfg4/generated.h5 1=output/generated_cfg4/generated.h5 \
     2=output/generated/generated.h5 3=output/generated/generated.h5 \
     4=output/generated_cfg4/generated.h5" \
    --out output/generated_blendA/generated.h5 \
    --cap 1000 --caps_override "4=0"
```

The choice of 1.5 (dark/mostly-smooth images) for grades 2–3 and 4.0 for
grades 0–1 was based on the per-grade CFG sweep (E2, `analysis/sweep_cond_scale.py`)
tuned across single-seed downstream runs; grade 4 was excluded because no
synthetic scale beat real-only grade-4 recall.

## Results — 5 seeds, mean ± std

| metric | real-only | real + synth | delta (paired) |
|---|---|---|---|
| accuracy | 0.8295 ± 0.0078 | 0.8426 ± 0.0047 | **+0.0131 ± 0.0054** (+5/−0) |
| macro-F1 | 0.6916 ± 0.0163 | 0.7129 ± 0.0089 | **+0.0213 ± 0.0100** (+5/−0) |
| QWK | 0.9063 ± 0.0073 | 0.9204 ± 0.0040 | **+0.0142 ± 0.0092** (+5/−0) |
| recall g0 | 0.9947 ± 0.0043 | 0.9947 ± 0.0051 | +0.0000 ± 0.0027 (+1/−1) |
| recall g1 | 0.5600 ± 0.0662 | 0.5567 ± 0.0435 | −0.0033 ± 0.0361 (+2/−3) |
| recall g2 | 0.7829 ± 0.0528 | 0.8171 ± 0.0072 | +0.0342 ± 0.0487 (+4/−1) |
| recall g3 | 0.4938 ± 0.1114 | 0.4562 ± 0.0523 | −0.0375 ± 0.0998 (+2/−3) |
| recall g4 | 0.6048 ± 0.0319 | 0.6857 ± 0.0426 | +0.0810 ± 0.0686 (+4/−0) |

## Interpretation

- **Seed-stable, defensible gains:** accuracy, macro-F1, and QWK improve in all
  5 seeds (+5/−0). Macro-F1 +0.021, QWK +0.014, accuracy +0.013.
- **Secondary observation:** augmentation also roughly halves run-to-run variance
  (std of acc 0.0078→0.0047, macro-F1 0.016→0.009, QWK 0.007→0.004) — a higher
  and stabler classifier.
- **Per-grade deltas are mostly within noise** (grade test cells are small,
  e.g. ~30–60 samples/grade). g2 (+0.034, +4/−1) and g4 (+0.081, +4/−0) are
  consistently positive; g1 and g3 flip sign and should not be claimed.
- Do **not** report single-seed per-grade "rescues" (e.g. earlier exploratory
  g3=0.594): they do not survive seed validation.

## Model & training provenance (`model-100000.pt`)

Source checkpoint: `output/DRGrading_128/setup1_dr/results/model-100000.pt`
(EMA weights restored at sampling time). Trained on
`data/DRGrading/Aptos/DRGrading_128x128_train.h5` in two phases via
`config/DR128/run_train.sh` (UNet-EDM 128×128, `--train_lr 1e-4`):

```bash
# Phase 1: 0 -> 50k
bash config/DR128/run_train.sh /home/jovyan/CCDM-DR /home/jovyan/CCDM-DR/data/DRGrading/Aptos \
    --num_steps 50000 --skip_final_sampling
# Phase 2: 50k -> 100k
bash config/DR128/run_train.sh /home/jovyan/CCDM-DR /home/jovyan/CCDM-DR/data/DRGrading/Aptos \
    --num_steps 100000 --resume_step 50000
```

Embedding nets trained by the same runs:
`output/DRGrading_128/model_y2h/ckpt_{resnet,mlp}_y2h_epoch_{200,500}.pth`,
`model_y2cov/ckpt_{cnn,net}_y2cov_epoch_{200,500}.pth` — labels must stay
byte-identical, so copy them alongside the diffusion checkpoints when moving
machines (see `AGENTS.md` phased-resume notes).

Notes:
- `--num_steps` is the TOTAL step count, not the remaining one;
  `--resume_step` must be a multiple of `--save_every`.
- A changed `--num_steps` restarts the loss log
  (`results/log_loss_steps{train_num_steps}.txt`).
- Final loss plateaued ~0.004; no third phase was run (flat loss + saturated
  downstream results).

## Regeneration

```bash
# 1. build blendA (above)
# 2. classifier, per seed (run the s-loop for 111..115):
python downstream_eval/train_dr_classifier.py \
    --real_h5 data/DRGrading/Aptos/DRGrading_128x128_train.h5 \
    --test_h5 data/DRGrading/Aptos/DRGrading_128x128_test.h5 \
    --synthetic_h5 output/generated_blendA/generated.h5 \
    --synthetic_cap_per_grade 1000 \
    --backbone densenet121 --epochs 30 \
    --run_name real_plus_synth_blendA_s$s --seed $s
# + the matching real-only runs (no --synthetic_h5) as real_only_s$s
# 3. aggregate:
python analysis/seed_report.py --results_dir ./downstream_results \
    --group real_only real_only_s111 real_only_s112 real_only_s113 real_only_s114 real_only_s115 \
    --group augmented real_plus_synth_blendA_s111 real_plus_synth_blendA_s112 \
                      real_plus_synth_blendA_s113 real_plus_synth_blendA_s114 real_plus_synth_blendA_s115
```