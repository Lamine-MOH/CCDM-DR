# analysis/ — post-training diagnosis for grade-conditioning

These scripts answer one question after a retrain: **did the label
conditioning actually learn this time?** (Pre-fix baseline: RF grade
separability flat at 0.17-0.44 across checkpoints, synthetic-only classifier
QWK ceiling ~0.80 with grade-2 recall 0.20, and/or visually incoherent
lesions.)

Quick start (see `docs/RETRAIN_GUIDE.md` §5 for thresholds):

```bash
bash analysis/run_diagnosis.sh ROOT_PATH DATA_PATH --synth_h5 output/generated/generated.h5
```

When the A1 trace stays at chance (conditioning "flows but is weak"), two
cheap pre-retrain probes isolate WHERE conditioning is lost:

```bash
# E1 — is the label embedding net healthy? (CPU, minutes)
python analysis/check_embedding.py --y2h_dir output/DRGrading_128/model_y2h

# E2 — can higher CFG (cond_scale) surface grade separation? (GPU, ~15 min)
python analysis/sweep_cond_scale.py \
    --model_ckpt output/DRGrading_128/setup1_dr/results/model-100000.pt \
    --model_config config/model_cfg/unet_edm_128_v1.yaml \
    --root_path . --image_size 128 \
    --out_dir output/sweep_cond_scale --grades 0 4 \
    --nfake 64 --cond_scales 1.5 3 6
```

- E1 PASS + E2 separation rises with cond_scale → no retrain; sample at higher CFG.
- E2 flat at all scales → enable `aux_reg_loss` (train a DR `resnet18` aux net
  on real data normalized to [-1,1], drop at
  `output/DRGrading_128/aux_reg_model/ckpt_resnet18_epoch_200.pth`, then
  resume with `--use_aux_reg_loss`) and bump `--sample_cond_scale` to 3-4.

## A1 — `trace_conditioning.py` (teaching-signal tracer)

Splits every `results/sample_{ode,sde}_<step>.png` grid into its 10×10 cells
(cell = trained image size, stride = cell+1 to undo `save_image(padding=1)`),
keeps the 7-feature schema of the original trace
(`brightness, R, G, B, std, RminusG, edge_mag`), assigns each row raw grade
`r*max_label/(n_rows-1)` (i.e. 0.0, 0.444, …, 4.0 → rounded integer for the
5-class metric), and reports stratified RF-CV accuracy per checkpoint.

Because the real-train reference is recomputed from the **same script/features**
(`--real_h5`), numbers are self-consistent across runs on any machine.

Outputs into `--out_dir`:
`per_step_summary.csv`, `per_row_features.csv`, `per_image_features/{ode,sde}_<step>.csv`,
`trace_plot.png`, `trace_report.md` (with PASS/IMPROVED/FAIL verdict).

## A2 — synthetic-only classifier (optional, slow)

`run_diagnosis.sh --classifier` calls
`downstream_eval/train_dr_classifier.py` with the **synthetic** h5 as
`--real_h5` and the real test h5 as `--test_h5` (no `--synthetic_h5`). A
capable grade-conditional generator reaches QWK > 0.85 and grade-2 recall
≥ 0.5; the broken one peaked at ~0.80 / 0.20.

## A3 — `render_inspection_montages.py` (human review)

Per grade 0-4: 2×12 montage with real images on top, synthetic on the bottom,
plus an all-grades combined image and `README_viewing_guide.md` with a
checklist (are MAs/hemorrhages/exudates present, does severity rise with
grade, are lesions anatomically plausible, does background look over-smooth?).