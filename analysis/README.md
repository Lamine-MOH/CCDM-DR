# analysis/ — post-training diagnosis for grade-conditioning

These scripts answer one question after a retrain: **did the label
conditioning actually learn this time?** (Pre-fix baseline: RF grade
separability flat at 0.17-0.44 across checkpoints, synthetic-only classifier
QWK ceiling ~0.80 with grade-2 recall 0.20, and/or visually incoherent
lesions.)

```bash
bash analysis/run_diagnosis.sh ROOT_PATH DATA_PATH --synth_h5 output/generated_cs4/generated.h5
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

### `merge_h5_by_grade.py` — per-grade blend of per-scale h5 files

When different grades prefer different CFG strengths (per-class downstream
recall), assemble one h5 where each grade comes from a different generated set:

```bash
python analysis/merge_h5_by_grade.py --sources \
    "0=output/generated_cfg4/generated.h5 1=output/generated_cfg4/generated.h5 \
     2=output/generated_cs1.5/generated.h5 3=output/generated_cs1.5/generated.h5 \
     4=output/generated_cfg4/generated.h5" \
    --out output/generated_blendA/generated.h5 --cap 1000 --caps_override "4=0"
```

Optional per-grade cap override: `--caps_override "4=500"`. Output uses the same
`images`(uint8 CHW)/`labels`(float64) schema, ready for `--synthetic_h5`.
Legacy h5 files (written before generate_from_ckpt.py stored attrs) print
`cond_scale=?`; stamp their provenance with `analysis/tag_h5.py --h5 <path>
--cond_scale <cs> --max_label 4` so blends stay self-describing.

### `seed_report.py` — seed-robust metric aggregation

Groups `*_metrics.json` by protocol and prints **mean ± std** per metric plus
index-aligned paired deltas (with same-sign counts), so the seed-sensitivity of
any downstream gain is explicit. Runs are passed as explicit run_names:

```bash
python analysis/seed_report.py --results_dir ./downstream_results \
    --group real_only real_only_s111 real_only_s112 real_only_s113 \
    --group augmented real_plus_synth_blendA_s111 real_plus_synth_blendA_s112 \
                      real_plus_synth_blendA_s113
```

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

## `run_downstream_protocol.sh` — frozen-protocol runner for a 2nd backbone

Re-runs the frozen downstream protocol (real-only + blendA-augmented, seeds
111-115) for a different classifier backbone as a robustness check — see
`docs/DOWNSTREAM_RESULTS.md`. Hyperparameters are locked to the frozen ones;
only `--backbone` and the run-name prefix differ so the original densenet121
rows are never overwritten or mixed:

```bash
bash analysis/run_downstream_protocol.sh resnet50 r50 \
    data/DRGrading/Aptos/DRGrading_128x128_train.h5 \
    data/DRGrading/Aptos/DRGrading_128x128_test.h5
```

Runs sequentially (real then augmented per seed); each run is ~20-40 min GPU.
Writes `{OUT_DIR}/real_only_r50_s{111..115}_metrics.json` and
`real_plus_synth_blendA_r50_s{111..115}_metrics.json`, then prints the
`seed_report.py` command to aggregate them.