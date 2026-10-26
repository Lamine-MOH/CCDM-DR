# analysis/ — post-training diagnosis for grade-conditioning

> Classifier-checkpoint paths below point into `downstream_results/`, which is
> **local-only** (gitignored, never committed). **Current evidence:** the Exp-4
> matrix (2026-09-16, post-Tier-3, val-based multi-backbone, 80 runs) is canonical
> — the results live in local `downstream_results/Exp 4` and are produced with
> `run_matrix.sh` below. The older scripts in this file (diagnosis / embedding
> probes / blend) still run as-is on local outputs; `run_downstream_protocol.sh` +
> `seed_report.sh` belong to the **historical** 5-seed protocol (best-epoch-on-test,
> pre-fix).

## Orchestrator: `run_matrix.sh` + `select_blend_cs.py`

The post-retrain evidence is produced end-to-end by these two (results land in
`downstream_results/Exp 4`):

- **`run_matrix.sh ROOT DATA --model_ckpt <path>`** — required direct checkpoint
  path (Tier-3 `model-100000.pt`); stages: generation (`--gen`) → cond_scale
  sweep per grade → `blend()` arms → per-arm classifier training (5 seeds
  densenet/resnet/eff, val QWK selection) → `seed_report`. Idempotent
  (skips existing runs/results on the local box and TM); run with `--dry-run` to
  inspect without launching. Blend arms include the frozen `blendA`, the
  data-driven `blend_sel`, and the promoted headline probe `blendA_g3cs4`.
  **Canonical evidence = `downstream_results/Exp 4`** (80-run full matrix;
  overlapping arms reproduce the earlier Exp 3 exactly).
- **`select_blend_cs.py`** — reads the sweep's per-grade per-cs VAL recall
  (mean over seeds), picks each grade's best cs (argmax), and assembles
  `blend_sel.h5`. Do not treat its output as canonical: per-grade val recall
  anti-transfers to test (g3/g4 Pearson < 0), so the frozen `blendA` recipe is
  the headline arm and per-grade selection is a documented limitation.
  `--arm_prefix` (default `uni`) sets the sweep arm-name prefix the runs are
  read from — use `uni_sde` / `uni_ode` for a per-sampler sweep (factorial mode).

### Factorial mode: `--factorial` (Exp 5, sampler × cond_scale)

A clean factorial that shares **nothing** with Exp 3/4 except the checkpoint:
every pool is regenerated and every classifier run is re-trained. Replaces the
sweep + blend stages with one uniform augmentation arm per (sampler, cond_scale)
cell — real train + that pool capped at `--cap` per grade, **all five grades
synthetic** (a constant handicap across cells, so the contrasts stay valid).

| flag | meaning |
|---|---|
| `--factorial` | enable the mode (blends are skipped; reporting switches to `sampler_cs_report.py`) |
| `--sampler S` | repeatable; e.g. `--sampler sde --sampler ode` |
| `--cses "1.0 2.0 3.0 4.0"` | cond_scale grid. **`1.0` is meaningful**: `unet_edm.forward_with_cond_scale` returns early at `cond_scale == 1`, i.e. guidance *off* — one UNet eval per Heun step instead of two, so that pool generates ~2× faster |
| `--cap N` | per-grade synthetic cap (default 1000) |
| `--results_subdir NAME` | nest results under `downstream_results/NAME` (keeps Exp 3/4 files untouched) |
| `--gen-seed N` / `--reseed-per-grade` | generation RNG. `--reseed-per-grade` seeds once per grade (`seed + grade`) so two samplers start each grade from the same init noise; **off by default** — without it the RNG carries across grades, as it always has, and every pre-existing pool stays byte-reproducible |

Arm names are `uni_{sampler}_cs{cond_scale}` (e.g. `densenet121_uni_sde_cs1.0_s111`),
plus `{backbone}_real_only_s{seed}`. Pools land in `output/generated_{sampler}_cs{cs}/`
— in factorial mode the legacy `output/generated_cs{cs}/` layout is *not* reused,
so the old pools are never regenerated or overwritten.

Intended sequential use, one backbone per invocation so densenet121's full 2×K
grid lands first (every stage skips what already exists, so all three are
resumable and re-running is cheap):

```bash
COMMON="--model_ckpt <ROOT>/output/DRGrading_128/setup1_dr/results/model-100000.pt \
        --factorial --sampler sde --sampler ode --cses 1.0 2.0 3.0 4.0 \
        --results_subdir Exp5_sampler_cs"
bash analysis/run_matrix.sh <ROOT> <DATA> $COMMON --headline-backbones densenet121
bash analysis/run_matrix.sh <ROOT> <DATA> $COMMON --headline-backbones resnet50
bash analysis/run_matrix.sh <ROOT> <DATA> $COMMON --headline-backbones efficientnet_b4
```

That is 8 generation passes + 135 classifier runs (3 backbones × [5 `real_only` +
8 arms × 5 seeds]). Use `--dry-run` to inspect the exact command list first.

### `sampler_cs_report.py` — the 2×K grid reporter

```bash
python analysis/sampler_cs_report.py \
    --results_dir downstream_results/Exp5_sampler_cs \
    --backbones "densenet121 resnet50 efficientnet_b4" \
    --seeds "111 112 113 114 115" --samplers "sde ode" --cses "1.0 2.0 3.0 4.0" \
    --out_csv downstream_results/Exp5_sampler_cs/sampler_cs_grid.csv
```

Per backbone it prints, for `--sides "test val"` (default both):

1. the **cell grid** (sampler × cond_scale) of mean±std over seeds for
   qwk / accuracy / macro-F1 / per-grade recall, with `real_only` alongside the
   recall grids;
2. the **paired delta vs `real_only`**, index-aligned by seed, with +n/−n counts;
3. the **head-to-head sampler contrast** at matched cond_scale (`sde - ode`,
   paired by seed) — the direct A/B;
4. **main effects**: each sampler averaged over cond_scales, and each cond_scale
   averaged over samplers.

It refuses to silently average over a broken design: missing/partial cells are
listed with their absent seeds, and it warns if `selection_on`, `val_seed`,
`val_frac` or `pretrained` differ anywhere in the grid. The val side reports only
what the metrics JSON actually stores (`val_qwk` + `val_per_class_report`);
val accuracy/macro-F1 print `--` rather than falling back to the test value, so
read decisions off the val rows. `--out_csv` writes every number long-format
(`kind` = `cell` / `delta_vs_real_only` / `sampler_contrast` / `main_effect_*`).

### Merge helper `merge_h5_by_grade.py` — per-grade blend of per-scale h5 files

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

If a source file carries a float64 `rank` dataset (written by
`filter_synthetic.py`), the per-grade cap is filled with the **top-k by rank**
instead of a random subsample — so a filtered pool's most-confident survivors
are the ones that land in the blend.

## Batch A — post-hoc synthetic quality & classifier upgrades (no generator retrain)

Three scripts implement the literature-driven experiments A1/A2/C9 of the
Batch-A review.

### `filter_synthetic.py` — semantic filter of the generated pool (A1)

MICCAI 2025 (ECC_DM_for_DR) idea on our pool: score every generated image with
trained real-data classifiers; keep samples whose **max-likelihood member**
predicts the target grade, ranked by that likelihood; optional dHash
near-duplicate removal:

```bash
python analysis/filter_synthetic.py \
    --sources output/generated_cfg4/generated.h5 output/generated_cs1.5/generated.h5 \
    --ckpts densenet121=downstream_results/real_only_s112_best.pth \
            resnet50=downstream_results/real_only_r50_s111_best.pth \
    --out_dir output/filtered --dedup_hamming 0
```

Output names carry the SOURCE DIRECTORY as the scale tag: `generated_cfg4_filtered.h5`,
`generated_cs1.5_filtered.h5`, `{scale}_scores.npz`, and `filter_pass_rate.csv`.
Use any real-only trained checkpoint of each backbone (e.g. densenet121 s112-115).

Writes per source `{scale}_filtered.h5` (with `rank`), `{scale}_scores.npz`, and
`filter_pass_rate.csv` — the per-grade pass-rate table ("how many synthetics
survive an ensemble of real-trained graders"). Feed the filtered h5 into
`merge_h5_by_grade.py` (rank-aware now) to build a filtered blend.

**Result (A1, negative):** at matched quantity a filtered blend did NOT beat a
random draw from the unfiltered pools on the 5-seed densenet protocol (filtered
≈ real_only; random > filtered on acc/macro-F1/QWK) — the blendA gain is
volume+diversity, not label purity. (Superseded historical result; the full
table and interpretation are reported in the paper.)

Note on `--dedup_hamming` (default 6): the 8x8-block dHash over-collides on
low-texture classes (healthy/g0 synthetics drop ~84% as "near-duplicates"), so
use `--dedup_hamming 0` for the primary blend and treat hamming>0 as a
sensitivity check only.

Needs the classifier **scoring mode** (scores any h5 with a trained checkpoint):
```bash
python downstream_eval/train_dr_classifier.py \
    --score_h5 output/generated_cfg4/generated.h5 \
    --ckpt downstream_results/real_only_s112_best.pth \
    --backbone densenet121 --run_name score_cfg4
```
(no training; writes `{score_h5base}_scores_{run_name}.csv`, plus `_probs_*.npz`
with `--score_save_probs`).

### `lesion_audit.py` — quantitative realism audit (A2)

Replaces eyeball montages with per-grade numbers (high-frequency ratio = the
"airbrushed" metric, edge energy, local noise, redness) and, with `--ckpt`, a
true-class Grad-CAM "lesion presence" curve:

```bash
python analysis/lesion_audit.py \
    --real data/DRGrading/Aptos/DRGrading_128x128_train.h5 \
    --sets output/generated_cfg4/generated.h5 output/generated_cs1.5/generated.h5 \
           output/generated_blendA/generated.h5 \
    --ckpt downstream_results/real_only_s112_best.pth --backbone densenet121 \
    --out_dir output/audit
```

### New classifier backbones (C9)

`train_dr_classifier.py --backbone` additionally accepts
`vit_base_patch14_dinov2` (timm, SSL-initialized) and `swin_large`.

### Protocol runner for a filtered blend

`run_downstream_protocol.sh` accepts `BLEND_H5` and a per-run `SYNTH_TAG` so a
filtered blend gets distinct run names and can reuse the frozen real-only rows:

```bash
SYNTH_TAG=blendA_filt BLEND_H5=output/generated_blendA_filt/generated.h5 \
bash analysis/run_downstream_protocol.sh densenet121 blendA_filt \
    data/DRGrading/Aptos/DRGrading_128x128_train.h5 \
    data/DRGrading/Aptos/DRGrading_128x128_test.h5
```

## `seed_report.py` — seed-robust metric aggregation

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

Splits every `results/sample_{ode,sde}_<step>.png` grid into its cells
(the grid is 10×10 at image_size≤128, 6×6 above — dims are auto-detected
from the PNG unless `--n_rows/--n_cols` are given; cell = trained image
size, stride = cell+1 to undo `save_image(padding=1)`), keeps the 7-feature
schema of the original trace
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
111-115) for a different classifier backbone as a robustness check.
Hyperparameters are locked to the frozen ones;
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

## `embedding_map.py` — 2D image-space map: real vs augmentation vs synthetic

Answers the distribution question directly: **where does the synthetic data
actually sit relative to the real data and to classical augmentation?** No
classifier and no pretrained network is involved, so the projection describes
the image content itself rather than a model's feature space.

Each image is fundus-cropped (removes black-border/framing differences),
padded to a square, resized to `48×48` RGB (default), flattened, and optionally
extended with model-free colour/texture statistics (`--extra_feats`). All
sources are concatenated, PCA-compressed to 64 dims (fit once on the union),
and projected to 2D with UMAP (default) and/or t-SNE. Because every source
shares one embedding, their positions are directly comparable.

### Compared arms

| arm | definition |
|---|---|
| `real` | real training h5, per-grade capped |
| `real_test` | optional real test h5 (natural-variation reference) |
| `aug_train` | exactly the downstream pipeline: horizontal flip p=.5, rotation ±8°, fill=0 |
| `aug_popular` | h/v flip, rotation ±15°, resized crop (scale .85-1.0, ratio .9-1.1), colour jitter (brightness/contrast .2, saturation .15, hue .02), Gaussian blur p=.2, 16×16 cutout p=.25 |
| `<name>` | any synthetic pool passed as `--synthetic NAME=PATH` (repeatable) |

Augmentations are generated with a fixed seed from the **same** real images, so
every augmented point has an exact real counterpart. One replica per image by
default (`--aug_reps N` for more).

### Dependencies

`h5py`, `scipy`, `scikit-learn`, `matplotlib`, `Pillow` are already in
`requirements.txt`. The default UMAP projection additionally needs:

```bash
pip install umap-learn
```

`--method tsne` and `--method pca` work without it.

### Commands (Aptos 128, canonical pools)

```bash
# 1. real data (skip if the h5 files already exist)
python data_preparation/download_h5.py --dataset Aptos --resolution 128
# -> data/DRGrading/Aptos/DRGrading_128x128_{train,test}.h5

# 2. canonical headline pool + one uniform (all-grade) pool
ls output/blends/blendA_g3cs4.h5            # built by analysis/run_matrix.sh
ls output/generated_cs4.0/generated.h5      # uniform cs=4.0 pool
# If blendA_g3cs4.h5 is missing, the merge recipe is at the bottom of this section.

# 3. the map
python analysis/embedding_map.py \
    --real_h5      data/DRGrading/Aptos/DRGrading_128x128_train.h5 \
    --real_test_h5 data/DRGrading/Aptos/DRGrading_128x128_test.h5 \
    --synthetic    blendA_g3cs4=output/blends/blendA_g3cs4.h5 \
    --synthetic    cs4.0=output/generated_cs4.0/generated.h5 \
    --aug train,popular \
    --max_per_grade 500 \
    --pairs real \
    --method both \
    --out_dir experiments_outputs/embedding_map
```

`--pairs real` (default) writes one dedicated **1-vs-1** figure per source against
the real training set; `--pairs all` compares every unordered pair and
`--pairs none` disables the pairwise figures. The pairwise maps reuse the same
embedding, so they can be regenerated with `--reuse_features`:

Replot the same features with another projection (no h5 re-read):

```bash
python analysis/embedding_map.py ... --reuse_features --method tsne
```

Structure-only robustness view (adds histograms, FFT high-frequency ratio, edge
energy, local noise, redness; the statistics block is scaled to
`--extra_feats_weight` = 0.5 of the pixel-block energy so it is not swamped by
the ~7k pixel dimensions):

```bash
python analysis/embedding_map.py ... --extra_feats
```

Higher-resolution data: pass the 256 h5 paths and keep `--pixel_size 48` (the
thumbnail size is independent of the source resolution). For a first quick
look use `--method pca --no_thumbnails --max_per_grade 200`.

### Outputs (`--out_dir`)

| file | content |
|---|---|
| `umap_by_source.png` | one panel per source, colour = grade; real stays as light-grey background in non-real panels; last panel overlays all sources |
| `umap_by_grade.png` | five per-grade panels, colour = source — the direct "does synthetic cover grade g?" view |
| `umap_pair_real_vs_<name>.png` | dedicated 1-vs-1 map for each pair (`--pairs real`): global panel + one panel per grade; with `--pairs all` the names are `umap_pair_<a>_vs_<b>.png` |
| `umap_thumbnails.png` | actual image thumbnails placed at their 2D coordinates (sampled) |
| `embedding_projection.csv` | `method, source, split, grade, index, x, y` |
| `embedding_metrics.csv` | per source × grade numbers (below) |
| `pairwise_metrics.csv` | per pair (overall + per grade): B→A nearest ratio, B's neighbour fraction in A, energy distance (below) |
| `embedding_features.npz` | feature/meta/thumbnail cache for `--reuse_features` |
| `run_config.json` | parameters, per-source/per-grade counts, PCA explained variance |

`tsne_*` / `pca_*` figures are written instead of `umap_*` when requested.

### Reading the metrics (computed in 64-d PCA space, not in 2D)

- **`nn_real_ratio`** — distance from a point to its nearest same-grade real
  image, divided by the median real-real nearest-neighbour distance in that
  grade. ≈1 means "as close as another real image"; ≫1 means off-manifold.
- **`knn_real_frac`** — fraction of its 10 nearest same-grade neighbours that
  are real (pool = source + real). Low values indicate a detached/novel region;
  for augmented arms it should be high because each point has a real twin.
- **`energy_distance`** — distribution distance between the source and the
  same-grade real cloud (0 = identical, larger = more separated).
- **`silhouette_grade`** — on the `grade=all` row: how well the five grades
  remain separated *within that source*. Compare it to the real row: a
  synthetic pool with much lower silhouette has smeared grade structure.

`pairwise_metrics.csv` is the 1-vs-1 companion, with B measured against A:

- **`nn_a_ratio`** — median B→nearest-A distance divided by the median
  within-grade A–A nearest-neighbour distance. ≈1: B sits as close to A as A
  sits to itself; ≫1: B is off A's manifold.
- **`knn_a_frac`** — fraction of B's 10 nearest neighbours (pool = A + B) that
  belong to A. Low = B occupies a region A does not cover.
- **`energy_distance`** — symmetric distribution distance between A and B.

In the pairwise figures, source A (normally real) is drawn in grey and source B
in its source colour; the five grade panels show whether B covers the same
region as A within each grade, rather than only in the global projection.

### Caveats

- Pixel space is dominated by global colour, illumination and framing. That is
  a real property of the data, not a bug, but it means the global UMAP blobs
  mostly answer "do these look like the same acquisition?", while the per-grade
  panels and the PCA-space metrics answer "does this cover the class?".
- UMAP and t-SNE distort distances; never read absolute inter-cluster distances
  off the 2D plots. Use the CSV metrics for distances.
- `--max_per_grade` balances the visual comparison; without it, grade 0 would
  dominate. The counts actually plotted are in `run_config.json` and the panel
  titles.
- `blendA_g3cs4` keeps grade 4 real-only by construction, so that pool has no
  synthetic grade 4 (its grade-4 metrics row is absent). The uniform `cs4.0`
  pool covers all five grades.
- `--aug_popular` is deliberately aggressive (crop/zoom + photometric + blur +
  cutout); it is a *classical baseline*, not the training pipeline. Use
  `--aug train` alone for the exact training-time distribution.

### Building `blendA_g3cs4.h5` if it is missing

```bash
python analysis/merge_h5_by_grade.py \
    --sources \
      "0=output/generated_cs4.0/generated.h5 1=output/generated_cs4.0/generated.h5 \
       4=output/generated_cs4.0/generated.h5 2=output/generated_cs1.5/generated.h5 \
       3=output/generated_cs4.0/generated.h5" \
    --out output/blends/blendA_g3cs4.h5 --cap 1000 --caps_override "4=0"
```

This is the same per-grade recipe used by `run_matrix.sh` (g4 real-only).
