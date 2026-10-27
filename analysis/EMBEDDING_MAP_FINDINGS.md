# Image-space embedding map — findings (APTOS 128, 2026-10-27)

**Question.** Where does the CCDM synthetic data actually sit relative to the
real data and to classical augmentation, when the map is built from image
content alone (no classifier, no pretrained network)?

**Canonical artifacts.** `experiments_outputs/embedding_map_rerun/`
(gitignored; valid feature cache, UMAP and t-SNE):

```
umap_by_source.png, umap_by_grade.png, umap_thumbnails.png
tsne_by_source.png, tsne_by_grade.png, tsne_thumbnails.png
umap_pair_real_vs_*.png, tsne_pair_real_vs_*.png
embedding_metrics.csv, pairwise_metrics.csv, embedding_projection.csv
run_config.json, embedding_features.npz   (CRC OK)
```

The earlier `experiments_outputs/embedding_map/` contains the same metric
tables and a byte-identical UMAP, but its copied feature cache is corrupt; use
the rerun as canonical. The two metric tables are also kept in the repository
at `analysis/embedding_map_metrics_2026-10-27/` (verified identical to the
rerun).

**Verification.** The 2026-10-27 rerun (`--method both`) reproduced
`embedding_metrics.csv` and `pairwise_metrics.csv` byte-identically, and the
UMAP projection rows have the same SHA-256 as the first run. The t-SNE view
added by the rerun is consistent with UMAP and does not change any conclusion.
All numbers below are exact from the feature matrix; the earlier
“approximate numbers” caveat is resolved.

## TL;DR

1. **`cs1.5` is the most real-like synthetic pool; `cs4.0` is the least.**
   Higher guidance buys grade-typicality at the cost of local realism,
   within-grade diversity and coverage.
2. **The `blendA_g3cs4` switch makes grade 3 less real-like, not more.** From
   `blendA` to `blendA_g3cs4` (only g3 changes): kNN-real 0.137 → 0.030,
   energy 0.739 → 1.067, grade-plane coverage 0.94 → 0.69, full-space coverage
   0.45 → 0.12. Yet downstream g3 recall improves. This is a class-typicality
   effect, not a pixel-level realism gain.
3. **Synthetic pools are more grade-separable and less diverse than real data**
   (grade silhouette −0.014 real vs +0.019…+0.077 synthetic; within-grade
   spread mostly 0.5–0.85× real, g0 ≈0.52 in every synthetic pool).
4. **The blends confound grade with generator identity.** `blendA`/`blendA_g3cs4`
   use cs4.0 for g0/g1 and cs1.5 for g2/g3, so grade is partly encoded by the
   generator look; a classifier can exploit that shortcut.
5. **Coverage and interleaving are both incomplete, and they are different.**
   At matched n=500/grade, `cs1.5` covers 92–99% of real points in the
   grade-defining PC2–PC3 plane (within half a within-grade radius) but only
   35–62% in the full 64-d appearance space; `cs4.0` covers 40–72% / 10–28%.
   Locally, only ~27% of a cs1.5 point's 10 nearest neighbours are real (4% for
   cs4.0; 67% for a genuine real holdout). Synthetic data forms dense
   sub-clusters next to the real islands rather than mixing into them.

## Run configuration

| item | value |
|---|---|
| dataset | APTOS, `DRGrading_128x128_{train,test}.h5` |
| representation | fundus crop → 48×48 RGB → PCA-64 (94.77% variance) → UMAP and t-SNE |
| seed | 111 |
| cap | 500 images per (source, grade) |
| augmentation arms | `aug_train` (hflip + rot ±8°), `aug_popular` (h/v flip, rot ±15°, resized crop, colour jitter, blur, cutout) |
| synthetic pools | `blendA`, `blendA_g3cs4`, `cs1.5`, `cs4.0` |
| projection | UMAP (`n_neighbors 30`, `min_dist 0.1`) and t-SNE (`perplexity 30`, PCA init) |

Counts after capping:

| source | n | g0 | g1 | g2 | g3 | g4 |
|---|---:|---:|---:|---:|---:|---:|
| `real` | 1686 | 500 | 296 | 500 | 154 | 236 |
| `aug_train` | 1686 | 500 | 296 | 500 | 154 | 236 |
| `aug_popular` | 1686 | 500 | 296 | 500 | 154 | 236 |
| `real_test` | 733 | 361 | 74 | 200 | 39 | 59 |
| `blendA` | 2000 | 500 | 500 | 500 | 500 | 0 |
| `blendA_g3cs4` | 2000 | 500 | 500 | 500 | 500 | 0 |
| `cs1.5` | 2500 | 500 | 500 | 500 | 500 | 500 |
| `cs4.0` | 2500 | 500 | 500 | 500 | 500 | 500 |

The blend arms have **no synthetic grade 4** by construction (g4 real-only).

## 1. Global distribution metrics

Computed in the 64-d PCA space, not in 2D:

| source | nn_real_ratio | knn_real_frac | energy_distance | silhouette_grade |
|---|---:|---:|---:|---:|
| `real` | 1.000 | 1.000 | 0.000 | −0.014 |
| `real_test` | 0.869 | **0.669** | 0.399 | +0.011 |
| `aug_train` | 0.862 | 0.450 | 0.287 | −0.016 |
| **`cs1.5`** | 0.810 | **0.274** | **0.535** | +0.019 |
| **`blendA`** | 0.949 | 0.227 | 0.604 | +0.077 |
| **`blendA_g3cs4`** | 1.008 | 0.176 | 0.758 | +0.073 |
| **`cs4.0`** | 1.061 | **0.040** | **0.964** | +0.027 |
| `aug_popular` | 1.792 | 0.224 | 1.019 | −0.030 |

Reading:

- Synthetic points are **globally close** to real (`nn_real_ratio` 0.8–1.1),
  but **locally segregated** (`knn_real_frac` 0.04–0.27 vs 0.67 for a real
  holdout). The generator reproduces the global manifold, not the local
  density of real data.
- **`cs1.5` is the most real-like synthetic pool**; **`cs4.0` is the least**.
  At global scale cs4.0 is as far from real as the deliberately aggressive
  `aug_popular` arm (energy 0.96 vs 1.02).
- `blendA` sits between cs1.5 and cs4.0. `blendA_g3cs4` is worse than `blendA`
  because it replaces the g3 cs1.5 images with cs4.0 images.
- `real_test` is the natural-variation yardstick: 67% real neighbours, energy
  0.40, a grade silhouette near zero.

## 2. Per-grade local realism (kNN real fraction)

| source | g0 | g1 | g2 | g3 | g4 |
|---|---:|---:|---:|---:|---:|
| `real_test` | 0.581 | 0.793 | 0.724 | 0.769 | 0.790 |
| `aug_train` | 0.528 | 0.434 | 0.457 | 0.520 | 0.504 |
| `cs1.5` | 0.357 | 0.286 | **0.398** | 0.147 | 0.191 |
| `cs4.0` | 0.032 | 0.043 | 0.106 | 0.028 | 0.035 |
| `blendA` | 0.026 | 0.048 | **0.380** | 0.137 | — |
| `blendA_g3cs4` | 0.029 | 0.046 | **0.394** | **0.030** | — |

`cs1.5` is best for grades 2–4; grade 3 is still its weakest class (0.147).
`cs4.0` is uniformly low: only 2.8–10.6% of its neighbours are real. The
`blendA` and `blendA_g3cs4` g0/g1/g2 rows agree within noise, confirming that
the g3 difference is the actual intervention, not a different sampling draw.

## 3. The g3cs4 probe, isolated

`blendA` and `blendA_g3cs4` are identical except grade 3 (cs1.5 → cs4.0):

| g3 source | nn_real_ratio | kNN real | energy | coverage (PC2–PC3, <0.5r) | coverage (64-d, <0.5r) |
|---|---:|---:|---:|---:|---:|
| `cs1.5` | 0.845 | **0.147** | 0.779 | 0.94 | 0.35 |
| `blendA` (g3 = cs1.5) | 0.841 | **0.137** | 0.739 | 0.94 | 0.45 |
| `cs4.0` | 0.999 | **0.028** | 1.048 | 0.68 | 0.10 |
| `blendA_g3cs4` (g3 = cs4.0) | 1.019 | **0.030** | 1.067 | **0.69** | **0.12** |

The g3cs4 images are **~4.5× less likely to have a real neighbour**, ~45%
farther in distribution distance, and cover far less of the real g3 region.
Yet the downstream protocol reports better g3 recall for `blendA_g3cs4`. The
likely explanation is **stronger, more prototypical grade-3 appearance**
(possibly exaggerated lesions/texture) rather than closer realism. This should
be validated with a lesion-level audit before claiming improved fidelity.

## 4. Grade structure and diversity

Global 64-d grade silhouette (from the CSV):

```
real          -0.014
cs1.5         +0.019
cs4.0         +0.027
blendA        +0.077
blendA_g3cs4  +0.073
```

The near-zero real value does **not** mean grade is absent. In the feature
space, the grade signal is concentrated in lower-variance components: PC2
(15.7% of variance) explains **63.5%** of grade variance, while PC1 (37.5%,
dominated by brightness/contrast/redness) explains 2.8%. The grade axis is
mostly **g0 vs any DR** (PC2 means: g0 +4.54, g1–g4 −1.1…−2.0); g1–g4 overlap
heavily (PC2–PC3 grade silhouette +0.047). The UMAP/t-SNE layout is driven by
PC1 and source/appearance axes, which is why the 2D map looks grade-blind.

Within-grade spread relative to real (PC2–PC3; <1 = less diverse):

| source | g0 | g1 | g2 | g3 | g4 |
|---|---:|---:|---:|---:|---:|
| `cs1.5` | 0.51 | 0.98 | 0.73 | 0.84 | 0.76 |
| `cs4.0` | 0.52 | 0.78 | 0.59 | 0.71 | 0.68 |
| `blendA` | 0.52 | 0.79 | 0.74 | 0.84 | — |
| `blendA_g3cs4` | 0.52 | 0.82 | 0.78 | 0.70 | — |
| `aug_train` | 1.09 | 1.01 | 1.11 | 1.19 | 1.12 |
| `aug_popular` | 1.05 | 1.08 | 1.20 | 1.23 | 1.18 |
| `real_test` | 0.99 | 1.04 | 1.19 | 1.18 | 1.05 |

- Synthetic grades have **roughly 50–85% of real within-grade spread** in most
  cells (cs1.5 g1 is the exception at 0.98); g0 is the most collapsed in every
  synthetic pool (≈0.52).
- `cs4.0` is consistently tighter than `cs1.5` (e.g. g2 0.59 vs 0.73,
  g3 0.71 vs 0.84). Higher guidance produces prototypical grade clusters.
- `aug_train`/`aug_popular` increase spread (1.0–1.23), and `real_test` is
  ≈1.0, validating the scale.

**t-SNE sanity check.** t-SNE gives the same picture as UMAP: grade silhouette
+0.031 vs +0.035 (UMAP), source silhouette −0.067 vs −0.082, and the same kNN
ordering (`real_test` 0.68 > `cs1.5` 0.32 > `blendA` 0.27 > `blendA_g3cs4`
0.24 > `cs4.0` 0.09). t-SNE visually separates grade 0 into its own islands
and shows the g1–g4 overlap in the upper cloud; it does not reveal a different
structure.

## 5. Coverage vs interleaving

Coverage and local interleaving are different properties. Coverage depends on
the number of points in the covering set, so only compare pools with the same
per-grade count. `cs1.5` and `cs4.0` both have 500/grade; `real_test` has
39–361/grade and is not directly comparable.

**Grade-plane coverage** (PC2–PC3): fraction of real points with a same-grade
synthetic sample within half a real within-grade radius:

| source | g0 | g1 | g2 | g3 | g4 |
|---|---:|---:|---:|---:|---:|
| `cs1.5` | 0.92 | 0.99 | 0.92 | 0.94 | 0.93 |
| `cs4.0` | **0.40** | 0.68 | 0.72 | 0.68 | 0.69 |
| `blendA` | 0.48 | 0.70 | 0.94 | 0.94 | — |
| `blendA_g3cs4` | 0.48 | 0.68 | 0.95 | **0.69** | — |

**Full-space coverage** (64-d, same threshold): fraction of real points with a
same-grade synthetic sample within half a real within-grade radius:

| source | g0 | g1 | g2 | g3 | g4 |
|---|---:|---:|---:|---:|---:|
| `cs1.5` | 0.62 | 0.50 | 0.51 | 0.35 | 0.58 |
| `cs4.0` | 0.26 | 0.12 | 0.22 | 0.10 | 0.28 |
| `blendA` | 0.26 | 0.13 | 0.51 | 0.45 | — |
| `blendA_g3cs4` | 0.28 | 0.12 | 0.53 | **0.12** | — |

Interpretation:

- **`cs1.5` covers the grade-defining plane well (92–99%)** but still leaves
  full-appearance gaps (35–62%). It covers the grade axis without uniformly
  covering the real appearance neighborhood.
- **`cs4.0` is much worse on both** (40–72% grade plane, 10–28% full space),
  with a pronounced grade-0 gap (40% / 26%).
- **The g3cs4 switch opens a g3 coverage gap**: 94% → 69% in the grade plane,
  45% → 12% in full space.
- **Local interleaving is poor for every synthetic pool.** Within-source
  density is 0.60–0.78× real (vs 1.00 for training augmentation), and only
  4–27% of a synthetic point's 10 nearest neighbours are real.
- The 2D UMAP/t-SNE maps **overstate coverage**; do not read coverage off the
  scatter plots. The numbers above are from PCA space.

Visually (`umap_pair_real_vs_*.png`, `tsne_pair_real_vs_*.png`): `cs1.5` hugs
the real islands; `cs4.0` forms adjacent dense bands inside the same islands
but offset; `aug_train` is essentially on top of real; `aug_popular` fills the
central band and leaves parts of the real cloud uncovered. All pools reproduce
the global acquisition/quality islands — there is no collapse to a single mode.

## 6. Blend confound: grade ↔ generator appearance

`blendA` and `blendA_g3cs4` use different generators per grade (g0/g1 from
cs4.0, g2/g3 from cs1.5, g4 real-only). In `blendA`'s own PCA, the silhouette
of “which generator made this?” is **0.101**, versus **0.076** for the four
grades; for `blendA_g3cs4` it is 0.077 vs 0.072. The generator look is at
least as separable as grade.

Consequence: a classifier trained on the blend can learn “cs4.0 appearance =
grades 0/1” and “cs1.5 appearance = grades 2/3” without learning DR biology.
Part of the downstream gain of the blend (and the g3cs4 probe) may therefore be
a **generator-appearance shortcut**. A clean control is an all-grade single-pool
arm (all grades cs1.5 or all grades cs4.0) at matched volume.

## 7. Classical augmentation reference

- `aug_train` is effectively co-located with real (kNN real 0.45, energy 0.29;
  within-grade spread ≈1.0–1.19; full-space coverage 0.41–0.83). It is the
  correct “what the model actually sees” baseline.
- `aug_popular` is far (kNN real 0.22, energy 1.02; spread 1.05–1.23) and has
  very low full-space coverage (0.01–0.12), because it deliberately changes
  brightness, contrast, crop and blur. This is expected in an
  appearance-based space and must **not** be read as a quality ranking. It
  does show that `cs1.5` is more appearance-conservative than aggressive
  classical augmentation.

## 8. What we can and cannot claim

**Supported by this run**

- CCDM reproduces the global acquisition/quality structure of APTOS and covers
  the grade-defining regions coarsely, with pool-dependent gaps.
- Synthetic data is locally segregated and low-diversity per grade; higher
  cond_scale increases grade-typicality while decreasing realism, diversity
  and coverage.
- `cs1.5` is the most real-like pool; `cs4.0` is the least.
- The g3cs4 g3 gain is not explained by closer pixel-level realism or broader
  coverage.
- The blends confound grade with generator appearance.

**Not supported**

- “Synthetic data is more diverse than real data.” In this space it is less
  diverse per grade (≈0.5–0.85× real within-grade spread).
- “g3cs4 is more realistic.” It is less locally real-like and covers less of
  the real g3 region.
- “Pixel-space proximity proves correct lesions.” It does not; lesion-level
  validation is separate.
- “The blend gain is purely biological.” The per-grade generator switch is a
  plausible shortcut.

## 9. Limitations

- **Appearance, not pathology**: raw pixels do not encode lesion identity;
  synthetic images can look globally real and still have wrong lesions.
- **Coverage is sample-size dependent**: only compare arms with matched
  per-grade counts; `real_test` has far fewer points than the synthetic pools.
- **Two coverage spaces**: PC2–PC3 measures grade-region coverage; 64-d
  measures full appearance-neighborhood coverage. Report both.
- **Caps**: 500/grade; synthetic pools are subsampled from 1,000–2,500, and
  real minority grades have 154–296. Density/coverage metrics are
  draw-sensitive.
- **Projections distort**: UMAP/t-SNE overstate overlap and equalise density;
  all quantitative claims above are from PCA space.
- **Energy distance** floors at 0 for small samples (e.g. real-test g1/g3).
- **Only APTOS 128** was analysed; no IDRiD/DDR/Messidor comparison yet.
- The blend arms have no synthetic grade 4 by design; their g4 metrics are
  real-only.

## 10. Reproduction

```bash
pip install umap-learn   # analysis-only

python analysis/embedding_map.py \
  --real_h5      data/DRGrading/Aptos/DRGrading_128x128_train.h5 \
  --real_test_h5 data/DRGrading/Aptos/DRGrading_128x128_test.h5 \
  --synthetic    blendA=output/blends/blendA.h5 \
  --synthetic    blendA_g3cs4=output/blends/blendA_g3cs4.h5 \
  --synthetic    cs1.5=output/generated_cs1.5/generated.h5 \
  --synthetic    cs4.0=output/generated_cs4.0/generated.h5 \
  --aug train,popular \
  --max_per_grade 500 \
  --pairs real \
  --method both \
  --out_dir experiments_outputs/embedding_map
```

`--method both` writes UMAP + t-SNE; `--pairs all` writes every unordered pair
map; `--reuse_features` replots from a valid `embedding_features.npz`.

## 11. Recommended next steps

1. **Single-pool all-grade controls**: all grades cs1.5 vs all grades cs4.0,
   matched volume, to separate the cond_scale effect from the blend's
   generator-per-grade confound.
2. **Lesion audit** (`analysis/lesion_audit.py`) for g3/g4, since the pixel map
   cannot certify lesion correctness.
3. **Grade-aware view**: add a PC2–PC3 panel (or whitened-PC UMAP) so the
   healthy-vs-DR axis and g1–g4 overlap are visible directly.
4. **Report fidelity, diversity and coverage together** in the paper: global
   proximity is necessary but not sufficient; local interleaving, within-grade
   spread and coverage all show a consistent cost at higher cond_scale.
