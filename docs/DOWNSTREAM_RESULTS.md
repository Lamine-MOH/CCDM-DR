# Downstream DR classification — results with grade-conditioned synthetic augmentation

> **Status: CANONICAL section = "Exp 3" (2026-09-16).** Results below in the
> [Exp 3](##Exp 3 — Tier-3 retrain + val-based multi-backbone matrix (executed 2026-09-16)) section were
> produced under the post-Tier-3 protocol (stratified 80-20 split, best-epoch
> selection on a validation split, `--pretrained`/augmentation/seeding fixes).
> Everything below that section is a complete **historical** record from the
> *old* protocol (85/15 non-stratified split, best-epoch selection on the test
> set, `--pretrained` bug, pre-Tier-3 diffusion) and is **no longer canonical**.

## Exp 3 — Tier-3 retrain + val-based multi-backbone matrix (executed 2026-09-16)

Canonical. Produced after the Tier-3 retrain (LayerNorm `cond_map`, EDM
loss-weight fix, minority-label embedding replication) with all Phase 1 audit
fixes landed (val selection, `--no-pretrained` fix, augments, seeding).
Orchestrated end-to-end by `analysis/run_matrix.sh`.

### Protocol (executed)

- **Data:** stratified 80/20 (`--test_frac 0.20 --seed 111`); train 2929
  (g0..g4 = 1444/296/799/154/236), test 733 (g0..g4 = 361/74/200/39/59).
- **Classifier selection:** per-run val carve from the real train h5
  (`--val_frac 0.12 --val_seed 999`, deterministic, grade-stratified, identical
  across arms/seeds); best epoch by **val QWK**; single final test evaluation.
  Reported: QWK / ACC / macro-F1 / per-grade recall.
- **Backbones:** `densenet121`, `resnet50`, `efficientnet_b4`; seeds 111–115;
  30 epochs; Adam lr 1e-4; batch 32; ImageNet-pretrained.
- **Generation** (`generate_from_ckpt.py`, `model-100000.pt` EMA, sampler `sde`
  32 steps): three all-grades passes F1/F2/F3 →
  `generated_cs{4.0,2.5,1.5}/generated.h5` (cond_scale 4.0 / 2.5 / 1.5),
  nfake ≥ 2000/grade.
- **Arms:** `real_only`; `blendA` (frozen recipe: g0/1/4@cs4.0, g2/3@cs1.5,
  cap 1000/grade, **g4 real-only**); `blend_sel` (data-driven per-grade cs);
  sensitivities `blendA_plus_g4` (all-synthetic g4), `blendA_cap500`,
  `blendA_cap1500`.
- **Integrity:** 65 runs; every `selection_on == "val"`, `val_seed = 999`;
  seed coverage complete (111 has all single-seed arms; 112–115 the 5-seed grid);
  no missing/partial runs. Results are local-only (`downstream_results/Exp 3`).

### Sweep & selection (densenet121 — mean VAL per-grade recall over 5 seeds)

| grade | cs1.5 | cs2.5 | cs4.0 | real-only | argmax |
|---|---|---|---|---|---|
| g0 | 0.971 | 0.977 | 0.978 | 0.979 | cs4.0 |
| g1 | 0.633 | 0.594 | **0.644** | 0.628 | cs4.0 |
| g2 | **0.863** | 0.840 | 0.808 | 0.777 | cs1.5 |
| g3 | 0.356 | 0.411 | **0.500** | 0.322 | cs4.0 |
| g4 | **0.657** | 0.579 | 0.579 | 0.564 | cs1.5 |

`blend_sel` = per-grade argmax (g0/g1@cs4, g2@cs1.5, g3@cs4, g4@cs1.5-synth),
differing from frozen blendA only at **g3 (cs1.5→cs4)** and **g4 (real→cs1.5
synth)**. Selection used only the val split.

### Headline (5 seeds, mean ± std) — QWK / ACC / macro-F1

| backbone | real_only | blendA | blend_sel |
|---|---|---|---|
| densenet121 | 0.8924±0.0090 / 0.8265 / 0.6673 | 0.8930±0.0075 / 0.8191 / 0.6373 | 0.8852±0.0028 / 0.8221 / 0.6498 |
| resnet50 | 0.8812±0.0063 / 0.8128 / 0.6441 | **0.8939±0.0063** / 0.8256 / 0.6605 | 0.8891±0.0108 / 0.8240 / 0.6508 |
| efficientnet_b4 | 0.8641±0.0104 / 0.7700 / 0.6119 | 0.8491±0.0068 / 0.7787 / 0.5830 | 0.8505±0.0123 / 0.7853 / 0.6028 |

### Paired deltas over real_only (mean Δ; seed wins +k/−m of 5)

| backbone | metric | blendA − real | blend_sel − real |
|---|---|---|---|
| densenet121 | QWK | +0.0006 (+2/−3) | −0.0072 (+2/−3) |
| densenet121 | ACC | −0.0074 (+2/−3) | −0.0044 (+3/−2) |
| densenet121 | macro-F1 | −0.0300 (+1/−4) | −0.0175 (+0/−5) |
| resnet50 | QWK | **+0.0126 (+5/−0)** | +0.0078 (+3/−2) |
| resnet50 | ACC | +0.0128 (+4/−1) | +0.0112 (+4/−1) |
| resnet50 | macro-F1 | +0.0164 (+4/−1) | +0.0067 (+4/−1) |
| efficientnet_b4 | QWK | −0.0150 (+1/−4) | −0.0136 (+1/−4) |
| efficientnet_b4 | ACC | +0.0087 (+3/−1) | +0.0153 (+4/−1) |
| efficientnet_b4 | macro-F1 | −0.0289 (+0/−5) | −0.0091 (+2/−3) |

`blend_sel` vs `blendA` loses QWK on every backbone (−0.0078 / −0.0048 / +0.0014)
but recovers ACC / macro-F1 on densenet121 (+3/−2, +4/−1) and efficientnet (+4/−1).

### Per-grade test recall (densenet121, mean ± std)

| arm | g0 | g1 | g2 | g3 | g4 |
|---|---|---|---|---|---|
| real_only | 0.986±0.003 | 0.573±0.030 | 0.798±0.028 | 0.359±0.018 | 0.576±0.024 |
| blendA | 0.983±0.003 | 0.530±0.061 | 0.816±0.033 | **0.221±0.047** | 0.586±0.052 |
| blend_sel | 0.983±0.005 | 0.551±0.066 | 0.820±0.036 | 0.318±0.095 | 0.515±0.054 |

### Probes (densenet121, seed 111)

| arm | QWK | note |
|---|---|---|
| real_only | 0.8853 | baseline |
| blendA (g4 real-only, cap 1000) | **0.9026** | best — the frozen recipe |
| blendA_cap500 | 0.8995 | lower cap ~ neutral |
| blendA_cap1500 | 0.8940 | higher cap slightly worse |
| blendA_plus_g4 (synthetic g4) | 0.8874 | **g4 recall drops on all backbones** (den 0.542 vs 0.644; r50 0.559 vs 0.661; eff 0.373 vs 0.593) → g4 stays real-only |

### Interpretation

- **Backbone heterogeneity is the honest headline.** blendA is seed-clean positive
  on **resnet50** (QWK +0.0126, +5/−0; ACC and macro-F1 +4/−1), essentially flat on
  **densenet121** (QWK +0.0006, mixed signs), and negative on **efficientnet_b4**
  (QWK −0.015, macro-F1 −0.029 with 0/5 wins). The augmentation benefit is
  therefore real but architecture-dependent.
- **The gain is ordinal (QWK), not per-class recall.** Test per-grade recall is
  real-only-best for g1/g3/g4 and synthetic-best only for g2, yet blendA still
  lifts resnet50 QWK in all 5 seeds — the benefit flows through confusion/ordinal
  structure, not class-level recall. This also explains the macro-F1 drops (a
  recall-weighted metric).
- **`blend_sel` (per-grade cs selected on val recall) is a negative result** — it
  never beats blendA on QWK, because per-grade val recall does not transfer to
  test (see below).
- **Synthetic g4 is toxic** on all three backbones → the frozen "g4 real-only"
  decision is validated and kept.
- **g3 is the one actionable per-grade finding:** blendA's cs1.5 g3 images
  coincide with the worst test cell (0.221); the sel map's g3@cs4 recovers it
  (0.318). The follow-up probe `blendA_g3cs4` (exact blendA, only g3@cs4) tests
  whether that recovery is real and QWK-neutral on all 3 backbones.

### Selection-transfer analysis (zero-GPU; why not to over-engineer selection)

Per-grade val recall vs test recall across {cs1.5, cs2.5, cs4.0, real} (Pearson r
over per-condition means, densenet121): g0 +0.94, g2 +0.57, but **g1 −0.21,
g3 −0.45, g4 −0.52** — the ordering **anti-transfers** for the minority grades
(val per-grade cells n = 173/36/96/18/28 vs test n = 361/74/200/39/59; 5-seed val
recall SD up to ±0.10 on g3/g4). No candidate rule (argmax-mean, max-min,
real-gated, adjacent-aware) reproduces the frozen recipe (2/5, 0/5, 1/5, 1/5
matches), and even a generous real-gated rule keeps selecting synthetic g4.
Conclusion: **per-grade selection grids (larger val split, k-fold) are low-ROI at
this data scale**; we keep the frozen blendA recipe and report per-grade selection
as an explicit limitation. (A QWK-based selection variant is also unreliable here:
`blend_sel` had the best val QWK yet the worst test QWK.)

### Follow-up probe — `blendA_g3cs4` (launched 2026-09-16)

Exact blendA except **g3@cs4.0** (g0/1/4@cs4, g2@cs1.5, g3@cs4, cap 1000, g4
real-only). Tests whether g3's test-recall recovery (0.221→0.318) is real and
QWK-neutral on all 3 backbones × 5 seeds. Results will be appended to this section.

### Backbone rationale & choice

The three headline backbones cover distinct architectural design paradigms, each
with published 5-class DR-grading usage on APTOS fundus data and standard
ImageNet transfer-learning precedent:

- **ResNet-50** — He et al., *Deep Residual Learning for Image Recognition*, CVPR
  2016. Residual skip connections for very deep CNNs. Documented APTOS
  comparisons: Dongre et al. 2026 (controlled APTOS 5-class study; ResNet50 96%
  val acc; ImageNet pre-training worth +43 pts over random init); Mohan Kumar &
  Shalini 2025 (best APTOS test acc 86.5% of the models compared); Sankara Babu
  et al. 2023 (79.3%, above DenseNet on the same setup).
- **DenseNet-121** — Huang et al., *Densely Connected Convolutional Networks*,
  CVPR 2017. Dense connectivity → feature reuse, parameter-efficient. Widely
  adopted for DR grading: Sruthi et al. 2023 (Appl. Sci.; best across
  APTOS/EyePACS/ODIR, ≥98.5% accuracy); Dixit et al. 2025 (81% on 5-class APTOS);
  Retima et al. 2025 (99%/81% binary/quinary); an APTOS optimizer study 2025
  (DenseNet121+SGD 96.9% test acc).
- **EfficientNet-B4** — Tan & Le, *EfficientNet: Rethinking Model Scaling for
  Convolutional Neural Networks*, ICML 2019. NAS + compound scaling of
  depth/width/resolution. APTOS 2019 leaderboard relied on EfficientNet ensembles
  (top solutions on B5/B7; an independent B3/B5 entry reached QWK 0.906);
  EfficientNet-B3/B4 repeatedly best on larger, imbalanced DR datasets.

Rationale: (i) architectural diversity (residual / dense / compound-scaled);
(ii) all three ship ImageNet-pretrained torchvision weights → identical transfer
protocol; (iii) all three have direct, citable APTOS DR-grading track records;
(iv) they bracket the parameter range feasible for 30-epoch training at 128×128
under a fixed budget (densenet121 ≈ 7 M, resnet50 ≈ 23 M, efficientnet_b4 ≈ 19 M).

**Additional backbones (documented, deferred):** torchvision-resolvable
candidates for a later expansion — `resnext50_32x4d` (SE-ResNeXt family behind
APTOS top solutions), `efficientnet_b5` (APTOS 2019 top-3% solutions),
`resnet101` / `densenet201` (deeper siblings, already supported),
`mobilenet_v2` (lightweight bracket), `inception_v3` (provenance: Gulshan et al.,
JAMA 2016; needs a 299-native input-size path). Not pursued at 128:
ViT/DINOv2/Swin (resolution-limited — see historical C9 below). Budget per added
headline backbone ≈ 4 arms × 5 seeds.

## How this was reached (historical)

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
| 2, 3 | `output/generated_cs1.5/generated.h5` (legacy, attr-tagged) | 1.5 |

Assembled with:

```bash
python analysis/merge_h5_by_grade.py --sources \
    "0=output/generated_cfg4/generated.h5 1=output/generated_cfg4/generated.h5 \
     2=output/generated_cs1.5/generated.h5 3=output/generated_cs1.5/generated.h5 \
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

## Second-backbone robustness check (resnet50)

Same frozen protocol with only `--backbone resnet50` (seeds 111–115, real-only
+ blendA at `--synthetic_cap_per_grade 1000`), run via
`analysis/run_downstream_protocol.sh resnet50 r50 ...`.

| metric | real-only | real + synth | delta (paired) |
|---|---|---|---|
| accuracy | 0.8215 ± 0.0117 | 0.8299 ± 0.0058 | +0.0084 ± 0.0140 (+3/−1) |
| macro-F1 | 0.6831 ± 0.0134 | 0.6870 ± 0.0186 | +0.0038 ± 0.0265 (+4/−1) |
| QWK | 0.8996 ± 0.0079 | 0.9036 ± 0.0058 | +0.0040 ± 0.0088 (+3/−2) |
| recall g0 | 0.9886 ± 0.0027 | 0.9848 ± 0.0213 | −0.0038 ± 0.0217 (+3/−1) |
| recall g1 | 0.5967 ± 0.0447 | 0.5667 ± 0.0540 | **−0.0300 ± 0.0139 (0/−5)** |
| recall g2 | 0.7513 ± 0.0557 | 0.8118 ± 0.0350 | **+0.0605 ± 0.0627 (+4/−0)** |
| recall g3 | 0.4313 ± 0.0895 | 0.3812 ± 0.1114 | −0.0500 ± 0.1633 (+2/−2) |
| recall g4 | 0.6476 ± 0.0543 | 0.6429 ± 0.0168 | −0.0048 ± 0.0616 (+1/−3) |

Interpretation:

- **Positive-on-average but not seed-clean.** All three core metrics improve on
  average, but agreement is only 3–4 of 5 seeds (vs densenet's +5/−0), so this
  is **not** evidence of a uniform cross-backbone gain.
- **Consistent where it matters most:** grade-2 recall (+0.060, +4/−0) and
  macro-F1 (+4/−1) improve; grade-3 is unchanged noise, as on densenet.
- **Backbone-dependent per-grade effects:** grade-1 recall reliably *worsens*
  (0/−5) on resnet50 (flat on densenet); grade-4 gain does not transfer
  (densenet +0.081 vs −0.005 here).
- Seed variance still drops for accuracy and QWK, slightly rises for macro-F1.
- **Bottom line for the paper:** the headline stays densenet121 (frozen,
  +5/−0). The resnet50 check is honest secondary evidence that the effect is
  positive-on-average but architecture-sensitive — a limitations sentence, and
  a 3rd backbone or a larger test set are the natural follow-ups if
  cross-backbone robustness is claimed.

## Batch A — supplemental results (2026-09-08)

Supplementary evidence collected **after** the frozen result above; the headline
densenet121 + blendA table is unchanged. Batch A = no generator retrain.

### A1. Semantic filtering of the synthetic pool — negative result (control)

Hypothesis from ECC_DM (MICCAI 2025): an ensemble of real-trained classifiers
should retain only "good" synthetic samples, improving the augmented set.
Per-grade pass rates (dedup off, max-likelihood ensemble of densenet121 +
resnet50): cfg4 957/645/207/3/527, cs1.5 763/690/541/80/490. Two matched 2223-image
blends were validated on the frozen 5-seed protocol.

| arm | acc | macro-F1 | QWK |
|---|---|---|---|
| real_only (frozen) | 0.8295 | 0.6916 | 0.9063 |
| blendA 1000/grade (frozen) | **0.8426** | **0.7129** | **0.9204** |
| random control (2223, matched counts) | 0.8350 | 0.7049 | 0.9111 |
| filtered (2223) | 0.8284 | 0.7018 | 0.9082 |

**Interpretation:** filtering does not beat random selection at matched quantity
(filtered ≈ real_only; random > filtered on all three headline metrics). The
gain of blendA comes from **volume + diversity**, not label purity. filtering is
dropped from the method narrative and reported as a negative control. (Per-grade:
filtered g3 +0.106 but on only 80 samples; not claimable.)

### A2. Quantitative lesion audit

`analysis/lesion_audit.py`: high-frequency ratio, edge energy, local noise, color
stats, and true-class Grad-CAM (mean lesion-attribution over the retina) for real
vs cfg4 / cs1.5 / blendA / filtered, per grade.

- **High-frequency "airbrushed" proxy:** real 0.074–0.115 vs synthetic
  0.080–0.135 — synthetics are *not* smoother than real; this common criticism
  is not confirmed by the metric.
- **Redness** follows the real gentle rise into g1/g2; g3/g4 softer.
- **Grad-CAM** decreases with severity in both real (0.00148→0.00051) and
  synthetic (cfg4 0.00145→0.00080), but synthetic amplitude is flatter/lower.
  The filtered blend's g2/g3 are the weakest-CAM samples — i.e. the filter
  kept the outliers, not the healthy-core samples.
- Caveat: CAM is a classifier-attribution proxy, not a lesion segmentor.
- Figures `realism_by_grade.png`, `lesion_presence_by_grade.png` staged; upload
  to the public share is pending (note: filtered and randmatch blend h5s
  `output/generated_blendA_filt/`, `output/generated_blendA_randmatch/` are on
  TM and can be shared for regeneration).

### E16. External-domain generalization (Messidor-2, IDRiD, DDR)

Train on APTOS (real-only vs +blendA 1000/grade), test on each external set's
test split, densenet121, 3 seeds. Pre-built 128 h5s via `download_h5.py`;
Messidor-2 ships with native 0–4 labels, so no grade-merge re-scoring was needed.

| set (test n) | arm | acc | macro-F1 | QWK | seed agreement (acc/mF1/QWK) |
|---|---|---|---|---|---|
| IDRiD (77) | real_only | 0.3723 ± 0.027 | 0.3457 ± 0.027 | 0.6823 ± 0.036 | — |
| IDRiD (77) | +blendA | 0.3983 ± 0.064 | 0.2974 ± 0.079 | 0.6915 ± 0.002 | +2/−1 / +1/−2 / +2/−1 |
| Messidor-2 (261) | real_only | 0.6296 ± 0.012 | 0.3706 ± 0.007 | 0.5083 ± 0.002 | — |
| Messidor-2 (261) | +blendA | 0.6054 ± 0.008 | 0.3537 ± 0.029 | 0.5145 ± 0.040 | 0/−3 / +2/−1 / +2/−1 |
| DDR (1878) | real_only | 0.5619 ± 0.026 | 0.3147 ± 0.007 | 0.5518 ± 0.013 | — |
| DDR (1878) | +blendA | 0.6022 ± 0.010 | 0.3682 ± 0.017 | 0.5993 ± 0.005 | +3/−0 / +3/−0 / +3/−0 |

**Interpretation:**
- On the **largest external set (DDR)** the augmentation benefit transfers
  robustly (+3/−0 on accuracy, macro-F1, QWK; +0.040/+0.054/+0.048) — the
  headline effect is a generalization, not APTOS-overfit.
- On the small sets the signal is noise-dominated: Messidor-2 accuracy dips on
  all 3 seeds (QWK still rises); IDRiD's 77-image test flips sign per metric.
- Absolute transfer is low everywhere (acc 0.37–0.63) — the expected
  cross-domain drop. Synthetic augmentation does **not** fix domain shift
  (consistent with the Commun Med 2025 external-non-transfer caveat), but it
  robustly improves the target-domain model on the largest available external
  vote of confidence.

### C9. DINOv2 backbone — gate failed at 128×128 (resolution-limited)

Initial run hit a timm quirk (`vit_base_patch14_dinov2` defaults to 518 input
while the pipeline feeds 128) — fixed by `_resize_vit_pos_embed()` in
`train_dr_classifier.py` (pos-embed interpolated 37²→9², patch_embed
re-targeted, CLS kept; applied whenever img_size differs). The 3-seed gate then
ran on the frozen protocol. Same-seed comparison (111–113):

| backbone arm | acc | macro-F1 | QWK |
|---|---|---|---|
| densenet real_only | 0.8312 | 0.6904 | 0.9080 |
| densenet +blendA | **0.8415** | **0.7095** | **0.9193** |
| DINOv2 real_only | 0.6053 ± 0.081 | 0.3657 ± 0.112 | 0.6619 ± 0.019 |
| DINOv2 +blendA | 0.6430 ± 0.017 | 0.4191 ± 0.017 | 0.6795 ± 0.015 |

**Verdict: DINOv2 is ~0.20 acc / ~0.24 QWK below densenet on every arm; not
adopted.** At 128 the model has only 81 tokens (9×9 grid vs native 37×37), its
real-only baseline is unstable (acc std 0.08, mF1 std 0.11), and its augment
delta holds only on QWK (+3/−0) and g4 recall (+0.254, +3/−0). This reproduces
the pre-committed caution in `docs_private/LITERATURE_REVIEW.md` §1.3 that the
128 layout likely cannot reproduce the published protocol numbers. densenet121 +
blendA remains the headline; a higher-resolution DINOv2 re-test is parked (the
256×256 generator/data path exists — see DR256 config).

## Artifacts (public download)

Experiment share (anyone with the link):
`https://drive.google.com/drive/folders/1bFBIGMzQqDg7yCmamA4dpVXmUqQsbHI4`

| folder | contents | role |
|---|---|---|
| `generated_blendA/` | `generated.h5` (116 MB) | the frozen augmentation set (blendA: g0–3 synthetic ×1000, g4 real-only) |
| `generated_cfg4/` | `generated.h5` + `sample_grade_0..4.png` | blendA source for grades 0/1/4 (cond_scale 4.0) |
| `generated_cs1.5/` | `generated.h5` + `sample_grade_0..4.png` | blendA source for grades 2/3 (cond_scale 1.5) |
| `montages_cfg4/` | `all_grades_compare.png`, `real_vs_synthetic_grade_{0..4}.png`, `README_viewing_guide.md` | human-inspection montages (real top / synthetic bottom) |

Fetch the whole share (needs `gdown`):

```bash
gdown --folder https://drive.google.com/drive/folders/1bFBIGMzQqDg7yCmamA4dpVXmUqQsbHI4
```

These h5s are convenience copies; the "Regeneration" section below reproduces
each one from source (model + blend commands), so reviewers can regenerate
rather than trust the upload.

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