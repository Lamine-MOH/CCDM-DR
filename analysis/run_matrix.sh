#!/bin/bash
# run_matrix.sh — post-retrain orchestration: generation, cs sweep, blends,
# downstream classifier matrix, and reporting.
#
# Usage:
#   bash analysis/run_matrix.sh ROOT_PATH DATA_PATH --model_ckpt /path/to/model-*.pt [OPTIONS]
#
# ROOT_PATH     training/artifacts root (embeddings + generated sets live under
#               <ROOT>/output/...; downstream_results under <ROOT>/).
# DATA_PATH     dir containing DRGrading_<S>x<S>_{train,test}.h5.
# --model_ckpt  REQUIRED direct path to the diffusion checkpoint (model-<step>.pt).
#
# Stages (all idempotent — rerunning skips what already exists):
#   1. gen       F1/F2/F3: all grades 0-4 @ cond_scale 4.0/2.5/1.5, nfake/grade
#   2. sweep     uni_cs1.5/2.5/4.0 classifier runs on the sensitivity backbone
#                at the FULL seed set (5 seeds) — the pre-blend cond_scale sweep
#   3. blends    frozen recipe (blendA g4 real-only, blendA_plus_g4, blendA_cap1500,
#                blendA_g3cs4 probe) + blend_sel.h5 from analysis/select_blend_cs.py
#                (mean VAL per-grade recall across the sweep, source per grade from
#                its winning cs)
#   4. eval      headline {real_only, blendA, blendA_g3cs4, blend_sel} x headline
#                backbones x seeds; blendA_plus_g4 + cap variants at the explore
#                seed (promote via --promote for the full seed set)
#   5. report    compare_runs.py + per-backbone seed_report.py
#
# --factorial mode (Exp 5: clean sampler x cond_scale grid, no blends):
#   1. gen       {--sampler sde|ode} x --cses, all grades 0-4
#   2. eval      every (sampler, cs) pool as a uniform augmentation arm
#                (real + pool capped at --cap, all 5 grades synthetic) over EVERY
#                backbone in --headline-backbones x --seeds, plus real_only per
#                backbone. Arm name: uni_{sampler}_cs{cs} (e.g. uni_sde_cs1.0).
#                Order: real_only, then sde cs1..4, then ode cs1..4, per backbone.
#   3. report    analysis/sampler_cs_report.py (2x4 grid + main effects) +
#                compare_runs.py. Blends (stage 3) are skipped.
#   Intended sequential use, one backbone per invocation so densenet121's full
#   grid lands first (every stage is resumable — existing outputs are skipped):
#     bash analysis/run_matrix.sh ROOT DATA --model_ckpt CKPT --factorial \
#         --sampler sde --sampler ode --cses "1.0 2.0 3.0 4.0" \
#         --results_subdir Exp5_sampler_cs --headline-backbones densenet121
#     ... then resnet50, then efficientnet_b4
#
# Optional flags:
#   --model_config PATH          (default <REPO>/config/model_cfg/unet_edm_128_v1.yaml)
#   --img_size N                 (default 128)
#   --nfake_per_grade N          (default 2000; must be >= largest cap used)
#   --gen-batch N                (default 100)
#   --epochs N                   (default 30)
#   --lr X                       (default 1e-4)
#   --cls-batch N                (default 32)
#   --num-workers N              (default 2)
#   --headline-backbones "..."   (default "densenet121 resnet50 efficientnet_b4")
#   --sensitivity-backbone B     (default densenet121)
#   --seeds "s1 s2 ..."          (default "111 112 113 114 115")
#   --explore-seed S             (default 111)
#   --promote "arm ..."          arms that also get the full seed set (default: none)
#   --train-h5 PATH / --test-h5 PATH
#   --gen                        only generation + blends (no classifier)
#   --no-gen / --no-blend / --no-eval
#   --gpu N                      (default 0)
#   --factorial                  Exp 5 sampler x cond_scale grid (see above)
#   --sampler S                  repeatable; default "sde". In --factorial mode
#                                every listed sampler is generated + evaluated.
#   --cses "c1 c2 ..."           (default "1.5 2.5 4.0")
#   --cap N                      per-grade synthetic cap (default 1000)
#   --results_subdir NAME        nest the results under downstream_results/NAME
#   --gen-seed N                 RNG seed for generation (default 111)
#   --reseed-per-grade           seed once per grade (seed + grade) so two
#                                samplers share init noise per grade
#   --dry-run                    print the commands without running them

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ROOT_PATH="${1:?Usage: bash analysis/run_matrix.sh ROOT_PATH DATA_PATH --model_ckpt PATH [OPTIONS]}"
DATA_PATH="${2:?Usage: bash analysis/run_matrix.sh ROOT_PATH DATA_PATH --model_ckpt PATH [OPTIONS]}"
shift 2

IMG_SIZE=128
MODEL_CKPT=""
MODEL_CONFIG="${REPO}/config/model_cfg/unet_edm_128_v1.yaml"
NFAKE=2000
GEN_BATCH=100
EPOCHS=30
LR=1e-4
CLS_BATCH=32
NUM_WORKERS=2
HEADLINE_BACKBONES="densenet121 resnet50 efficientnet_b4"
SENS_BACKBONE="densenet121"
SEEDS="111 112 113 114 115"
EXPLORE_SEED=111
PROMOTE=""
TRAIN_H5=""
TEST_H5=""
GPU=0
DO_GEN=1
DO_BLEND=1
DO_EVAL=1
DRY_RUN=0
FACTORIAL=0
SAMPLERS=""
CSES="1.5 2.5 4.0"
CAP=1000
RESULTS_SUBDIR=""
GEN_SEED=111
RESEED_PER_GRADE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_ckpt)          MODEL_CKPT="$2"; shift 2 ;;
        --model_config)        MODEL_CONFIG="$2"; shift 2 ;;
        --img_size)            IMG_SIZE="$2"; shift 2 ;;
        --nfake_per_grade)     NFAKE="$2"; shift 2 ;;
        --gen-batch)           GEN_BATCH="$2"; shift 2 ;;
        --epochs)              EPOCHS="$2"; shift 2 ;;
        --lr)                  LR="$2"; shift 2 ;;
        --cls-batch)           CLS_BATCH="$2"; shift 2 ;;
        --num-workers)         NUM_WORKERS="$2"; shift 2 ;;
        --headline-backbones)  HEADLINE_BACKBONES="$2"; shift 2 ;;
        --sensitivity-backbone) SENS_BACKBONE="$2"; shift 2 ;;
        --seeds)               SEEDS="$2"; shift 2 ;;
        --explore-seed)        EXPLORE_SEED="$2"; shift 2 ;;
        --promote)             PROMOTE="$2"; shift 2 ;;
        --train-h5)            TRAIN_H5="$2"; shift 2 ;;
        --test-h5)             TEST_H5="$2"; shift 2 ;;
        --gpu)                 GPU="$2"; shift 2 ;;
        --gen)                 DO_EVAL=0; shift ;;
        --no-gen)              DO_GEN=0; shift ;;
        --no-blend)            DO_BLEND=0; shift ;;
        --no-eval)             DO_EVAL=0; shift ;;
        --factorial)           FACTORIAL=1; shift ;;
        --sampler)             SAMPLERS="${SAMPLERS:+$SAMPLERS }$2"; shift 2 ;;
        --cses)                CSES="$2"; shift 2 ;;
        --cap)                 CAP="$2"; shift 2 ;;
        --results_subdir)      RESULTS_SUBDIR="$2"; shift 2 ;;
        --gen-seed)            GEN_SEED="$2"; shift 2 ;;
        --reseed-per-grade)    RESEED_PER_GRADE=1; shift ;;
        --dry-run)             DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done


if [ -z "$MODEL_CKPT" ]; then
    echo "ERROR: --model_ckpt is required (direct path to model-<step>.pt)." >&2
    echo "       The diffusion checkpoint is NOT tied to ROOT_PATH anymore." >&2
    echo "       Copy model_y2h/ + model_y2cov/ + edm_sigma_data.json along with it." >&2
    exit 1
fi

TRAIN_H5="${TRAIN_H5:-${DATA_PATH}/DRGrading_${IMG_SIZE}x${IMG_SIZE}_train.h5}"
TEST_H5="${TEST_H5:-${DATA_PATH}/DRGrading_${IMG_SIZE}x${IMG_SIZE}_test.h5}"
GEN_DIR="${ROOT_PATH}/output"
BLEND_DIR="${ROOT_PATH}/output/blends"
RESULTS_DIR="${ROOT_PATH}/downstream_results"
[ -n "$RESULTS_SUBDIR" ] && RESULTS_DIR="${RESULTS_DIR}/${RESULTS_SUBDIR}"
# legacy (non-factorial) mode is the sde pipeline; --factorial without an
# explicit --sampler would otherwise have no pools to build.
[ -z "$(echo "$SAMPLERS" | tr -d ' ')" ] && SAMPLERS="sde"

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] $*"
        return 0
    fi
    echo "  > $*"
    "$@"
}

pmessage() {
    echo ""
    echo "================================================================================"
    echo " $1"
    echo "================================================================================"
}

check_file() {
    local f="$1" what="$2"
    if [ ! -f "$f" ]; then
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  [dry-run] WARNING: $what not found: $f" >&2
            return 0
        fi
        echo "ERROR: $what not found: $f" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Stage 1 — generation passes
#   legacy (default): F1 (cs4.0), F2 (cs2.5), F3 (cs1.5), all grades, sampler sde,
#                     written to output/generated_cs<cs>/ (unchanged paths)
#   --factorial:      {--sampler} x --cses -> output/generated_<sampler>_cs<cs>/
# ---------------------------------------------------------------------------
# Output dir for a (sampler, cond_scale) pool. The legacy sde layout is kept
# verbatim so the pre-existing pools are reused, never regenerated.
gen_dirname() {
    local sampler="$1" cs="$2"
    if [ "$FACTORIAL" -eq 0 ] && [ "$sampler" = "sde" ]; then
        echo "generated_cs${cs}"
    else
        echo "generated_${sampler}_cs${cs}"
    fi
}

gen_pass() {
    local sampler="$1" cs="$2"
    local out="${GEN_DIR}/$(gen_dirname "$sampler" "$cs")"
    local h5="${out}/generated.h5"
    if [ -f "$h5" ]; then
        echo "  [skip] $h5 exists"
        return 0
    fi
    check_file "$MODEL_CKPT" "diffusion checkpoint (--model_ckpt)"
    check_file "$MODEL_CONFIG" "model yaml (--model_config)"
    mkdir -p "$out"
    local seed_flag=()
    [ "$RESEED_PER_GRADE" -eq 1 ] && seed_flag=(--reseed_per_grade)
    run python generate_from_ckpt.py \
        --model_ckpt "$MODEL_CKPT" \
        --model_config "$MODEL_CONFIG" \
        --root_path "$ROOT_PATH" \
        --image_size "$IMG_SIZE" \
        --cond_scale "$cs" \
        --rescaled_phi 0.7 \
        --sampler "$sampler" --num_sample_steps 32 \
        --grades 0 1 2 3 4 \
        --nfake_per_grade "$NFAKE" \
        --batch_size "$GEN_BATCH" \
        --seed "$GEN_SEED" ${seed_flag[@]+"${seed_flag[@]}"} \
        --use_y2cov --y2cov_hy_weight_train 0.05 --y2cov_hy_weight_test 0.05 \
        --out_dir "$out"
}

if [ "$DO_GEN" -eq 1 ]; then
    if [ "$FACTORIAL" -eq 1 ]; then
        pmessage "Stage 1/3 — generation: samplers {${SAMPLERS}} x cond_scales {${CSES}}, all grades"
    else
        pmessage "Stage 1/5 — generation: F1 (cs4.0), F2 (cs2.5), F3 (cs1.5), all grades"
    fi
    if [ "$FACTORIAL" -eq 1 ]; then
        for s in $SAMPLERS; do
            for cs in $CSES; do
                gen_pass "$s" "$cs"
            done
        done
    else
        gen_pass "sde" "4.0"
        gen_pass "sde" "2.5"
        gen_pass "sde" "1.5"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 2 — classifier cs sweep (uni_cs arms, full seed set)
# ---------------------------------------------------------------------------
classifier() {
    local backbone="$1" arm="$2" seed="$3" cap="$4" synth="$5"
    local run_name="${backbone}_${arm}_s${seed}"
    if [ -f "${RESULTS_DIR}/${run_name}_metrics.json" ]; then
        echo "  [skip] ${run_name}_metrics.json exists"
        return 0
    fi
    local cmd=(python downstream_eval/train_dr_classifier.py
        --real_h5 "$TRAIN_H5"
        --test_h5 "$TEST_H5"
        --backbone "$backbone"
        --epochs "$EPOCHS"
        --lr "$LR"
        --batch_size "$CLS_BATCH"
        --img_size "$IMG_SIZE"
        --seed "$seed"
        --run_name "$run_name"
        --out_dir "$RESULTS_DIR"
        --num_workers "$NUM_WORKERS")
    if [ -n "$synth" ]; then
        cmd+=(--synthetic_h5 "$synth" --synthetic_cap_per_grade "$cap")
    fi
    run "${cmd[@]}"
}

sweep_cs() {
    local cs="$1"
    for s in $SEEDS; do
        classifier "$SENS_BACKBONE" "uni_cs${cs}" "$s" 1000 "${GEN_DIR}/generated_cs${cs}/generated.h5"
    done
}

# --- factorial mode: real_only + every (sampler, cond_scale) pool, per backbone
factorial_arm_spec() {
    # uni_<sampler>_cs<cond_scale> -> "<h5>:<cap>"
    local arm="$1" sampler cs
    sampler="${arm#uni_}"
    cs="${sampler##*_cs}"
    sampler="${sampler%_cs*}"
    echo "${GEN_DIR}/$(gen_dirname "$sampler" "$cs")/generated.h5:${CAP}"
}

factorial_eval() {
    local backbone="$1"
    for s in $SEEDS; do
        classifier "$backbone" "real_only" "$s" "$CAP" ""
    done
    for sampler in $SAMPLERS; do
        for cs in $CSES; do
            local arm="uni_${sampler}_cs${cs}"
            check_file "${GEN_DIR}/$(gen_dirname "$sampler" "$cs")/generated.h5" \
                "factorial pool (generate first)"
            for s in $SEEDS; do
                local spec="$(factorial_arm_spec "$arm")"
                classifier "$backbone" "$arm" "$s" "${spec#*:}" "${spec%%:*}"
            done
        done
    done
}

if [ "$DO_EVAL" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    mkdir -p "$RESULTS_DIR"
    if [ "$FACTORIAL" -eq 1 ]; then
        pmessage "Stage 2/3 — factorial: real_only + {${SAMPLERS}} x {${CSES}} (cap ${CAP}/grade) x {${HEADLINE_BACKBONES}} x {$SEEDS}"
        for backbone in $HEADLINE_BACKBONES; do
            factorial_eval "$backbone"
        done
    else
        pmessage "Stage 2/5 — cs sweep: uni_cs1.5/2.5/4.0 x {${SENS_BACKBONE}} x {$SEEDS}"
        for cs in 1.5 2.5 4.0; do
            check_file "${GEN_DIR}/generated_cs${cs}/generated.h5" "sweep source (generate first or copy from TM)"
            sweep_cs "$cs"
        done
    fi
fi

# ---------------------------------------------------------------------------
# Stage 3 — blend assembly
# ---------------------------------------------------------------------------
blend() {
    local out="$1" cap="$2" override="$3"
    local sources="${4:-0=${GEN_DIR}/generated_cs4.0/generated.h5 1=${GEN_DIR}/generated_cs4.0/generated.h5 4=${GEN_DIR}/generated_cs4.0/generated.h5 2=${GEN_DIR}/generated_cs1.5/generated.h5 3=${GEN_DIR}/generated_cs1.5/generated.h5}"
    local h5="${BLEND_DIR}/${out}.h5"
    if [ -f "$h5" ]; then
        echo "  [skip] $h5 exists"
        return 0
    fi
    local ov=""
    [ -n "$override" ] && ov="--caps_override $override"
    run python analysis/merge_h5_by_grade.py \
        --sources "$sources" \
        --out "$h5" \
        --cap "$cap" \
        $ov
}

if [ "$FACTORIAL" -eq 1 ]; then
    echo "  [skip] Stage 3 blends — not part of the sampler x cond_scale factorial."
elif [ "$DO_BLEND" -eq 1 ]; then
    pmessage "Stage 3/5 — blends: frozen recipe + data-driven blend_sel"
    mkdir -p "$BLEND_DIR"
    if [ "$DO_GEN" -eq 1 ]; then
        check_file "${GEN_DIR}/generated_cs4.0/generated.h5" "F1 output"
        check_file "${GEN_DIR}/generated_cs1.5/generated.h5" "F3 output"
    fi
    blend "blendA"        1000 "4=0"
    blend "blendA_plus_g4" 1000 ""
    blend "blendA_cap1500" 1500 "4=0"
    blend "blendA_g3cs4"   1000 "4=0" "0=${GEN_DIR}/generated_cs4.0/generated.h5 1=${GEN_DIR}/generated_cs4.0/generated.h5 4=${GEN_DIR}/generated_cs4.0/generated.h5 2=${GEN_DIR}/generated_cs1.5/generated.h5 3=${GEN_DIR}/generated_cs4.0/generated.h5"

    if [ "$DO_EVAL" -eq 1 ] && [ -f "${RESULTS_DIR}/${SENS_BACKBONE}_uni_cs4.0_s${EXPLORE_SEED}_metrics.json" ]; then
        pools=""
        for cs in 1.5 2.5 4.0; do
            pools="${pools} ${cs}=${GEN_DIR}/generated_cs${cs}/generated.h5"
        done
        run python analysis/select_blend_cs.py \
            --results_dir "$RESULTS_DIR" \
            --backbone "$SENS_BACKBONE" \
            --seeds "$SEEDS" \
            --pools "$pools" \
            --out "${BLEND_DIR}/blend_sel.h5" \
            --cap 1000
    else
        echo "  [skip] blend_sel assembly (sweep results missing or --no-eval);" \
             "run the sweep first or copy pre-built blend_sel.h5"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 4 — headline + sensitivity classifier runs
# ---------------------------------------------------------------------------
arm_synth() {
    local arm="$1"
    case "$arm" in
        real_only)          echo "" ;;
        blendA)             echo "${BLEND_DIR}/blendA.h5:1000" ;;
        blendA_g3cs4)       echo "${BLEND_DIR}/blendA_g3cs4.h5:1000" ;;
        blendA_plus_g4)     echo "${BLEND_DIR}/blendA_plus_g4.h5:1000" ;;
        blendA_cap500)      echo "${BLEND_DIR}/blendA.h5:500" ;;
        blendA_cap1500)     echo "${BLEND_DIR}/blendA_cap1500.h5:1500" ;;
        blend_sel)          echo "${BLEND_DIR}/blend_sel.h5:1000" ;;
        uni_cs1.5)          echo "${GEN_DIR}/generated_cs1.5/generated.h5:1000" ;;
        uni_cs2.5)          echo "${GEN_DIR}/generated_cs2.5/generated.h5:1000" ;;
        uni_cs4.0)          echo "${GEN_DIR}/generated_cs4.0/generated.h5:1000" ;;
        *) echo "unknown arm: $arm" >&2; exit 1 ;;
    esac
}

is_promoted() {
    local arm="$1"
    for a in $PROMOTE; do
        [ "$a" = "$arm" ] && return 0
    done
    return 1
}

if [ "$DO_EVAL" -eq 1 ] && [ "$FACTORIAL" -eq 0 ]; then
    pmessage "Stage 4/5 — headline arms x {${HEADLINE_BACKBONES}}"
    for backbone in $HEADLINE_BACKBONES; do
        for arm in real_only blendA blendA_g3cs4 blend_sel; do
            if [ "$arm" = "blendA_g3cs4" ] && [ ! -f "${BLEND_DIR}/blendA_g3cs4.h5" ]; then
                echo "  [skip] blendA_g3cs4 arm: ${BLEND_DIR}/blendA_g3cs4.h5 missing"
                continue
            fi
            if [ "$arm" = "blend_sel" ] && [ ! -f "${BLEND_DIR}/blend_sel.h5" ]; then
                echo "  [skip] blend_sel arm: ${BLEND_DIR}/blend_sel.h5 missing"
                continue
            fi
            for s in $SEEDS; do
                spec="$(arm_synth "$arm")"
                classifier "$backbone" "$arm" "$s" "${spec#*:}" "${spec%%:*}"
            done
        done
        for arm in blendA_plus_g4; do
            local_seeds="$EXPLORE_SEED"
            if is_promoted "$arm"; then
                local_seeds="$SEEDS"
            fi
            for s in $local_seeds; do
                spec="$(arm_synth "$arm")"
                classifier "$backbone" "$arm" "$s" "${spec#*:}" "${spec%%:*}"
            done
        done
    done

    pmessage "Stage 4/5 — sensitivity arms x {${SENS_BACKBONE}} (seed ${EXPLORE_SEED})"
    for arm in blendA_cap500 blendA_cap1500; do
        local_seeds="$EXPLORE_SEED"
        if is_promoted "$arm"; then
            local_seeds="$SEEDS"
        fi
        for s in $local_seeds; do
            spec="$(arm_synth "$arm")"
            classifier "$SENS_BACKBONE" "$arm" "$s" "${spec#*:}" "${spec%%:*}"
        done
    done

    # -----------------------------------------------------------------------
    # Stage 5 — reporting
    # -----------------------------------------------------------------------
    pmessage "Stage 5/5 — reporting"
    run python downstream_eval/compare_runs.py --results_dir "$RESULTS_DIR"

    for backbone in $HEADLINE_BACKBONES; do
        all_present=1
        for s in $SEEDS; do
            [ -f "${RESULTS_DIR}/${backbone}_real_only_s${s}_metrics.json" ] || all_present=0
            [ -f "${RESULTS_DIR}/${backbone}_blendA_s${s}_metrics.json" ] || all_present=0
            [ -f "${RESULTS_DIR}/${backbone}_blendA_g3cs4_s${s}_metrics.json" ] || all_present=0
            [ -f "${RESULTS_DIR}/${backbone}_blend_sel_s${s}_metrics.json" ] || all_present=0
        done
        [ "$all_present" -eq 1 ] || continue
        extra=(
            --group real_only     $(for s in $SEEDS; do echo "${backbone}_real_only_s${s}"; done)
            --group blendA        $(for s in $SEEDS; do echo "${backbone}_blendA_s${s}"; done)
            --group blendA_g3cs4  $(for s in $SEEDS; do echo "${backbone}_blendA_g3cs4_s${s}"; done)
            --group blend_sel     $(for s in $SEEDS; do echo "${backbone}_blend_sel_s${s}"; done)
        )
        for arm in $PROMOTE; do
            ok=1
            for s in $SEEDS; do
                [ -f "${RESULTS_DIR}/${backbone}_${arm}_s${s}_metrics.json" ] || ok=0
            done
            [ "$ok" -eq 1 ] && extra+=(--group "$arm" $(for s in $SEEDS; do echo "${backbone}_${arm}_s${s}"; done))
        done
        run python analysis/seed_report.py --results_dir "$RESULTS_DIR" "${extra[@]}"
    done
fi

# ---------------------------------------------------------------------------
# Factorial reporting — the 2x4 sampler x cond_scale grid + main effects
# ---------------------------------------------------------------------------
if [ "$FACTORIAL" -eq 1 ] && [ "$DO_EVAL" -eq 1 ]; then
    pmessage "Stage 3/3 — reporting: sampler x cond_scale grid"
    run python analysis/sampler_cs_report.py \
        --results_dir "$RESULTS_DIR" \
        --backbones "$HEADLINE_BACKBONES" \
        --seeds "$SEEDS" \
        --samplers $SAMPLERS \
        --cses $CSES \
        --out_csv "${RESULTS_DIR}/sampler_cs_grid.csv"
    run python downstream_eval/compare_runs.py --results_dir "$RESULTS_DIR"
fi

pmessage "Done. Artifacts under:"
if [ "$FACTORIAL" -eq 1 ]; then
    for s in $SAMPLERS; do
        for cs in $CSES; do
            echo "  generated : $GEN_DIR/$(gen_dirname "$s" "$cs")/generated.h5"
        done
    done
else
    echo "  generated : $GEN_DIR/generated_cs{4.0,2.5,1.5}/generated.h5"
    echo "  blends    : $BLEND_DIR/{blendA,blendA_g3cs4,blendA_plus_g4,blendA_cap1500,blend_sel}.h5"
fi
echo "  results   : $RESULTS_DIR/  (run compare_runs.py to refresh the table)"
