#!/bin/bash
# Tier-4 retrain wrapper: augmentation + calibrated sigma_data + stronger CFG
# dropout + bf16/auto AMP + cosine LR. Uses a new setting name by default so
# Tier-3 checkpoints/results are preserved.
#
# Usage:
#   bash config/DR128/run_train_tier4.sh ROOT_PATH DATA_PATH [extra run_train.sh flags]
#
# Environment overrides (or pass the corresponding flag after DATA_PATH):
#   SETTING=setup1_dr_tier4
#   DR_AUG=hvflip                # none|hflip|hvflip|rot90|all
#   SIGMA_DATA_TYPE=global       # default|global|local (measure first)
#   COND_DROP_PROB=0.2
#   MIXED_PRECISION=auto         # no|fp16|bf16|auto
#   NUM_STEPS=100000
#   STOP_AT_STEP=0               # e.g. 50000 for the first phase; keep NUM_STEPS=100000
#   LR_SCHEDULE=cosine LR_WARMUP=1000 LR_MIN=0.1
#   SAVE_EVERY=10000

set -euo pipefail

ROOT_PATH="${1:?Usage: bash run_train_tier4.sh ROOT_PATH DATA_PATH [OPTIONS]}"
DATA_PATH="${2:?Usage: bash run_train_tier4.sh ROOT_PATH DATA_PATH [OPTIONS]}"
shift 2

SETTING="${SETTING:-setup1_dr_tier4}"
DR_AUG="${DR_AUG:-hvflip}"
SIGMA_DATA_TYPE="${SIGMA_DATA_TYPE:-global}"
COND_DROP_PROB="${COND_DROP_PROB:-0.2}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}"
NUM_STEPS="${NUM_STEPS:-100000}"
STOP_AT_STEP="${STOP_AT_STEP:-0}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LR_WARMUP="${LR_WARMUP:-1000}"
LR_MIN="${LR_MIN:-0.1}"
SAVE_EVERY="${SAVE_EVERY:-10000}"

exec bash "$(dirname "$0")/run_train.sh" "$ROOT_PATH" "$DATA_PATH" \
    --setting_name "$SETTING" \
    --dr_aug "$DR_AUG" \
    --edm_sigma_data_type "$SIGMA_DATA_TYPE" \
    --cond_drop_prob "$COND_DROP_PROB" \
    --train_mixed_precision "$MIXED_PRECISION" \
    --num_steps "$NUM_STEPS" \
    --stop_at_step "$STOP_AT_STEP" \
    --lr_schedule "$LR_SCHEDULE" --lr_warmup_steps "$LR_WARMUP" --lr_min_lr_frac "$LR_MIN" \
    --save_every "$SAVE_EVERY" \
    "$@"
