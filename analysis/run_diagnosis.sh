#!/bin/bash
# ==============================================================================
# run_diagnosis.sh -- post-training diagnosis for the CCDM-DR grade conditioning
#
# Verifies whether the "fix the label conditioning" work actually succeeded:
#   A1. trace_conditioning : per-checkpoint RF grade-separability of the
#       `results/sample_{ode,sde}_<step>.png` grids vs a real-train reference.
#       (broken model plateaued ~0.17-0.44 flat; target = approach real ref)
#   A3. inspection montages : human-readable real-vs-synthetic per grade.
#   [A2 optional] synthetic-only classifier : QWK / grade-2 recall ceiling.
#
# Usage:
#   bash analysis/run_diagnosis.sh ROOT_PATH DATA_PATH [OPTIONS]
#     ROOT_PATH : repo root (where output/ lives)
#     DATA_PATH : dataset dir that contains DRGrading_128x128_{train,test}.h5
#
#   Options:
#     --setting NAME          training setting dir (default setup1_dr)
#     --img_size N            default 128
#     --synth_h5 PATH         generated.h5 (post-fix). Enables A3 montages.
#     --classifier            also run the synthetic-only classifier (A2, slow;
#                             needs GPU + 50 epochs)
#     --num_steps N           total steps for log filename (default 150000)
# ==============================================================================
set -euo pipefail

ROOT_PATH="${1:?Usage: run_diagnosis.sh ROOT_PATH DATA_PATH [OPTIONS]}"
DATA_PATH="${2:?Usage: run_diagnosis.sh ROOT_PATH DATA_PATH [OPTIONS]}"
shift 2

SETTING="setup1_dr"
IMG_SIZE=128
SYNTH_H5=""
RUN_CLF=0
NUM_STEPS=150000

while [[ $# -gt 0 ]]; do
    case $1 in
        --setting)   SETTING="$2"; shift 2 ;;
        --img_size)  IMG_SIZE="$2"; shift 2 ;;
        --synth_h5)  SYNTH_H5="$2"; shift 2 ;;
        --classifier) RUN_CLF=1; shift ;;
        --num_steps) NUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$ROOT_PATH"

DATA_NAME="DRGrading"
TRAIN_H5="$DATA_PATH/${DATA_NAME}_${IMG_SIZE}x${IMG_SIZE}_train.h5"
TEST_H5="$DATA_PATH/${DATA_NAME}_${IMG_SIZE}x${IMG_SIZE}_test.h5"
OUT_DIR="output/${DATA_NAME}_${IMG_SIZE}/${SETTING}"
GRID_DIR="$OUT_DIR/results"
DIAG_OUT="output/diagnosis_${DATA_NAME}_${IMG_SIZE}/${SETTING}"

echo "== root: $ROOT_PATH | data: $DATA_PATH | setting: $SETTING"
echo "== grid dir: $GRID_DIR | diag out: $DIAG_OUT"

# --- A1. conditioning trace --------------------------------------------------
[ ! -f "$TRAIN_H5" ] && { echo "!! missing real train h5: $TRAIN_H5"; exit 1; }
[ ! -d "$GRID_DIR" ] && { echo "!! no training-samples grids in $GRID_DIR (did training save sample_*_<step>.png?)"; exit 1; }

python analysis/trace_conditioning.py \
    --grid_dir "$GRID_DIR" \
    --out_dir "$DIAG_OUT/trace" \
    --real_h5 "$TRAIN_H5" \
    --max_label 4 --cell "$IMG_SIZE" --pad 1

echo
echo "===== A1 VERDICT (see $DIAG_OUT/trace/trace_report.md) ====="
tail -n 6 "$DIAG_OUT/trace/trace_report.md" | sed 's/^/    /'

# --- A3. human inspection montages (needs generated synthetic h5) ------------
MONT=""
if [ -n "$SYNTH_H5" ] && [ -f "$SYNTH_H5" ] && [ -f "$TEST_H5" ]; then
    MONT="$DIAG_OUT/montages"
    python analysis/render_inspection_montages.py \
        --real_h5 "$TRAIN_H5" \
        --synth_h5 "$SYNTH_H5" \
        --out_dir "$MONT" \
        --cell "$IMG_SIZE"
    echo "    A3 montages -> $MONT (view real_vs_synthetic_grade_{0..4}.png)"
else
    echo "    A3 skipped (pass --synth_h5 PATH/to/generated.h5)"
fi

# --- A2. synthetic-only classifier (optional, slow) --------------------------
if [ "$RUN_CLF" -eq 1 ]; then
    [ -z "$SYNTH_H5" ] && { echo "!! --classifier requires --synth_h5"; exit 1; }
    [ ! -f "$TEST_H5" ] && { echo "!! missing real test h5: $TEST_H5"; exit 1; }
    CLF_DIR="$DIAG_OUT/classifier"
    echo
    echo "===== A2 synthetic-only classifier (50 epochs) ====="
    python downstream_eval/train_dr_classifier.py \
        --real_h5 "$SYNTH_H5" \
        --test_h5 "$TEST_H5" \
        --backbone densenet121 --epochs 50 \
        --run_name synthetic_only_check \
        --out_dir "$CLF_DIR"
        python - <<'PY'
import json, glob, os
p = glob.glob(os.path.join("'$CLF_DIR'", "synthetic_only_check_metrics.json"))
if p:
    d = json.load(open(p[0]))
    print(f"    synthetic-only: acc={d['accuracy']:.3f} macro_f1={d['macro_f1']:.3f} "
          f"qwk={d['qwk']:.3f} grade2_recall={d['per_class_report']['grade_2']['recall']:.3f}")
print("    reference (real-only ~): acc 0.8415, macro_f1 0.7147, qwk 0.9137, g2 recall 0.80")
PY
fi

echo
echo "done. Full report at $DIAG_OUT"