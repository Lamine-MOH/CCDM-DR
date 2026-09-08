#!/usr/bin/env bash
# Runs the frozen downstream protocol (see docs/DOWNSTREAM_RESULTS.md) for a
# given classifier backbone: 5 real-only seeds + 5 blendA-augmented seeds.
#
# Hyperparameters are locked to the frozen protocol (epochs/batch/lr/img_size);
# only PAGE_BACKBONE and the run-name prefix change, so a second backbone is a
# drop-in robustness check instead of a new experiment.
#
# Usage:
#   bash analysis/run_downstream_protocol.sh BACKBONE PREFIX REAL_H5 TEST_H5
#
# Example (resnet50, results distinguishable from the frozen densenet121 runs):
#   bash analysis/run_downstream_protocol.sh resnet50 r50 \
#       data/DRGrading/Aptos/DRGrading_128x128_train.h5 \
#       data/DRGrading/Aptos/DRGrading_128x128_test.h5
#
# Env overrides:
#   BLEND_H5   synthetic h5 to augment with (default ./output/generated_blendA/generated.h5)
#   SYNTH_TAG  run-name token for the augmented condition
#             (default blendA; e.g. SYNTH_TAG=blendA_filt for filtered blends)
#   OUT_DIR    classifier results dir        (default ./downstream_results)
#   SEEDS      space-separated seed list      (default "111 112 113 114 115")
set -euo pipefail

if [ $# -ne 4 ]; then
    echo "usage: $0 BACKBONE PREFIX REAL_H5 TEST_H5" >&2
    exit 2
fi
BACKBONE=$1
PREFIX=$2
REAL_H5=$3
TEST_H5=$4

BLEND_H5=${BLEND_H5:-./output/generated_blendA/generated.h5}
SYNTH_TAG=${SYNTH_TAG:-blendA}
OUT_DIR=${OUT_DIR:-./downstream_results}
SEEDS=${SEEDS:-111 112 113 114 115}

[ -f "$REAL_H5" ] || { echo "missing real_h5: $REAL_H5" >&2; exit 1; }
[ -f "$TEST_H5" ] || { echo "missing test_h5: $TEST_H5" >&2; exit 1; }
[ -f "$BLEND_H5" ] || { echo "missing blend h5 (set BLEND_H5): $BLEND_H5" >&2; exit 1; }

run_one() {
    local run_name=$1; shift
    python downstream_eval/train_dr_classifier.py "$@"
    echo "[done] $run_name -> $OUT_DIR/${run_name}_metrics.json"
}

for s in $SEEDS; do
    RC="--real_h5 $REAL_H5 --test_h5 $TEST_H5 --backbone $BACKBONE"
    P="--out_dir $OUT_DIR --epochs 30 --batch_size 32 --lr 1e-4 --img_size 128 --num_workers 2"
    run_one "real_only_${PREFIX}_s$s" \
        $RC $P --run_name "real_only_${PREFIX}_s$s" --seed "$s"
    run_one "real_plus_synth_${SYNTH_TAG}_${PREFIX}_s$s" \
        $RC $P \
        --synthetic_h5 "$BLEND_H5" --synthetic_cap_per_grade 1000 \
        --run_name "real_plus_synth_${SYNTH_TAG}_${PREFIX}_s$s" --seed "$s"
done

echo "=========================================="
echo "protocol complete ($BACKBONE, prefix '$PREFIX')"
echo "aggregate with:"
echo "  python analysis/seed_report.py --results_dir $OUT_DIR \\"
echo "      --group real_only 'real_only_${PREFIX}_s111' 'real_only_${PREFIX}_s112' 'real_only_${PREFIX}_s113' 'real_only_${PREFIX}_s114' 'real_only_${PREFIX}_s115' \\"
echo "      --group augmented 'real_plus_synth_${SYNTH_TAG}_${PREFIX}_s111' 'real_plus_synth_${SYNTH_TAG}_${PREFIX}_s112' 'real_plus_synth_${SYNTH_TAG}_${PREFIX}_s113' 'real_plus_synth_${SYNTH_TAG}_${PREFIX}_s114' 'real_plus_synth_${SYNTH_TAG}_${PREFIX}_s115'"