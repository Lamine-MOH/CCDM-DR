#!/bin/bash

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

DATA_NAME="DRGrading"
IMG_SIZE=256

ROOT_PATH="${1:?Usage: bash run_train.sh ROOT_PATH DATA_PATH [OPTIONS]}"
DATA_PATH="${2:?Usage: bash run_train.sh ROOT_PATH DATA_PATH [OPTIONS]}"
shift 2

# Defaults (fits 24GB GPU). The 256px UNet costs ~5-6x per image vs 128px
# (attn_resolutions [128,64,32]), so batch drops to 8 with grad-accum 8
# keeping the effective batch at 64 (same as DR128). Override via flags:
#   --num_steps N, --batch_size N, --grad_accum N, --samp_batch_size N
#   --resume_step N (resume from model-N.pt), --save_every N
#   --skip_final_sampling (end phase after training, no sampling/fake-data dump)
NUM_STEPS=150000
BATCH_SIZE=8
GRAD_ACCUM=8
SAMP_BATCH_SIZE=32
RESUME_STEP=0
SAVE_EVERY=10000
SKIP_FINAL_SAMPLING=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_steps)          NUM_STEPS="$2"; shift 2 ;;
        --batch_size)         BATCH_SIZE="$2"; shift 2 ;;
        --grad_accum)         GRAD_ACCUM="$2"; shift 2 ;;
        --samp_batch_size)    SAMP_BATCH_SIZE="$2"; shift 2 ;;
        --resume_step)        RESUME_STEP="$2"; shift 2 ;;
        --save_every)         SAVE_EVERY="$2"; shift 2 ;;
        --skip_final_sampling) SKIP_FINAL_SAMPLING=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SKIP_FLAG=""
if [ "$SKIP_FINAL_SAMPLING" -eq 1 ]; then
    SKIP_FLAG="--skip_final_sampling"
fi

SETTING="setup1_dr"
# Grades are discrete/ordinal (0-4) with typically severe class imbalance
# (grade 0 is the large majority, grade 4 is rare). These vicinity settings
# start conservative; retune kappa/min_n_per_vic once you've inspected your
# own class counts (see data_preparation/build_dr_h5.py printout).
SIGMA=-1.0
KAPPA=-1.0
TYPE="hard"

python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_name $DATA_NAME --data_path $DATA_PATH \
    --num_channels 3 --image_size $IMG_SIZE \
    --min_label 0 --max_label 4 \
    --model_config "./config/model_cfg/unet_edm_256_v1.yaml" \
    --y2h_embed_type "resnet" \
    --use_y2cov --y2cov_hy_weight_train 0.05 --y2cov_hy_weight_test 0.05 --y2cov_embed_type "resnet" --net_embed_y2cov_y2emb "cnn" \
    --train_num_steps $NUM_STEPS --resume_step $RESUME_STEP --train_lr 1e-5 \
    --train_batch_size $BATCH_SIZE --gradient_accumulate_every $GRAD_ACCUM \
    --train_amp --train_mixed_precision fp16 \
    --kernel_sigma $SIGMA --threshold_type $TYPE --kappa $KAPPA \
    --use_ada_vic --ada_vic_type vanilla --min_n_per_vic 50 --use_symm_vic \
    --sample_every 5000 --save_every $SAVE_EVERY \
    --sampler sde --num_sample_steps 32 \
    --sample_cond_scale 1.5 --sample_cond_rescaled_phi 0.7 \
    --nfake_per_label 1000 --samp_batch_size $SAMP_BATCH_SIZE \
    --dump_fake_data $SKIP_FLAG \
    2>&1 | tee output_${DATA_NAME}_${IMG_SIZE}_${SETTING}.txt

    # --do_eval requires dataset-specific eval checkpoints you train yourself first
    # (see evaluation/evaluator.py comments + the experiment guide, Phase 3).
    # Add it back once ./evaluation/eval_ckpts/DRGrading/... exists:
    # --do_eval \
