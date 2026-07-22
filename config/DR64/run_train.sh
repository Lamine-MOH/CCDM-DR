#!/bin/bash

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

# Fast-iteration setup: use this at 64x64 to debug the pipeline and tune
# vicinity/imbalance hyperparameters cheaply before committing GPU time to
# the 128x128 config in ../DR128/run_train.sh. Fine lesion detail (micro-
# aneurysms, small hemorrhages) is hard to see at 64x64, so don't use this
# resolution for your final reported results -- 128 or 256 is more defensible
# for a DR-focused contribution.

DATA_NAME="DRGrading"
IMG_SIZE=64

ROOT_PATH="${1:?Usage: bash run_train.sh ROOT_PATH DATA_PATH [OPTIONS]}"
DATA_PATH="${2:?Usage: bash run_train.sh ROOT_PATH DATA_PATH [OPTIONS]}"
shift 2

# Defaults (fits 24GB GPU). Override via flags:
#   --num_steps N, --batch_size N, --grad_accum N, --samp_batch_size N
NUM_STEPS=60000
BATCH_SIZE=32
GRAD_ACCUM=4
SAMP_BATCH_SIZE=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_steps)       NUM_STEPS="$2"; shift 2 ;;
        --batch_size)      BATCH_SIZE="$2"; shift 2 ;;
        --grad_accum)      GRAD_ACCUM="$2"; shift 2 ;;
        --samp_batch_size) SAMP_BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SETTING="setup1_dr_fast"
SIGMA=-1.0
KAPPA=-1.0
TYPE="hard"

python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_name $DATA_NAME --data_path $DATA_PATH \
    --num_channels 3 --image_size $IMG_SIZE \
    --min_label 0 --max_label 4 \
    --model_config "./config/model_cfg/unet_edm_64_v1.yaml" \
    --y2h_embed_type "resnet" \
    --use_y2cov --y2cov_hy_weight_train 0.05 --y2cov_hy_weight_test 0.05 --y2cov_embed_type "resnet" --net_embed_y2cov_y2emb "cnn" \
    --train_num_steps $NUM_STEPS --resume_step 0 --train_lr 1e-4 \
    --train_batch_size $BATCH_SIZE --gradient_accumulate_every $GRAD_ACCUM \
    --train_amp --train_mixed_precision fp16 \
    --kernel_sigma $SIGMA --threshold_type $TYPE --kappa $KAPPA \
    --use_ada_vic --ada_vic_type vanilla --min_n_per_vic 50 --use_symm_vic \
    --sample_every 2000 --save_every 5000 \
    --sampler sde --num_sample_steps 32 \
    --sample_cond_scale 1.5 --sample_cond_rescaled_phi 0.7 \
    --nfake_per_label 500 --samp_batch_size $SAMP_BATCH_SIZE \
    --dump_fake_data \
    2>&1 | tee output_${DATA_NAME}_${IMG_SIZE}_${SETTING}.txt
