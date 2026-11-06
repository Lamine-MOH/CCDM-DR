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
#   --resume_step N (resume from model-N.pt), --save_every N
#   --stop_at_step N (stop this phase while keeping --num_steps as the LR-schedule total)
#   --skip_final_sampling (end phase after training, no sampling/fake-data dump)
#   --lr_schedule cosine --lr_warmup_steps 1000 --lr_min_lr_frac 0.1 (opt-in LR decay;
#     off by default, i.e. constant LR, which is the EDM reference recipe)
#   --dr_aug none|hflip|hvflip|rot90|all
#   --edm_sigma_data_type default|global|local
#   --train_mixed_precision no|fp16|bf16|auto
#   --cond_drop_prob F --setting_name NAME
NUM_STEPS=60000
BATCH_SIZE=32
GRAD_ACCUM=4
SAMP_BATCH_SIZE=100
RESUME_STEP=0
STOP_AT_STEP=0
SAVE_EVERY=5000
SKIP_FINAL_SAMPLING=0
LR_SCHEDULE=constant
LR_WARMUP=0
LR_MIN_FRAC=0.0
TRAIN_FRAC=1.0
TRAIN_SUBSET_SEED=111
TRAIN_SUBSET_SCOPE=diffusion
SETTING=setup1_dr_fast
DR_AUG=none
SIGMA_DATA_TYPE=default
MIXED_PRECISION=fp16
COND_DROP_PROB=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_steps)          NUM_STEPS="$2"; shift 2 ;;
        --batch_size)         BATCH_SIZE="$2"; shift 2 ;;
        --grad_accum)         GRAD_ACCUM="$2"; shift 2 ;;
        --samp_batch_size)    SAMP_BATCH_SIZE="$2"; shift 2 ;;
        --resume_step)        RESUME_STEP="$2"; shift 2 ;;
        --stop_at_step)       STOP_AT_STEP="$2"; shift 2 ;;
        --save_every)         SAVE_EVERY="$2"; shift 2 ;;
        --lr_schedule)        LR_SCHEDULE="$2"; shift 2 ;;
        --lr_warmup_steps)    LR_WARMUP="$2"; shift 2 ;;
        --lr_min_lr_frac)     LR_MIN_FRAC="$2"; shift 2 ;;
        --train_frac)         TRAIN_FRAC="$2"; shift 2 ;;
        --train_subset_seed)  TRAIN_SUBSET_SEED="$2"; shift 2 ;;
        --train_subset_scope) TRAIN_SUBSET_SCOPE="$2"; shift 2 ;;
        --setting_name)       SETTING="$2"; shift 2 ;;
        --dr_aug)             DR_AUG="$2"; shift 2 ;;
        --edm_sigma_data_type) SIGMA_DATA_TYPE="$2"; shift 2 ;;
        --train_mixed_precision) MIXED_PRECISION="$2"; shift 2 ;;
        --cond_drop_prob)     COND_DROP_PROB="$2"; shift 2 ;;
        --skip_final_sampling) SKIP_FINAL_SAMPLING=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SKIP_FLAG=""
if [ "$SKIP_FINAL_SAMPLING" -eq 1 ]; then
    SKIP_FLAG="--skip_final_sampling"
fi

COND_DROP_FLAG=""
if [ -n "$COND_DROP_PROB" ]; then
    COND_DROP_FLAG="--cond_drop_prob $COND_DROP_PROB"
fi

SIGMA=-1.0
KAPPA=-1.0
TYPE="hard"

# Console/tee log lives in the run folder, not the repo root.  Create it
# before the pipeline so tee can open the file even if main.py fails early.
RUN_OUT_DIR="${ROOT_PATH%/}/output/${DATA_NAME}_${IMG_SIZE}/${SETTING}"
mkdir -p "$RUN_OUT_DIR"
CONSOLE_LOG="${RUN_OUT_DIR}/output_${DATA_NAME}_${IMG_SIZE}_${SETTING}.txt"

python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_name $DATA_NAME --data_path $DATA_PATH \
    --num_channels 3 --image_size $IMG_SIZE \
    --min_label 0 --max_label 4 \
    --model_config "./config/model_cfg/unet_edm_64_v1.yaml" \
    --y2h_embed_type "resnet" \
    --use_y2cov --y2cov_hy_weight_train 0.05 --y2cov_hy_weight_test 0.05 --y2cov_embed_type "resnet" --net_embed_y2cov_y2emb "cnn" \
    --epoch_cnn_embed 200 --epoch_cnn_embed_y2cov 200 --epoch_net_y2h 500 --epoch_net_y2cov 500 \
    --batch_size_embed 256 --batch_size_embed_y2cov 256 \
    --num_img_per_label_after_replica 1000 \
    --train_frac $TRAIN_FRAC --train_subset_seed $TRAIN_SUBSET_SEED --train_subset_scope $TRAIN_SUBSET_SCOPE \
    --train_num_steps $NUM_STEPS --stop_at_step $STOP_AT_STEP --resume_step $RESUME_STEP --train_lr 1e-4 \
    --lr_schedule $LR_SCHEDULE --lr_warmup_steps $LR_WARMUP --lr_min_lr_frac $LR_MIN_FRAC \
    --edm_sigma_data_type $SIGMA_DATA_TYPE \
    --dr_aug $DR_AUG \
    $COND_DROP_FLAG \
    --train_batch_size $BATCH_SIZE --gradient_accumulate_every $GRAD_ACCUM \
    --train_amp --train_mixed_precision $MIXED_PRECISION \
    --kernel_sigma $SIGMA --threshold_type $TYPE --kappa $KAPPA \
    --use_ada_vic --ada_vic_type vanilla --min_n_per_vic 50 --use_symm_vic \
    --sample_every 2000 --save_every $SAVE_EVERY \
    --sampler sde --num_sample_steps 32 \
    --sample_cond_scale 1.5 --sample_cond_rescaled_phi 0.7 \
    --nfake_per_label 500 --samp_batch_size $SAMP_BATCH_SIZE \
    --dump_fake_data $SKIP_FLAG \
    2>&1 | tee "$CONSOLE_LOG"
