#!/bin/bash

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

DATA_NAME="DRGrading"
IMG_SIZE=128

ROOT_PATH="<YOUR_PATH>"                     # path containing this repo's iCCDM code
DATA_PATH="<YOUR_PATH>/DRGrading"           # dir containing DRGrading_128x128.h5 (from build_dr_h5.py)

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
    --model_config "./config/model_cfg/unet_edm_128_v1.yaml" \
    --y2h_embed_type "resnet" \
    --use_y2cov --y2cov_hy_weight_train 0.05 --y2cov_hy_weight_test 0.05 --y2cov_embed_type "resnet" --net_embed_y2cov_y2emb "cnn" \
    --train_num_steps 150000 --resume_step 0 --train_lr 1e-5 \
    --train_batch_size 64 --gradient_accumulate_every 2 \
    --train_amp --train_mixed_precision fp16 \
    --kernel_sigma $SIGMA --threshold_type $TYPE --kappa $KAPPA \
    --use_ada_vic --ada_vic_type vanilla --min_n_per_vic 50 --use_symm_vic \
    --sample_every 2500 --save_every 10000 \
    --sampler sde --num_sample_steps 32 \
    --sample_cond_scale 1.5 --sample_cond_rescaled_phi 0.7 \
    --nfake_per_label 1000 --samp_batch_size 100 \
    --dump_fake_data \
    2>&1 | tee output_${DATA_NAME}_${IMG_SIZE}_${SETTING}.txt

    # --do_eval requires dataset-specific eval checkpoints you train yourself first
    # (see evaluation/evaluator.py comments + the experiment guide, Phase 3).
    # Add it back once ./evaluation/eval_ckpts/DRGrading/... exists:
    # --do_eval \
