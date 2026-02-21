#!/bin/bash
set -e

pretrain_path=/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/avff_mavos.pth

lr=1e-5
head_lr=50
epoch=10
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa_start=1
wa_end=10
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
batch_size=16
n_print_steps=50

tr_data=/mnt/d/projects/MAVOS-DD-GenClassifer/data/mavos-dd_train.csv
te_data=/mnt/d/projects/MAVOS-DD-GenClassifer/data/mavos-dd_validation.csv

save_dir=/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/contrastive_random_mask_MINISET
mkdir -p $save_dir
mkdir -p ${save_dir}/models

# Contrastive learning hyperparameters
TEMPERATURE=0.07
PROJECTION_DIM=128
SUPCON_WEIGHT=1.0
CLS_WEIGHT=1.0

# Random masking: same ratio as the learned masking experiment
MASK_RATIO=0.4

CUDA_CACHE_DISABLE=1 python -W ignore ../src/run_ft_contrastive_random_mask.py \
    --data-train ${tr_data} --data-val ${te_data} --save-dir $save_dir --n_classes 2 \
    --lr $lr --n-epochs ${epoch} --batch-size $batch_size \
    --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
    --loss BCE --metrics acc --warmup True \
    --wa_start ${wa_start} --wa_end ${wa_end} \
    --head_lr ${head_lr} \
    --pretrain_path ${pretrain_path} --num_workers 4 \
    --temperature $TEMPERATURE --projection_dim $PROJECTION_DIM --supcon_weight $SUPCON_WEIGHT \
    --cls_weight $CLS_WEIGHT \
    --n_print_steps ${n_print_steps} \
    --apply_mask True --mask_ratio $MASK_RATIO \
    --miniset True
