#!/bin/bash

# Random masking training for AVFF
# Binary classification (real/fake) objective

pretrain_path=/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/avff_mavos.pth

# Learning rate settings
lr=1e-5
head_lr=50
epoch=10
lrscheduler_start=5
lrscheduler_decay=0.5
lrscheduler_step=2

# Data normalization
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=False
batch_size=32

# Random masking ratio (0-1)
# 0.4 means 40% of tokens are masked
mask_ratio=0.4

n_print_steps=100

# Dataset paths
input_path=/mnt/d/projects/datasets/MAVOS-DD
tr_data=/mnt/d/projects/MAVOS-DD-GenClassifer/data/mavos-dd_train.csv
te_data=/mnt/d/projects/MAVOS-DD-GenClassifer/data/mavos-dd_validation.sv

# Output directory
save_dir=/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/adversarial_training_2_step_softmask_MINISET_03
mkdir -p $save_dir
mkdir -p ${save_dir}/models

cd ../src

CUDA_CACHE_DISABLE=1 python -W ignore ../src/run_ft_random_masking.py \
    --input_path ${input_path} \
    --target_length ${target_length} \
    --dataset_mean ${dataset_mean} \
    --dataset_std ${dataset_std} \
    --batch-size ${batch_size} \
    --lr ${lr} \
    --head_lr ${head_lr} \
    --n-epochs ${epoch} \
    --lrscheduler_start ${lrscheduler_start} \
    --lrscheduler_step ${lrscheduler_step} \
    --lrscheduler_decay ${lrscheduler_decay} \
    --mask_ratio ${mask_ratio} \
    --data-train ${tr_data} --data-val ${te_data} --save-dir $save_dir \
    --pretrain_path ${pretrain_path} \
    --n_print_steps ${n_print_steps} \
    --metrics mAP \
    --loss BCE \
    --miniset True
