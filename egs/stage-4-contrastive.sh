#!/bin/bash
set -e

contrast_loss_weight=0.01
mae_loss_weight=1.0
norm_pix_loss=True

# you can use any checkpoints with a decoder, but by default, we use vision-MAE checkpoint
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
batch_size=9
lr_adapt=False

n_print_steps=100

tr_data=/mnt/d/projects/MAVOS-DD-GenClassifer/data/mavos-dd_train.csv
te_data=/mnt/d/projects/MAVOS-DD-GenClassifer/data/mavos-dd_validation.sv

# exp_dir=./exp/self-pretrain
# save_dir=/mnt/d/projects/MAVOS-DD-GenClassifer/exp/stage-3
save_dir=/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/contrastive_two_steps_adversarial_MINISET
mkdir -p $save_dir
mkdir -p ${save_dir}/models

# Training hyperparameters

# Contrastive learning specific
TEMPERATURE=0.07  # Lower = sharper distribution, typically 0.05-0.1
PROJECTION_DIM=128  # Dimension of projection head output
SUPCON_WEIGHT=1.0  # Weight for supervised contrastive loss
CROSSMODAL_WEIGHT=0.5  # Weight for audio-video alignment loss
CLS_WEIGHT=1.0  # Weight for classification loss
CONTRASTIVE_WEIGHT=0.5  # Overall weight for all contrastive losses

CUDA_CACHE_DISABLE=1 python -W ignore ../src/run_ft_contrastive.py \
    --data-train ${tr_data} --data-val ${te_data} --save-dir $save_dir --n_classes 2 \
    --lr $lr --n-epochs ${epoch} --batch-size $batch_size \
    --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
    --loss BCE --metrics acc --warmup True \
    --wa_start ${wa_start} --wa_end ${wa_end}  \
    --head_lr ${head_lr} \
    --pretrain_path ${pretrain_path} --num_workers 4 \
    --temperature $TEMPERATURE --projection_dim $PROJECTION_DIM --supcon_weight $SUPCON_WEIGHT \
    --crossmodal_weight $CROSSMODAL_WEIGHT --cls_weight $CLS_WEIGHT --contrastive_weight $CONTRASTIVE_WEIGHT \
    --n_print_steps 50 --apply_mask True --miniset True
