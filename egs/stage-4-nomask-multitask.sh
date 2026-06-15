#!/bin/bash
# Stage 4 – No-mask multi-task ablations
#
# Toggle-based setup:
#   use_video_gen_head=False,use_contrastive=False -> binary only
#   use_video_gen_head=True,use_contrastive=False  -> binary + video-gen head
#   use_video_gen_head=True,use_contrastive=True   -> binary + video-gen + supcon
#   use_video_gen_head=False,use_contrastive=True   -> binary + supcon

set -e

# you can use any compatible checkpoint, by default we keep the AVFF one
pretrain_path=/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/avff_mavos.pth

# direct toggles
use_video_gen_head=False
use_contrastive=True
contrastive_mode='generative_methods'

lr=1e-5
head_lr=10
epoch=10
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
batch_size=32

n_print_steps=100

# loss weights
BINARY_WEIGHT=1.0
VIDEO_GEN_WEIGHT=1.0
SUPCON_WEIGHT=1.0

# model hyperparameters
TEMPERATURE=0.25
PROJECTION_DIM=128

save_dir=/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/nomask_multitask_vgh-${use_video_gen_head}_supcon-${use_contrastive}-${contrastive_mode}
mkdir -p $save_dir
mkdir -p ${save_dir}/models

CUDA_CACHE_DISABLE=1 python -W ignore ../src/run_ft_nomask_multitask.py \
    --use_video_gen_head ${use_video_gen_head} --use_contrastive ${use_contrastive} --contrastive_mode ${contrastive_mode} --save-dir $save_dir \
    --lr $lr --head_lr $head_lr --n-epochs ${epoch} --batch-size $batch_size \
    --num_workers 4 --metrics acc \
    --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
    --temperature $TEMPERATURE --projection_dim $PROJECTION_DIM \
    --binary_weight $BINARY_WEIGHT --video_gen_weight $VIDEO_GEN_WEIGHT --supcon_weight $SUPCON_WEIGHT \
    --pretrain_path ${pretrain_path} \
    --n_print_steps ${n_print_steps} --save_model True --miniset True
