"""
Contrastive Learning Training Script for Deepfake Detection.

This script implements supervised contrastive learning for the MAVOS-DD dataset.
Instead of adversarial training (masking net vs classifier), it uses:
1. Supervised Contrastive Loss: Pulls together same-class samples, pushes apart different classes
2. Cross-Modal Contrastive Loss: Aligns audio-video pairs from the same sample
"""

import argparse
import os
import torch
import datasets
from torch.utils.data import DataLoader
from models.video_cav_mae import VideoCAVMAEContrastive
from traintest_ft import train_contrastive
import warnings

from mavosdd_dataset_multiclass import MavosDD
from mini_datasets import get_mini_train_set_deepfake_detection


parser = argparse.ArgumentParser(description='Video CAV-MAE Contrastive Learning Training')

# Data arguments
parser.add_argument('--data-train', type=str, help='path to train data csv')
parser.add_argument('--data-val', type=str, help='path to val data csv')
parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
parser.add_argument("--dataset_mean", default=-5.081, type=float, help="the dataset audio spec mean")
parser.add_argument("--dataset_std", default=4.4849, type=float, help="the dataset audio spec std")
parser.add_argument("--noise", default=False, type=bool, help="add noise to the input")

# Training arguments
parser.add_argument('--batch-size', default=32, type=int, help='batch size (larger is better for contrastive learning)')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument("--lr_patience", type=int, default=2, help="epochs to wait before reducing lr")
parser.add_argument("--metrics", type=str, default="mAP", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", choices=["BCE", "CE"])
parser.add_argument('--n-epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--n_classes', default=2, type=int, help='Num of classes (real/fake)')

# Model arguments
parser.add_argument('--save-dir', default='checkpoints/contrastive_training', type=str)
parser.add_argument('--pretrain_path', default=None, type=str, help='path to pretrain model')
parser.add_argument('--save_model', default=True)

# Contrastive learning specific arguments
parser.add_argument('--temperature', type=float, default=0.07, help='temperature for contrastive loss')
parser.add_argument('--projection_dim', type=int, default=128, help='dimension of projection head output')
parser.add_argument('--supcon_weight', type=float, default=1.0, help='weight for supervised contrastive loss')
parser.add_argument('--crossmodal_weight', type=float, default=0.5, help='weight for cross-modal contrastive loss')
parser.add_argument('--cls_weight', type=float, default=1.0, help='weight for classification loss')
parser.add_argument('--contrastive_weight', type=float, default=0.5, help='overall weight for contrastive losses')

# Learning rate scheduler
parser.add_argument("--lrscheduler_start", default=5, type=int)
parser.add_argument("--lrscheduler_step", default=3, type=int)
parser.add_argument("--lrscheduler_decay", default=0.5, type=float)
parser.add_argument('--warmup', type=bool, default=True)
parser.add_argument('--warmup_epochs', type=int, default=2)
parser.add_argument('--head_lr', type=int, default=10, help='lr multiplier for classifier head')

# Masking (optional, can be disabled for pure contrastive)
parser.add_argument('--apply_mask', type=bool, default=False, help='whether to apply learned masking')
parser.add_argument('--mask_ratio', type=float, default=0.4, help='ratio of tokens to mask')

# Other
parser.add_argument("--n_print_steps", default=100, type=int)
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)

parser.add_argument("--wa_start", type=int, default=1, help="epoch to start weight averaging")
parser.add_argument("--wa_end", type=int, default=10, help="epoch to end weight averaging")
parser.add_argument("--miniset", type=bool, default=False, help="use mini dataset for quick testing")

args = parser.parse_args()

im_res = 224
audio_conf = {
    'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mode': 'train',
    'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': args.noise, 'label_smooth': 0, 'im_res': im_res
}
val_audio_conf = {
    'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'mode': 'eval',
    'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res
}

print('='*60)
print('Contrastive Learning Configuration:')
print('='*60)
print(f'  - Temperature: {args.temperature}')
print(f'  - Projection dim: {args.projection_dim}')
print(f'  - SupCon weight: {args.supcon_weight}')
print(f'  - Cross-modal weight: {args.crossmodal_weight}')
print(f'  - Classification weight: {args.cls_weight}')
print(f'  - Overall contrastive weight: {args.contrastive_weight}')
print(f'  - Batch size: {args.batch_size}')
print(f'  - Apply masking: {args.apply_mask}')
print('='*60)

# Class mappings (used for logging, not for contrastive loss)
video_labels = {
    "memo": 0,
    "liveportrait": 1,
    "inswapper": 2,
    "echomimic": 3,
}
audio_labels = {
    "knnvc": 4,
    "freevc": 5,
    "openvoice": 6,
    "xtts_v2": 7,
    "yourtts": 8,
}
all_labels = {**video_labels, **audio_labels}

# Construct dataloader
input_path = "/mnt/d/projects/datasets/MAVOS-DD"

if args.miniset:
    print("Running with mini training set")
    mavos_dd = get_mini_train_set_deepfake_detection(input_path)

    train_loader = DataLoader(
        MavosDD(
            dataset=mavos_dd,
            input_path=input_path,
            audio_conf=audio_conf,
            video_class_name_to_idx=video_labels,
            audio_class_name_to_idx=audio_labels,
            stage=2),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True  # drop_last important for contrastive
    )

    val_loader = DataLoader(
        MavosDD(
            dataset=datasets.Dataset.load_from_disk(input_path).filter(lambda sample: sample['split']=="validation"),
            input_path=input_path,
            audio_conf=val_audio_conf,
            video_class_name_to_idx=video_labels,
            audio_class_name_to_idx=audio_labels,
            stage=2),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
else:
    print("Running with full training set")
    mavos_dd = datasets.Dataset.load_from_disk(input_path)

    train_loader = DataLoader(
        MavosDD(
            dataset=mavos_dd.filter(lambda sample: sample['split'] == "train"),
            input_path=input_path,
            audio_conf=audio_conf,
            stage=2,
            video_class_name_to_idx=video_labels,
            audio_class_name_to_idx=audio_labels,
        ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        MavosDD(
            dataset=mavos_dd.filter(lambda sample: sample['split'] == "validation"),
            input_path=input_path,
            audio_conf=val_audio_conf,
            stage=2,
            video_class_name_to_idx=video_labels,
            audio_class_name_to_idx=audio_labels),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Initialize model
model = VideoCAVMAEContrastive(
    n_classes=args.n_classes,
    temperature=args.temperature,
    projection_dim=args.projection_dim,
)

if not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

# Load pretrained weights if available
if args.pretrain_path is not None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl_weight = torch.load(args.pretrain_path, map_location=device)
    
    print(f'Running on {device}')
    
    miss, unexpected = model.load_state_dict(mdl_weight, strict=False)
    
    print('Missing keys: ', miss)
    print('Unexpected keys: ', unexpected)
    print('Loaded pretrain model from {:s}, missing: {:d}, unexpected: {:d}'.format(
        args.pretrain_path, len(miss), len(unexpected)))
else:
    warnings.warn("Training from scratch without pretrained weights.")

# Create experiment directory
print("\nCreating experiment directory: %s" % args.save_dir)
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'models'), exist_ok=True)

# Train
print("Starting contrastive learning for %d epochs" % args.n_epochs)
train_contrastive(model, train_loader, val_loader, args)
