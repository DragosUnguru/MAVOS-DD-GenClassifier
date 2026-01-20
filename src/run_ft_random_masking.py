"""
Training script for random masking experiments.

Uses gradient-preserving random masking on video embeddings.
Binary classification objective (real/fake) using mavosdd_dataset.py.
No adversarial components or learned masking.
"""
import argparse
import os
import torch
import datasets
from torch.utils.data import DataLoader
from models.video_cav_mae import VideoCAVMAEFT
from traintest_ft import train_random_masking
import warnings

from mavosdd_dataset import MavosDD
from mini_datasets import get_mini_test_set, get_mini_train_set_deepfake_detection


parser = argparse.ArgumentParser(description='Video CAV-MAE with Random Masking')

# Data arguments
parser.add_argument('--data-train', type=str, help='path to train data csv')
parser.add_argument('--data-val', type=str, help='path to val data csv')
parser.add_argument('--input_path', type=str, default='/mnt/d/projects/datasets/MAVOS-DD', help='path to dataset')
parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
parser.add_argument("--dataset_mean", default=-5.081, type=float, help="the dataset audio spec mean")
parser.add_argument("--dataset_std", default=4.4849, type=float, help="the dataset audio spec std")
parser.add_argument("--noise", default=False, type=bool, help="add noise to the input")

# Training arguments
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument("--metrics", type=str, default="mAP", help="main evaluation metrics", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="loss function", choices=["BCE", "CE"])
parser.add_argument('--n-epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--n_classes', default=2, type=int, help='Num of classes (2 for real/fake)')

# Model/checkpoint arguments
parser.add_argument('--save-dir', default='checkpoints/random_masking_binary', type=str, help='directory to save checkpoints')
parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrain model')
parser.add_argument('--save_model', default=True, type=bool)

# Learning rate scheduler
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="step to decrease lr")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="lr decay ratio")
parser.add_argument('--head_lr', type=int, default=50, help='multiplier for MLP head lr')

# Random masking arguments
parser.add_argument('--mask_ratio', type=float, default=0.75, help='ratio of tokens to mask (0-1)')

# Misc
parser.add_argument("--n_print_steps", default=100, type=int)
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument('--miniset', action='store_true', help='use mini dataset')

args = parser.parse_args()

# Audio configuration
im_res = 224
audio_conf = {
    'num_mel_bins': 128, 
    'target_length': args.target_length, 
    'freqm': args.freqm, 
    'timem': args.timem, 
    'mode': 'train',
    'mean': args.dataset_mean, 
    'std': args.dataset_std, 
    'noise': args.noise, 
    'label_smooth': 0, 
    'im_res': im_res
}
val_audio_conf = {
    'num_mel_bins': 128, 
    'target_length': args.target_length, 
    'freqm': 0, 
    'timem': 0, 
    'mixup': 0, 
    'mode': 'eval',
    'mean': args.dataset_mean, 
    'std': args.dataset_std, 
    'noise': False, 
    'im_res': im_res
}

print('='*60)
print('Random Masking Experiment Configuration')
print('='*60)
print(f'Mask ratio: {args.mask_ratio}')
print(f'Learning rate: {args.lr}')
print(f'Batch size: {args.batch_size}')
print(f'Epochs: {args.n_epochs}')
print('='*60)

# Construct dataloader
input_path = args.input_path

if args.miniset:
    print("Using mini dataset...")
    mavos_dd = get_mini_train_set_deepfake_detection(input_path)
    train_loader = DataLoader(
        MavosDD(
            mavos_dd,
            input_path,
            audio_conf,
            stage=2,
            custom_file_path=False),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
else:
    print("Using full dataset...")
    mavos_dd = datasets.Dataset.load_from_disk(input_path)
    train_loader = DataLoader(
        MavosDD(
            mavos_dd.filter(lambda sample: sample['split'] == "train"),
            input_path,
            audio_conf,
            stage=2,
            custom_file_path=False),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

val_loader = DataLoader(
    MavosDD(
        datasets.Dataset.load_from_disk(input_path).filter(lambda sample: sample['split'] == "validation"),
        input_path,
        val_audio_conf,
        stage=2,
        custom_file_path=False),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Load pre-trained AVFF model
cavmae_ft = VideoCAVMAEFT(n_classes=args.n_classes)
if not isinstance(cavmae_ft, torch.nn.DataParallel):
    cavmae_ft = torch.nn.DataParallel(cavmae_ft)

if args.pretrain_path is not None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl_weight = torch.load(args.pretrain_path, map_location=device)
    print(f'Running on {device}')
    
    miss, unexpected = cavmae_ft.load_state_dict(mdl_weight, strict=False)
    print('Missing: ', miss)
    print('Unexpected: ', unexpected)
    print('Loaded pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(
        args.pretrain_path, len(miss), len(unexpected)))
else:
    warnings.warn("Note: finetuning a model without pretrained weights.")

# Create experiment directory
print("\nCreating experiment directory: %s" % args.save_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if not os.path.exists(os.path.join(args.save_dir, 'models')):
    os.makedirs(os.path.join(args.save_dir, 'models'))

# Train model
print("Starting random masking training for %d epochs" % args.n_epochs)
train_random_masking(cavmae_ft, train_loader, val_loader, args)
