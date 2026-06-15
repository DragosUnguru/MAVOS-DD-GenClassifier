"""
Run no-mask multi-task ablations from VideoCAVMAE-style backbone.

Experiments:
  1) Binary head only
  2) Binary + video generative method head
  3) Binary + video generative method head + supervised contrastive loss
"""

import argparse
import os
import warnings

import datasets
import torch
from torch.utils.data import DataLoader

from mavosdd_dataset_multiclass import MavosDD
from mini_datasets import get_mini_train_set_deepfake_detection
from models.video_cav_mae import VideoCAVMAENoMaskMultiTask
from traintest_ft import train_nomask_multitask


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_pretrained_flexible(model, checkpoint_path, device):
    """Load matching keys only (by name + shape), skip task-specific mismatches."""
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    model_state = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state.items():
        candidates = [k]
        if k.startswith('module.'):
            candidates.append(k[len('module.'):])
        else:
            candidates.append(f'module.{k}')

        mk = None
        for c in candidates:
            if c in model_state:
                mk = c
                break
        if mk is None:
            continue
        if model_state[mk].shape != v.shape:
            skipped.append((mk, tuple(v.shape), tuple(model_state[mk].shape)))
            continue
        filtered[mk] = v

    miss, unexpected = model.load_state_dict(filtered, strict=False)
    return miss, unexpected, skipped


parser = argparse.ArgumentParser(description='No-mask multi-task ablation training')

# Data
parser.add_argument('--target_length', default=1024, type=int)
parser.add_argument('--dataset_mean', default=-5.081, type=float)
parser.add_argument('--dataset_std', default=4.4849, type=float)
parser.add_argument('--noise', default=False, type=bool)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--miniset', type=bool, default=False)

# Training
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--head_lr', default=10, type=int)
parser.add_argument('--n-epochs', default=20, type=int)
parser.add_argument('--metrics', type=str, default='acc', choices=['mAP', 'acc'])
parser.add_argument('--save-dir', default='checkpoints/nomask_multitask', type=str)
parser.add_argument('--save_model', default=True)
parser.add_argument('--pretrain_path', default=None, type=str)
parser.add_argument('--n_print_steps', default=100, type=int)

# Model
parser.add_argument('--n_binary_classes', default=2, type=int)
parser.add_argument('--n_video_gen_classes', default=5, type=int)  # real + 4 in-domain methods
parser.add_argument('--projection_dim', default=128, type=int)
parser.add_argument('--temperature', default=0.07, type=float)

# Direct feature toggles
parser.add_argument('--use_video_gen_head', type=str2bool, default=False,
                    help='Enable video generative-method auxiliary head')
parser.add_argument('--use_contrastive', type=str2bool, default=False,
                    help='Enable supervised contrastive loss')
parser.add_argument('--contrastive_mode', choices=['real_fake', 'generative_methods'], default='real_fake',
                    help='What the supervised loss follows: the real/fake classification or the different video generative methods (both ignore same-video-gen mehtod pairs)')

# Loss weights
parser.add_argument('--binary_weight', default=1.0, type=float)
parser.add_argument('--video_gen_weight', default=1.0, type=float)
parser.add_argument('--supcon_weight', default=1.0, type=float)

# Scheduler
parser.add_argument('--lrscheduler_start', default=5, type=int)
parser.add_argument('--lrscheduler_step', default=3, type=int)
parser.add_argument('--lrscheduler_decay', default=0.5, type=float)

# Audio aug
parser.add_argument('--freqm', type=int, default=0)
parser.add_argument('--timem', type=int, default=0)

args = parser.parse_args()

im_res = 224
audio_conf = {
    'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm,
    'timem': args.timem, 'mode': 'train', 'mean': args.dataset_mean,
    'std': args.dataset_std, 'noise': args.noise, 'label_smooth': 0, 'im_res': im_res
}
val_audio_conf = {
    'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0,
    'timem': 0, 'mixup': 0, 'mode': 'eval', 'mean': args.dataset_mean,
    'std': args.dataset_std, 'noise': False, 'im_res': im_res
}

video_labels = {
    'memo': 0,
    'liveportrait': 1,
    'inswapper': 2,
    'echomimic': 3,
}
audio_labels = {
    'knnvc': 4,
    'freevc': 5,
    'openvoice': 6,
    'xtts_v2': 7,
    'yourtts': 8,
}

input_path = '/mnt/d/projects/datasets/MAVOS-DD'
if args.miniset:
    mavos_dd = get_mini_train_set_deepfake_detection(input_path)
    val_ds = datasets.Dataset.load_from_disk(input_path).filter(lambda s: s['split'] == 'validation')
else:
    full = datasets.Dataset.load_from_disk(input_path)
    mavos_dd = full.filter(lambda s: s['split'] == 'train')
    val_ds = full.filter(lambda s: s['split'] == 'validation')

train_loader = DataLoader(
    MavosDD(
        dataset=mavos_dd,
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
        dataset=val_ds,
        input_path=input_path,
        audio_conf=val_audio_conf,
        stage=2,
        video_class_name_to_idx=video_labels,
        audio_class_name_to_idx=audio_labels,
    ),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True
)

print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
print(f'use_video_gen_head={args.use_video_gen_head}, use_contrastive={args.use_contrastive}')

model = VideoCAVMAENoMaskMultiTask(
    n_binary_classes=args.n_binary_classes,
    n_video_gen_classes=args.n_video_gen_classes,
    use_video_gen_head=args.use_video_gen_head,
    temperature=args.temperature,
    projection_dim=args.projection_dim,
)

if not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

if args.pretrain_path is not None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    miss, unexpected, skipped = load_pretrained_flexible(model, args.pretrain_path, device)
    print(f'Loaded pretrain from {args.pretrain_path}')
    print(f'Missing keys: {len(miss)} | Unexpected keys: {len(unexpected)} | Skipped shape mismatch: {len(skipped)}')
else:
    warnings.warn('Training from scratch without pretrained weights.')

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'models'), exist_ok=True)

train_nomask_multitask(
    model,
    train_loader,
    val_loader,
    args,
    n_video_classes=len(video_labels),
)
