"""
Video Generative Method Classification Training Script.

Reuses VideoCAVMAEContrastive but targets the *visual* generative method instead of
real/fake binary detection.

Classification head: 5 classes
    0 = real video
    1 = memo
    2 = liveportrait
    3 = inswapper
    4 = echomimic
"""

import argparse
import os
import torch
import datasets
from torch.utils.data import DataLoader
from models.video_cav_mae import VideoCAVMAEContrastive
from traintest_ft import train_contrastive_video_gen
import warnings

from mavosdd_dataset_multiclass import MavosDD
from mini_datasets import get_mini_train_set_deepfake_detection


parser = argparse.ArgumentParser(
    description='Video Generative Method Contrastive Classification')

# Data arguments
parser.add_argument('--data-train', type=str, help='path to train data csv (unused – HF dataset)')
parser.add_argument('--data-val', type=str, help='path to val data csv (unused – HF dataset)')
parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
parser.add_argument("--dataset_mean", default=-5.081, type=float)
parser.add_argument("--dataset_std", default=4.4849, type=float)
parser.add_argument("--noise", default=False, type=bool)

# Training arguments
parser.add_argument('--batch-size', default=32, type=int,
                    help='batch size (larger is better for contrastive learning)')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument("--lr_patience", type=int, default=2)
parser.add_argument("--metrics", type=str, default="acc", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="CE", choices=["BCE", "CE"])
parser.add_argument('--n-epochs', default=20, type=int)

# Model arguments — 5 classes: real + 4 video generative methods
parser.add_argument('--n_classes', default=5, type=int,
                    help='Number of classes: real + 4 video generative methods')
parser.add_argument('--save-dir', default='checkpoints/contrastive_video_gen', type=str)
parser.add_argument('--pretrain_path', default=None, type=str)
parser.add_argument('--save_model', default=True)

# Contrastive learning arguments
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--supcon_weight', type=float, default=1.0,
                    help='weight for supervised contrastive loss (supcon_weight)')
parser.add_argument('--lambda_adv', type=float, default=0.1,
                    help='weight for adversarial loss that fools the discriminator (lambda_adv)')
parser.add_argument('--cls_weight', type=float, default=1.0,
                    help='weight for CE classification loss')

# LR scheduler
parser.add_argument("--lrscheduler_start", default=5, type=int)
parser.add_argument("--lrscheduler_step", default=3, type=int)
parser.add_argument("--lrscheduler_decay", default=0.5, type=float)
parser.add_argument('--warmup', type=bool, default=True)
parser.add_argument('--warmup_epochs', type=int, default=2)
parser.add_argument('--head_lr', type=int, default=10,
                    help='lr multiplier for the classifier + projector heads')

# Masking (optional)
parser.add_argument('--apply_mask', type=bool, default=True,
                    help='enable learned token masking (adversarial training)')
parser.add_argument('--mask_ratio', type=float, default=0.4)

# Other
parser.add_argument("--n_print_steps", default=100, type=int)
parser.add_argument('--freqm', type=int, default=0)
parser.add_argument('--timem', type=int, default=0)
parser.add_argument("--wa_start", type=int, default=1)
parser.add_argument("--wa_end", type=int, default=10)
parser.add_argument("--miniset", type=bool, default=False,
                    help="use mini dataset for quick testing")

args = parser.parse_args()

def load_pretrained_allow_classifier_head_mismatch(model, checkpoint_path, device):
    """
    Load pretrained weights conservatively.

    Allowed to be missing/mismatched: classification head, projection heads, masking net.
    Any other mismatch/missing/unexpected key raises RuntimeError.
    """
    mdl_weight = torch.load(checkpoint_path, map_location=device)

    # Support checkpoints that wrap weights in a top-level key
    if isinstance(mdl_weight, dict) and 'state_dict' in mdl_weight:
        mdl_weight = mdl_weight['state_dict']

    # Conservative loading policy with explicit allow-list by module prefix.
    # Allowed missing/mismatch prefixes:
    #   - mlp_head (classification head)
    #   - fusion_projector / gen_method_projector (projection heads)
    #   - masking_net
    # Everything else is strict.
    model_state = model.state_dict()
    filtered_weight = {}
    allowed_prefixes = (
        'module.mlp_head.', 'mlp_head.',
        'module.fusion_projector.', 'fusion_projector.',
        'module.gen_method_projector.', 'gen_method_projector.',
        'module.masking_net.', 'masking_net.',
    )
    skipped_allowed_mismatch = []
    unmatched_ckpt_keys_disallowed = []

    def is_allowed_key(key: str) -> bool:
        return key.startswith(allowed_prefixes)

    for k, v in mdl_weight.items():
        # Support checkpoints saved with and without DataParallel 'module.' prefix.
        candidate_keys = [k]
        if k.startswith('module.'):
            candidate_keys.append(k[len('module.'):])
        else:
            candidate_keys.append(f'module.{k}')

        matched_key = None
        for ck in candidate_keys:
            if ck in model_state:
                matched_key = ck
                break

        if matched_key is None:
            if is_allowed_key(k):
                continue
            unmatched_ckpt_keys_disallowed.append(k)
            continue

        if model_state[matched_key].shape != v.shape:
            if is_allowed_key(matched_key):
                skipped_allowed_mismatch.append((matched_key, tuple(v.shape), tuple(model_state[matched_key].shape)))
                continue
            raise RuntimeError(
                f"Unexpected shape mismatch for key '{matched_key}': "
                f"ckpt{tuple(v.shape)} vs model{tuple(model_state[matched_key].shape)}"
            )

        filtered_weight[matched_key] = v

    if len(unmatched_ckpt_keys_disallowed) > 0:
        preview = unmatched_ckpt_keys_disallowed[:10]
        raise RuntimeError(
            "Checkpoint contains unexpected keys not present in model. "
            f"First {len(preview)} keys: {preview}"
        )

    miss, unexpected = model.load_state_dict(filtered_weight, strict=False)

    unexpected_nonempty = [k for k in unexpected]
    if len(unexpected_nonempty) > 0:
        raise RuntimeError(f"Unexpected keys during loading: {unexpected_nonempty}")

    missing_disallowed = [k for k in miss if not is_allowed_key(k)]
    if len(missing_disallowed) > 0:
        raise RuntimeError(
            "Missing keys are not limited to allowed prefixes "
            "(masking_net/projection heads/classification head). "
            f"Disallowed missing keys: {missing_disallowed}"
        )

    return miss, unexpected, skipped_allowed_mismatch

im_res = 224
audio_conf = {
    'num_mel_bins': 128, 'target_length': args.target_length,
    'freqm': args.freqm, 'timem': args.timem, 'mode': 'train',
    'mean': args.dataset_mean, 'std': args.dataset_std,
    'noise': args.noise, 'label_smooth': 0, 'im_res': im_res,
}
val_audio_conf = {
    'num_mel_bins': 128, 'target_length': args.target_length,
    'freqm': 0, 'timem': 0, 'mixup': 0, 'mode': 'eval',
    'mean': args.dataset_mean, 'std': args.dataset_std,
    'noise': False, 'im_res': im_res,
}

# Video methods: indices 0-3 in gen_label
# Audio methods: indices 4-8 in gen_label
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

N_VIDEO_CLASSES = len(video_labels)   # 4  →  5-class head (+ real)

print('=' * 60)
print('Video Generative Method Classification Configuration:')
print('=' * 60)
print(f'  - Num classes: {args.n_classes} (0=real, 1-4=video methods)')
print(f'  - Temperature: {args.temperature}')
print(f'  - Projection dim: {args.projection_dim}')
print(f'  - SupCon weight: {args.supcon_weight}')
print(f'  - Adversarial weight: {args.lambda_adv}')
print(f'  - Classification weight: {args.cls_weight}')
print(f'  - Batch size: {args.batch_size}')
print(f'  - Apply masking: {args.apply_mask}')
print('=' * 60)

input_path = "/mnt/d/projects/datasets/MAVOS-DD"

if args.miniset:
    print("Running with mini training set")
    mavos_dd_mini = get_mini_train_set_deepfake_detection(input_path)

    train_loader = DataLoader(
        MavosDD(
            dataset=mavos_dd_mini,
            input_path=input_path,
            audio_conf=audio_conf,
            video_class_name_to_idx=video_labels,
            audio_class_name_to_idx=audio_labels,
            stage=2,
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = DataLoader(
        MavosDD(
            dataset=datasets.Dataset.load_from_disk(input_path).filter(
                lambda s: s['split'] == "validation"),
            input_path=input_path,
            audio_conf=val_audio_conf,
            video_class_name_to_idx=video_labels,
            audio_class_name_to_idx=audio_labels,
            stage=2,
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
else:
    print("Running with full training set")
    mavos_dd = datasets.Dataset.load_from_disk(input_path)

    train_loader = DataLoader(
        MavosDD(
            dataset=mavos_dd.filter(lambda s: s['split'] == "train"),
            input_path=input_path,
            audio_conf=audio_conf,
            video_class_name_to_idx=video_labels,
            audio_class_name_to_idx=audio_labels,
            stage=2,
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = DataLoader(
        MavosDD(
            dataset=mavos_dd.filter(lambda s: s['split'] == "validation"),
            input_path=input_path,
            audio_conf=val_audio_conf,
            video_class_name_to_idx=video_labels,
            audio_class_name_to_idx=audio_labels,
            stage=2,
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# n_classes=5: the MLP head outputs 5 logits (real + 4 video methods)
model = VideoCAVMAEContrastive(
    n_classes=args.n_classes,
    temperature=args.temperature,
    projection_dim=args.projection_dim,
)

if not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

if args.pretrain_path is not None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')
    miss, unexpected, skipped_head_mismatch = load_pretrained_allow_classifier_head_mismatch(
        model=model,
        checkpoint_path=args.pretrain_path,
        device=device,
    )

    print('Missing keys: ', miss)
    print('Unexpected keys: ', unexpected)
    if len(skipped_head_mismatch) > 0:
        print('Skipped allowed mismatches (masking/projection/classification heads):')
        for k, old_shape, new_shape in skipped_head_mismatch:
            print(f'  - {k}: ckpt{old_shape} -> model{new_shape}')
    print('Loaded pretrain model from {:s}, missing: {:d}, unexpected: {:d}'.format(
        args.pretrain_path, len(miss), len(unexpected)))
else:
    warnings.warn("Training from scratch without pretrained weights.")

print("\nCreating experiment directory: %s" % args.save_dir)
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'models'), exist_ok=True)

print("Starting video-gen contrastive training for %d epochs" % args.n_epochs)
train_contrastive_video_gen(
    model, train_loader, val_loader, args,
    n_video_classes=N_VIDEO_CLASSES,
)
