import argparse
import os
import random
import sys
import re
from typing import Dict, List, Tuple

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.mavosdd_dataset_multiclass import MavosDD, get_audio_label
from src.models.video_cav_mae import VideoCAVMAENoMaskMultiTask

VIDEO_METHOD_NAMES = ["real", "memo", "liveportrait", "inswapper", "echomimic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot t-SNE for latent features from a contrastive-trained VideoCAVMAE model"
    )

    # Data
    parser.add_argument("--checkpoint", type=str, required=True, help="The path to the model checkpoint (.pth)")
    parser.add_argument("--input_path", type=str, default="/mnt/d/projects/datasets/MAVOS-DD", help="Root path of input dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=2500, help="Maximum number of samples to embed")

    # Checkpoint/model
    parser.add_argument("--n_classes", type=int, default=len(VIDEO_METHOD_NAMES), help="Model classifier head classes in checkpoint")

    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.25)
    # parser.add_argument("--apply_mask", default=False, action="store_true", help="Apply learned masking during feature extraction")
    # parser.add_argument("--mask_ratio", type=float, default=0.4)

    # Which latent representation to use
    parser.add_argument(
        "--feature_type",
        type=str,
        default="fused_features",
        choices=["fused_features", "fusion_projection"],
        help="fused_features = penultimate latent (before classifier), fusion_projection = projected contrastive embedding",
    )

    parser.add_argument(
        "--label_mode",
        type=str,
        default="video_methods",
        choices=["binary", "video_methods", "video_audio_methods"],
        help="binary = real/fake, video_methods = one plot colored by all video methods in split, video_audio_methods = two plots",
    )

    # t-SNE params
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning_rate", type=float, default=200.0)
    parser.add_argument("--n_iter", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    return parser.parse_args()


def _build_dynamic_method_mappings(ds: datasets.Dataset) -> Tuple[Dict[str, int], Dict[str, int], List[str], List[str]]:
    """
    Build method->index mappings from the selected split itself.

    This allows visualising OOD methods that are not present in training mappings.
    """
    # Video methods (keep all non-real methods from this split)
    video_methods = sorted({m for m in ds["generative_method"] if m != "real"})

    # Audio methods normalized through get_audio_label and excluding real
    audio_methods_norm = []
    for m in ds["audio_generative_method"]:
        norm = get_audio_label(m)
        if norm != "real":
            audio_methods_norm.append(norm)
    audio_methods = sorted(set(audio_methods_norm))

    video_labels = {name: idx for idx, name in enumerate(video_methods)}
    audio_labels = {name: idx + len(video_labels) for idx, name in enumerate(audio_methods)}

    video_method_names = ["real"] + video_methods
    audio_method_names = ["real"] + audio_methods
    return video_labels, audio_labels, video_method_names, audio_method_names


def _sample_key(sample: Dict, label_mode: str) -> str:
    """Build a sampling key for balanced subset selection."""
    if label_mode == "binary":
        is_fake = (sample["generative_method"] != "real") or (get_audio_label(sample["audio_generative_method"]) != "real")
        return "fake" if is_fake else "real"
    if label_mode == "video_methods":
        return sample["generative_method"]
    # video_audio_methods
    return f"{sample['generative_method']}|{get_audio_label(sample['audio_generative_method'])}"


def _balanced_subsample(ds: datasets.Dataset, max_samples: int, label_mode: str, seed: int) -> datasets.Dataset:
    """
    Select up to `max_samples` with round-robin sampling across classes.

    This avoids bias from dataset ordering and improves class spread in t-SNE.
    """
    if max_samples is None or max_samples <= 0 or len(ds) <= max_samples:
        return ds

    buckets: Dict[str, List[int]] = {}
    for i, sample in enumerate(ds):
        key = _sample_key(sample, label_mode)
        buckets.setdefault(key, []).append(i)

    rng = random.Random(seed)
    keys = sorted(buckets.keys())
    for k in keys:
        rng.shuffle(buckets[k])

    picked: List[int] = []
    exhausted = False
    while len(picked) < max_samples and not exhausted:
        exhausted = True
        for k in keys:
            if buckets[k]:
                picked.append(buckets[k].pop())
                exhausted = False
                if len(picked) >= max_samples:
                    break

    return ds.select(picked)


def build_loader(args: argparse.Namespace) -> Tuple[DataLoader, int, int, List[str], List[str]]:
    audio_conf = {
        "num_mel_bins": 128,
        "target_length": 1024,
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "mode": "eval",
        "mean": -5.081,
        "std": 4.4849,
        "noise": False,
        "im_res": 224,
    }

    ds = datasets.Dataset.load_from_disk(args.input_path)
    ds = ds.filter(lambda sample: sample["split"] == args.split)

    # Enforce sample cap at dataset level with balanced sampling.
    ds = _balanced_subsample(ds, args.max_samples, args.label_mode, args.seed)

    video_labels, audio_labels, video_method_names, audio_method_names = _build_dynamic_method_mappings(ds)

    dataset = MavosDD(
        dataset=ds,
        input_path=args.input_path,
        audio_conf=audio_conf,
        stage=2,
        video_class_name_to_idx=video_labels,
        audio_class_name_to_idx=audio_labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader, len(video_labels), len(audio_labels), video_method_names, audio_method_names


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    regexp = re.compile(r'.*?_vgh-(?P<use_video_generative_head>(True)|(False))_.*')
    use_video_generative_head = regexp.match(args.checkpoint).group("use_video_generative_head") == "True"

    cavmae_ft = VideoCAVMAENoMaskMultiTask(
        n_binary_classes=2,
        n_video_gen_classes=args.n_classes,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        use_video_gen_head=use_video_generative_head
    )

    if not isinstance(cavmae_ft, torch.nn.DataParallel):
        cavmae_ft = torch.nn.DataParallel(cavmae_ft)
    cavmae_ft.eval()
    cavmae_ft.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = cavmae_ft.load_state_dict(ckpt, strict=False)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    print(f"Missing: {','.join(missing)}\n\nUnexpected: {','.join(unexpected)}")
    assert len(missing) == 0 and len(unexpected) == 0
    
    return cavmae_ft


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    # apply_mask: bool,
    # mask_ratio: float,
    feature_type: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feats = []
    labels = []
    paths = []
    gen_labels = []

    with torch.no_grad():
        for a_input, v_input, main_label, gen_label, video_path in tqdm(loader, desc="Extracting features"):
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)

            _, _, projections = model(
                a_input,
                v_input,
                # apply_mask=apply_mask,
                # hard_mask=False,
                # hard_mask_ratio=mask_ratio,
                return_projections=True,
            )

            if feature_type == "fused_features":
                batch_feat = projections["fused_features"]
            else:
                batch_feat = projections["fusion"]

            batch_feat = batch_feat.detach().cpu().numpy()

            # main_label = [is_fake, is_real], so class index 1 => fake, 0 => real
            batch_lbl = main_label[:, 0].long().cpu().numpy()

            feats.append(batch_feat)
            labels.append(batch_lbl)
            gen_labels.append(gen_label.cpu().numpy())
            paths.extend(video_path)

    if not feats:
        raise RuntimeError("No features extracted. Check dataset path/split and loader settings.")

    features = np.concatenate(feats, axis=0)
    label_arr = np.concatenate(labels, axis=0)
    gen_label_arr = np.concatenate(gen_labels, axis=0)

    # Hard consistency check: lengths must match exactly.
    if not (len(features) == len(label_arr) == len(gen_label_arr) == len(paths)):
        raise RuntimeError(
            "Inconsistent extracted batch lengths: "
            f"features={len(features)}, labels={len(label_arr)}, "
            f"gen_labels={len(gen_label_arr)}, paths={len(paths)}"
        )

    return features, label_arr, gen_label_arr, paths


def run_tsne(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    n = features.shape[0]
    # t-SNE requires perplexity < n_samples
    perplexity = min(args.perplexity, max(2.0, n - 1.0))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        random_state=args.seed,
        init="pca",
    )
    return tsne.fit_transform(features)

def decode_video_audio_method_labels(
    gen_labels: np.ndarray,
    n_video_methods: int,
    n_audio_methods: int,
    video_method_names: List[str],
    audio_method_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode multi-label vector into two categorical labels:
      - video method: real/memo/liveportrait/inswapper/echomimic
      - audio method: real/knnvc/freevc/openvoice/xtts_v2/yourtts
    """
    video_part = gen_labels[:, :n_video_methods]
    audio_part = gen_labels[:, n_video_methods:n_video_methods + n_audio_methods]

    # 0 means "real" for each modality; fake methods are shifted by +1
    video_idx = np.where(video_part.sum(axis=1) == 0, 0, video_part.argmax(axis=1) + 1)
    audio_idx = np.where(audio_part.sum(axis=1) == 0, 0, audio_part.argmax(axis=1) + 1)

    return video_idx, audio_idx


def _scatter_by_class(ax, z: np.ndarray, class_idx: np.ndarray, class_names: List[str], title: str) -> None:
    for i, name in enumerate(class_names):
        mask = class_idx == i
        if np.any(mask):
            ax.scatter(z[mask, 0], z[mask, 1], s=12, alpha=0.75, label=name)

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)


def plot_video_audio_method_embeddings(
    z: np.ndarray,
    gen_labels: np.ndarray,
    n_video_methods: int,
    n_audio_methods: int,
    video_method_names: List[str],
    audio_method_names: List[str],
    output_path: str,
    feature_type: str,
    split: str,
) -> None:
    video_idx, audio_idx = decode_video_audio_method_labels(
        gen_labels,
        n_video_methods=n_video_methods,
        n_audio_methods=n_audio_methods,
        video_method_names=video_method_names,
        audio_method_names=audio_method_names,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    _scatter_by_class(
        axes[0],
        z,
        video_idx,
        video_method_names,
        f"t-SNE by VIDEO method ({feature_type}) - split={split}",
    )
    _scatter_by_class(
        axes[1],
        z,
        audio_idx,
        audio_method_names,
        f"t-SNE by AUDIO method ({feature_type}) - split={split}",
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_binary_embeddings(z: np.ndarray, labels: np.ndarray, output_path: str, feature_type: str, split: str) -> None:
    plt.figure(figsize=(10, 8))

    real_mask = labels == 0
    fake_mask = labels == 1

    plt.scatter(z[real_mask, 0], z[real_mask, 1], s=12, alpha=0.75, label="real")
    plt.scatter(z[fake_mask, 0], z[fake_mask, 1], s=12, alpha=0.75, label="fake")

    plt.title(f"t-SNE of latent features ({feature_type}) - split={split}")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_video_method_embeddings(
    z: np.ndarray,
    gen_labels: np.ndarray,
    n_video_methods: int,
    video_method_names: List[str],
    output_path: str,
    feature_type: str,
    split: str,
) -> None:
    video_part = gen_labels[:, :n_video_methods]
    video_idx = np.where(video_part.sum(axis=1) == 0, 0, video_part.argmax(axis=1) + 1)

    plt.figure(figsize=(11, 9))
    _scatter_by_class(
        plt.gca(),
        z,
        video_idx,
        video_method_names,
        f"t-SNE by VIDEO method ({feature_type}) - split={split}",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader, n_video_methods, n_audio_methods, video_method_names, audio_method_names = build_loader(args)
    model = load_model(args, device)

    features, labels, gen_labels, _ = extract_features(
        model=model,
        loader=loader,
        device=device,
        # apply_mask=args.apply_mask,
        # mask_ratio=args.mask_ratio,
        feature_type=args.feature_type,
    )

    print(f"Collected features: {features.shape}")
    print(f"Class counts -> real: {(labels == 0).sum()}, fake: {(labels == 1).sum()}")

    z = run_tsne(features, args)

    dump_path = args.output
    if dump_path is None:
        root, _, _ = args.checkpoint.rsplit("/", 2)
        dump_path = f"{root}/eval/tsne_{args.feature_type}.png"

    if args.label_mode == "binary":
        plot_binary_embeddings(z, labels, dump_path, args.feature_type, args.split)
    elif args.label_mode == "video_methods":
        plot_video_method_embeddings(
            z,
            gen_labels,
            n_video_methods=n_video_methods,
            video_method_names=video_method_names,
            output_path=dump_path,
            feature_type=args.feature_type,
            split=args.split,
        )
    else:
        plot_video_audio_method_embeddings(
            z,
            gen_labels,
            n_video_methods=n_video_methods,
            n_audio_methods=n_audio_methods,
            video_method_names=video_method_names,
            audio_method_names=audio_method_names,
            output_path=dump_path,
            feature_type=args.feature_type,
            split=args.split,
        )

    print(f"Saved t-SNE plot to: {args.output}")


if __name__ == "__main__":
    main()
