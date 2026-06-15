import argparse
import json
import re

import datasets
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.mavosdd_dataset_multiclass import MavosDD
from src.models.video_cav_mae import VideoCAVMAENoMaskMultiTask


VIDEO_METHOD_NAMES = ["real", "memo", "liveportrait", "inswapper", "echomimic"]
DATASET_INPUT_PATH = "/mnt/d/projects/datasets/MAVOS-DD"

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for VideoCAVMAENoMaskMultiTask")

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dump_path", type=str, required=True)

    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--dataset_mean", type=float, default=-5.081)
    parser.add_argument("--dataset_std", type=float, default=4.4849)
    parser.add_argument("--target_length", type=int, default=1024)

    # Model args
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--projection_dim", type=int, default=128)

    return parser.parse_args()


def load_checkpoint_flexible(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Try direct load first
    try:
        miss, unexp = model.load_state_dict(state, strict=False)
        return miss, unexp
    except RuntimeError:
        pass

    # Retry with module-prefix normalization
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    else:
        state = {f"module.{k}": v for k, v in state.items()}

    miss, unexp = model.load_state_dict(state, strict=False)
    return miss, unexp


def build_loader(args):
    audio_conf = {
        "num_mel_bins": 128,
        "target_length": args.target_length,
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "mode": "eval",
        "mean": args.dataset_mean,
        "std": args.dataset_std,
        "noise": False,
        "im_res": 224,
    }

    # Keep the same training mapping for consistent gen_label layout.
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

    ds = datasets.Dataset.load_from_disk(DATASET_INPUT_PATH)
    ds = ds.filter(lambda sample: sample["split"] == args.split)

    dataset = MavosDD(
        dataset=ds,
        input_path=DATASET_INPUT_PATH,
        audio_conf=audio_conf,
        stage=2,
        video_class_name_to_idx=video_labels,
        audio_class_name_to_idx=audio_labels,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader


def build_model(args, device):
    regexp = re.compile(r'.*?_vgh-(?P<use_video_generative_head>(True)|(False))_.*')
    use_video_generative_head = regexp.match(args.checkpoint_path).group("use_video_generative_head") == "True"

    model = VideoCAVMAENoMaskMultiTask(
        n_binary_classes=2,
        n_video_gen_classes=len(VIDEO_METHOD_NAMES),
        use_video_gen_head=use_video_generative_head,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
    )

    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    miss, unexp = load_checkpoint_flexible(model, args.checkpoint_path, device)
    print(f"Loaded checkpoint: {args.checkpoint_path}")
    print(f"Missing keys: {len(miss)} | Unexpected keys: {len(unexp)}")

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args, device)
    loader = build_loader(args)

    data_out = {}
    with torch.no_grad():
        for a_input, v_input, main_label, gen_label, video_paths in tqdm(loader, desc="Inference"):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                binary_logits, video_gen_logits, _ = model(
                    a_input,
                    v_input,
                    return_video_gen_logits=False,
                    return_projections=False,
                )

            binary_probs = torch.softmax(binary_logits, dim=1).cpu().numpy()
            binary_pred_idx = binary_logits.argmax(dim=1).cpu().numpy()  # 0=fake, 1=real

            if video_gen_logits is not None:
                video_gen_probs = torch.softmax(video_gen_logits, dim=1).cpu().numpy()
                video_gen_pred_idx = video_gen_logits.argmax(dim=1).cpu().numpy()
            else:
                video_gen_probs = None
                video_gen_pred_idx = None

            for i, video_path in enumerate(video_paths):
                out = {
                    # Keys for eval_old.py compatibility
                    "pred": binary_logits[i].detach().cpu().tolist(),
                    "true": main_label[i].tolist(),

                    # Extended keys
                    "binary_logits": binary_logits[i].detach().cpu().tolist(),
                    "binary_probs": binary_probs[i].tolist(),
                    "pred_is_fake": int(binary_pred_idx[i] == 0),
                    "pred_is_real": int(binary_pred_idx[i] == 1),
                    "true_main_label": main_label[i].tolist(),  # [is_fake, is_real]
                    "true_gen_label": gen_label[i].tolist(),
                }

                if video_gen_pred_idx is not None:
                    pred_idx = int(video_gen_pred_idx[i])
                    out.update({
                        "video_gen_logits": video_gen_logits[i].detach().cpu().tolist(),
                        "video_gen_probs": video_gen_probs[i].tolist(),
                        "pred_video_method_idx": pred_idx,
                        "pred_video_method_name": VIDEO_METHOD_NAMES[pred_idx]
                        if pred_idx < len(VIDEO_METHOD_NAMES) else str(pred_idx),
                    })

                data_out[video_path] = out

    with open(args.dump_path, "w") as f:
        json.dump(data_out, f, indent=2)

    print(f"Saved predictions to: {args.dump_path}")


if __name__ == "__main__":
    main()
