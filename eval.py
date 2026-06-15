"""
Evaluation script for VideoCAVMAENoMaskMultiTask.
Reads predictions JSON produced by inference.py and evaluates across MAVOS-DD splits.

Usage:
    python eval_old.py \\
        --predictions_path /path/to/predictions.json \\
        --dataset_input_path /mnt/d/projects/datasets/MAVOS-DD \\
        [--plot_confusion]
"""

import argparse
import json
import os
import numpy as np
import torch
import datasets
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Non-interactive backend for headless execution
from sklearn.metrics import confusion_matrix

from src.utilities.stats import calculate_stats_2

matplotlib.rcParams.update({'font.size': 26})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True,
                        help="Path to JSON produced by inference.py")
    parser.add_argument("--dataset_input_path", type=str,
                        default="/mnt/d/projects/datasets/MAVOS-DD")
    parser.add_argument("--plot_confusion", action="store_true",
                        help="Save confusion matrix plots alongside the JSON")
    parser.add_argument("--eval_video_gen_method", action="store_true",
                        help="Evaluate 5-class video generative method instead of binary classification")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    return parser.parse_args()


def plot_confusion_matrix_percent(y_true_idx, y_pred_idx, name="name", save_dir=None, class_labels=None):
    if class_labels is None:
        # Binary case
        class_labels = ["Fake", "Real"]
        label_names = ["Real" if y == 1 else "Fake" for y in y_true_idx]
        pred_names  = ["Real" if y == 1 else "Fake" for y in y_pred_idx]
    else:
        # Multi-class case
        label_names = [class_labels[int(y)] for y in y_true_idx]
        pred_names  = [class_labels[int(y)] for y in y_pred_idx]

    cm = confusion_matrix(label_names, pred_names,
                          labels=class_labels, normalize="true")
    cm_percent = cm * 100

    figsize = (5, 5) if len(class_labels) == 2 else (8, 8)
    plt.figure(figsize=figsize)
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=False)
    plt.tight_layout()

    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, f"{name}_confusion.png"), dpi=150)
    plt.close()


SPLITS = [
    "closed-set",
    "open-model",
    "open-language",
    "open-set",
    "out-of-distribution",
]


def get_split_samples(split_name, split, mavos_dd):
    closed = mavos_dd.filter(
        lambda s: s["split"] == split
        and s["open_set_model"] == False
        and s["open_set_language"] == False
    )

    if split_name == "closed-set":
        return closed
    elif split_name == "open-model":
        return datasets.concatenate_datasets([
            closed,
            mavos_dd.filter(lambda s: s["split"] == split
                            and s["open_set_model"] == True
                            and s["open_set_language"] == False),
        ])
    elif split_name == "open-language":
        return datasets.concatenate_datasets([
            closed,
            mavos_dd.filter(lambda s: s["split"] == split
                            and s["open_set_model"] == False
                            and s["open_set_language"] == True),
        ])
    elif split_name == "open-set":
        return mavos_dd.filter(lambda s: s["split"] == split)
    elif split_name == "out-of-distribution":
        return mavos_dd.filter(
            lambda s: s["split"] == split
            and s["generative_method"] in ["hififace", "sonic", "real", "roop"]
        )
    else:
        raise ValueError(f"Unknown split: {split_name}")


if __name__ == "__main__":
    args = parse_args()

    with open(args.predictions_path) as f:
        preds_json = json.load(f)
    print(f"Loaded {len(preds_json)} predictions from {args.predictions_path}")

    mavos_dd = datasets.Dataset.load_from_disk(args.dataset_input_path)
    save_dir = os.path.dirname(args.predictions_path) if args.plot_confusion else None

    results = {}
    if not args.eval_video_gen_method:
        for split_name in SPLITS:
            curr_split = get_split_samples(split_name, args.split, mavos_dd)

            y_pred, y_true = [], []
            missing = 0
            nan_count = 0
            for sample in curr_split:
                entry = preds_json.get(sample["video_path"])
                if entry is None:
                    missing += 1
                    continue
                pred = entry["pred"]
                if any(not np.isfinite(v) for v in pred):
                    nan_count += 1
                    continue
                y_pred.append(pred)            # 2-logit list  (from inference.py)
                y_true.append(entry["true"])   # 2-dim one-hot (from inference.py)

            if missing:
                print(f"  [warn] {missing} samples not found in predictions — skipped")
            if nan_count:
                print(f"  [warn] {nan_count} samples had NaN/Inf logits — skipped")

            if not y_pred:
                print(f"{split_name}: no predictions — skipped\n")
                continue

            y_pred_probs = torch.softmax(torch.tensor(y_pred), dim=1)
            stats = calculate_stats_2(y_pred_probs, torch.tensor(y_true))

            mAP  = float(np.mean([s["AP"]  for s in stats]))
            mAUC = float(np.mean([s["auc"] for s in stats]))
            acc  = float(stats[0]["acc"])

            results[split_name] = {"mAP": mAP, "mAUC": mAUC, "acc": acc}
            print(f"{split_name}: mAP={mAP:.4f}, mAUC={mAUC:.4f}, acc={acc:.4f}\n")

            if args.plot_confusion:
                y_pred_idx = np.argmax(y_pred_probs.numpy(), axis=1)
                y_true_idx = np.argmax(y_true, axis=1)
                plot_confusion_matrix_percent(y_true_idx, y_pred_idx,
                                            name=split_name, save_dir=save_dir)

    if args.eval_video_gen_method:
        print("\n" + "="*60)
        print("VIDEO GENERATIVE METHOD EVALUATION (5 classes)")
        print("="*60 + "\n")

        video_gen_labels = ["real", "memo", "liveportrait", "inswapper", "echomimic"]
        video_gen_label_to_idx = {name: idx for idx, name in enumerate(video_gen_labels)}

        y_pred_vg, y_true_vg = [], []
        missing_vg = 0
        nan_count_vg = 0

        for sample in mavos_dd.filter(lambda s: s["split"] == "validation"):
            entry = preds_json.get(sample["video_path"])
            if entry is None:
                missing_vg += 1
                continue
            if "video_gen_logits" not in entry:
                continue
            logits = entry["video_gen_logits"]
            if any(not np.isfinite(v) for v in logits):
                nan_count_vg += 1
                continue
            # IMPORTANT: true_gen_label is 9-way (4 video + 5 audio) and does NOT include "real".
            # For 5-way video-gen head eval, derive ground-truth from dataset generative_method.
            true_video_method = sample["generative_method"]
            if len(logits) != 5 or true_video_method not in video_gen_label_to_idx:
                continue

            true_vec = [0.0] * 5
            true_vec[video_gen_label_to_idx[true_video_method]] = 1.0
            y_pred_vg.append(logits)
            y_true_vg.append(true_vec)

        if missing_vg:
            print(f"  [warn] {missing_vg} samples not found in predictions — skipped")
        if nan_count_vg:
            print(f"  [warn] {nan_count_vg} samples had NaN/Inf logits — skipped")

        if not y_pred_vg:
            print("No predictions for video gen method — skipped\n")
        else:
            stats_vg = calculate_stats_2(torch.tensor(y_pred_vg), torch.tensor(y_true_vg))

            mAP_vg = float(np.mean([s["AP"]  for s in stats_vg]))
            mAUC_vg = float(np.mean([s["auc"] for s in stats_vg]))
            acc_vg = float(stats_vg[0]["acc"])

            print(f"Video Gen Method: mAP={mAP_vg:.4f}, mAUC={mAUC_vg:.4f}, acc={acc_vg:.4f}\n")
            results = {
                "video_gen_method_validation": {
                    "mAP": mAP_vg,
                    "mAUC": mAUC_vg,
                    "acc": acc_vg,
                }
            }

            if args.plot_confusion:
                y_pred_idx_vg = np.argmax(np.array(y_pred_vg), axis=1)
                y_true_idx_vg = np.argmax(np.array(y_true_vg), axis=1)
                plot_confusion_matrix_percent(y_true_idx_vg, y_pred_idx_vg,
                                              name="video_gen_method", save_dir=save_dir,
                                              class_labels=video_gen_labels)

    summary_path = args.predictions_path.replace(".json", "_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_path}")