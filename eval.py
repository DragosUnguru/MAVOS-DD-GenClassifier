import torch
import numpy as np
import json
import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

from src.utilities.stats import calculate_stats


import matplotlib
matplotlib.rcParams.update({'font.size': 26})


SPLIT_TO_EVALUATE = "closed-set"
# SPLIT_TO_EVALUATE = "open-model"
# SPLIT_TO_EVALUATE = "open-language"
# SPLIT_TO_EVALUATE = "open-set"

DATASET_INPUT_PATH = "/mnt/d/projects/datasets/MAVOS-DD"
CHECKPOINT_ROOT_DIR = "/mnt/d/projects/MAVOS-DD-GenClassifer/exp/stage-3/audio+video_classes_but_just_video_labels"
CHECKPOINT_PATH = f"{CHECKPOINT_ROOT_DIR}/models/audio_model.10.pth"
INFERENCE_OUT_PATH = f"{CHECKPOINT_ROOT_DIR}/eval/audio_model.10.PREDICTIONS.json"
PLOT_OUT_PATH = f"{CHECKPOINT_ROOT_DIR}/eval/audio_model.10.{SPLIT_TO_EVALUATE}.png"

class_name_to_label_mapping = {
    'real': 0,
    'echomimic': 1,
    'freevc': 2,
    'hififace': 3,
    'inswapper': 4,
    'knnvc': 5,
    'liveportrait': 6,
    'memo': 7,
    'roop': 8,
    'sonic': 9,
    'audio_real': 10,
    'audio_fake': 11
}

dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}


def plot_multilabel_confusion_matrix(y_pred, y_true, class_name_to_label_mapping, normalize=True):
    """
    Plot per-class confusion matrices (in percentages) for a multilabel classification problem.
    
    Parameters
    ----------
    y_pred : list[list[float]] or np.ndarray
        Model output logits (e.g. the values before applying the final sigmoid activation). Sized as (samples, classes).
    y_true : list[list[int]] or np.ndarray
        Ground-truth binary matrix (samples, classes) with values in {0., 1.}.
    class_name_to_label_mapping : dict
        Mapping from class name to label index.
    normalize : bool, default=True
        If True, display percentages. If False, display raw counts.
    """
    y_true = np.array(y_true)
    y_pred = torch.round(torch.sigmoid(torch.Tensor(y_pred))).cpu().numpy()

    class_names = list(class_name_to_label_mapping.keys())
    n_classes = len(class_names)

    fig, axes = plt.subplots(
        nrows=int(np.ceil(n_classes / 3)), ncols=3, 
        figsize=(15, 4 * np.ceil(n_classes / 3))
    )
    axes = axes.flatten()
    
    for idx, class_name in enumerate(class_names):
        cm = metrics.confusion_matrix(y_true[:, idx], y_pred[:, idx])

        if normalize:
            cm_display = cm.astype("float") / cm.sum() * 100 if cm.sum() > 0 else cm
            fmt = ".2f"
            title = "Confusion Matrix (percentages)"
        else:
            cm_display = cm
            fmt = "d"
            title = "Confusion Matrix (counts)"
        
        sns.heatmap(
            cm_display, annot=True, fmt=fmt, cmap="Blues", 
            xticklabels=["0", "1"], yticklabels=["0", "1"], 
            ax=axes[idx], cbar=False
        )
        axes[idx].set_title(f"{class_name}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    
    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOT_OUT_PATH)
    plt.show()


def plot_multiclass_confusion_matrix(y_pred, y_true, class_name_to_label_mapping, normalize=True):
    """
    Plot a confusion matrix for a multiclass classification problem.
    Expects one-hot encoded y_true and y_pred (shape: n_samples x n_classes).
    
    Parameters
    ----------
    y_pred : list[list[float]] or np.ndarray
        Model output logits (e.g. the values before applying the final sigmoid activation). Sized as (samples, classes).
    y_true : list[list[int]] or np.ndarray
        Ground-truth one-hot encoded matrix (samples, classes).
    class_name_to_label_mapping : dict
        Mapping from class name to label index.
    normalize : bool, default=True
        If True, display percentages. If False, display raw counts.
    """
    y_true = np.array(y_true)
    y_pred = torch.softmax(torch.Tensor(y_pred), dim=1).cpu().numpy()

    # Convert one-hot to class indices
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)

    class_names = list(class_name_to_label_mapping.keys())
    
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(class_names)))

    if normalize:
        cm_display = cm.astype("float") / cm.sum() * 100 if cm.sum() > 0 else cm
        fmt = ".2f"
        title = "Confusion Matrix (percentages)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix (counts)"
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, cbar=False
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(PLOT_OUT_PATH)
    plt.show()


if __name__ == "__main__":
    with open(INFERENCE_OUT_PATH) as input_json_file:
        preds_json = json.load(input_json_file)

    mavos_dd = datasets.Dataset.load_from_disk(DATASET_INPUT_PATH)

    if SPLIT_TO_EVALUATE == "closed-set":
        curr_split = mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==False and sample["open_set_language"]==False)
    elif SPLIT_TO_EVALUATE == "open-model":
        curr_split = datasets.concatenate_datasets([
            mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==False and sample["open_set_language"]==False),
            mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==True and sample["open_set_language"]==False)
        ])
    elif SPLIT_TO_EVALUATE == "open-language":
        curr_split = datasets.concatenate_datasets([
            mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==False and sample["open_set_language"]==False),
            mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==False and sample["open_set_language"]==True)
        ])
    elif SPLIT_TO_EVALUATE == "open-set":
        curr_split = mavos_dd.filter(lambda sample: sample['split']=="test")
    else:
        raise RuntimeError(f"Invalid split given: {SPLIT_TO_EVALUATE}")

    y_pred = []
    y_true = []
    for sample in curr_split:
        entry = preds_json[sample["video_path"]]
        y_pred.append(entry["pred"])
        y_true.append(entry["true"])

    stats = calculate_stats(y_pred, y_true)

    print(f"======= {SPLIT_TO_EVALUATE}: =======")
    print(f"AUC={list(map(lambda entry: entry['AUC'], stats['per_class']))}")
    print(f"precisions={list(map(lambda entry: entry['precisions'], stats['per_class']))}")
    print(f"recalls={list(map(lambda entry: entry['recalls'], stats['per_class']))}")
    print(stats)

    multilabel = np.any(np.array(y_true).sum(axis=1) > 1)

    if multilabel:
        plot_multilabel_confusion_matrix(y_pred, y_true, class_name_to_label_mapping, normalize=False)
    else:
        plot_multiclass_confusion_matrix(y_pred, y_true, class_name_to_label_mapping, normalize=False)

"""
======= audio+video_classes_but_just_video_labels =======
    classes_to_idx = {
        'real': 0,
        'echomimic': 1,
        'hififace': 2,
        'inswapper': 3,
        'liveportrait': 4,
        'memo': 5,
        'roop': 6,
        'sonic': 7,
    }

--- best_audio_model.pth # closed-set ---
{
    "accuracy": 0.9446448917584505,
    "f1_macro": 0.9271457881657005,
    "f1_micro": 0.9446448917584505,
    "precision_per_class": [ 0.944385593220339, 0.9199475065616798, 0.0, 0.9430962343096234, 0.9544554455445544, 0.9603633360858794, 0.0, 0.0 ],
    "recall_per_class": [ 0.9752051048313582, 0.9272486772486772, 0.0, 0.9166327775518504, 0.7980132450331126, 0.9470684039087948, 0.0, 0.0 ],
    "f1_per_class": [ 0.9595479415194188, 0.9235836627140975, 0.0, 0.929676221901423, 0.8692515779981965, 0.9536695366953668, 0.0, 0.0 ]
}

--- audio_model.10.pth # closed-set ---
{
    "mode": "multiclass",
    "accuracy": 0.9454044815799468,
    "f1_macro": 0.9294779324126082,
    "f1_micro": 0.9454044815799468,
    "AP_macro": 0.6102186724039896,
    "AP_micro": 0.9855098518064085,
    "AUC_macro": 0.9930264070226095,
    "AUC_micro": 0.997000946607545,
    "precision_per_class: [ 0.9911027950738813, 0.9774638756965911, -0.0, 0.9807828106030021, 0.9410743670411317, 0.9913255308173108, -0.0, -0.0 ]
    "auc_per_class: [ 0.9916786581701739, 0.9963693344677388, -1, 0.9919814651854895, 0.9864397836609017, 0.998662793628743, -1, -1 ]"
}

Got precision == 0 for: hififace, roop, sonic
    hififace -> real/inswapper
    roop -> real/inswapper
    sonic -> memo
"""
