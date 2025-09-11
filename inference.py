import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import numpy as np
from torch.cuda.amp import autocast
import json
from tqdm import tqdm
import datasets

from src.utilities.stats import calculate_stats
from src.mavosdd_dataset_multiclass import MavosDD
from src.exddv_dataset import ExDDV
from src.custom_dataset import CustomDDV


DATASET_INPUT_PATH = "/mnt/d/projects/datasets/MAVOS-DD"
CHECKPOINT_ROOT_DIR = "/mnt/d/projects/MAVOS-DD-GenClassifer/exp/stage-3/audio+video_classes_but_just_video_labels"
CHECKPOINT_PATH = f"{CHECKPOINT_ROOT_DIR}/models/best_audio_model.pth"
DUMP_PATH = f"{CHECKPOINT_ROOT_DIR}/eval/best_audio_model.PREDICTIONS.json"

class_name_to_label_mapping = {
    'real': 0,
    'echomimic': 1,
    'hififace': 2,
    'inswapper': 3,
    'liveportrait': 4,
    'memo': 5,
    'roop': 6,
    'sonic': 7,
    'audio_real': 8,
    'audio_fake': 9
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model
cavmae_ft = VideoCAVMAEFT(n_classes=len(class_name_to_label_mapping))
if not isinstance(cavmae_ft, torch.nn.DataParallel):
    cavmae_ft = torch.nn.DataParallel(cavmae_ft)
cavmae_ft.eval()

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
miss, unexp = cavmae_ft.load_state_dict(ckpt, strict=False)
assert len(miss) == 0 and len(unexp) == 0 

dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}

if __name__ == "__main__":
    cavmae_ft.to(device)

    mavos_dd = datasets.Dataset.load_from_disk(DATASET_INPUT_PATH)

    val_loader = torch.utils.data.DataLoader(
        MavosDD(mavos_dd.filter(lambda sample: sample['split']=="test"), DATASET_INPUT_PATH, val_audio_conf, stage=2, class_name_to_idx=class_name_to_label_mapping),
        batch_size=32, shuffle=False, num_workers=2, pin_memory=False
    )

    A_predictions, A_targets = [], []
    data_out = {}
    with torch.no_grad():
        for i, (a_input, v_input, labels, video_paths) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing data"):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                # model_output = torch.round(torch.sigmoid(cavmae_ft(a_input, v_input))).cpu().numpy()
                model_output = cavmae_ft(a_input, v_input).cpu().numpy()

            for y_pred, y_true, video_path in zip(model_output, labels.numpy(), video_paths):
                data_out[video_path] = {
                    "pred": y_pred.tolist(),
                    "true": y_true.tolist(),
                }

    with open(DUMP_PATH, 'w') as f:
        json.dump(data_out, f, indent=2)
