import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import numpy as np
from torch.cuda.amp import autocast
import json
from tqdm import tqdm
import datasets

from src.mavosdd_dataset import MavosDD


DATASET_INPUT_PATH = "/mnt/d/projects/datasets/MAVOS-DD"
CHECKPOINT_ROOT_DIR = "/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/binary_classification_RANDOM_mask_experiment_MINISET_03"
CHECKPOINT_PATH = f"{CHECKPOINT_ROOT_DIR}/models/audio_model.10.pth"
DUMP_PATH = f"{CHECKPOINT_ROOT_DIR}/eval/audio_model.10.PREDICTIONS.json"

# video_labels = {
#     "memo": 0,
#     "liveportrait": 1,
#     "inswapper": 2,
#     "echomimic": 3,
# }
# audio_labels = {
#     "knnvc": 4,
#     "freevc": 5,
#     "openvoice": 6,
#     "xtts_v2": 7,
#     "yourtts": 8,
# }
# class_name_to_label_mapping = { **video_labels, **audio_labels }

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}

if __name__ == "__main__":
    # Load model
    cavmae_ft = VideoCAVMAEFT(n_classes=2)
    if not isinstance(cavmae_ft, torch.nn.DataParallel):
        cavmae_ft = torch.nn.DataParallel(cavmae_ft)
    cavmae_ft.eval()
    cavmae_ft.to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    miss, unexp = cavmae_ft.load_state_dict(ckpt, strict=False)
    assert len(miss) == 0 and len(unexp) == 0

    mavos_dd = datasets.Dataset.load_from_disk(DATASET_INPUT_PATH)

    val_loader = torch.utils.data.DataLoader(
        MavosDD(
            mavos_dd.filter(lambda sample: sample['split']=="test"),
            DATASET_INPUT_PATH,
            val_audio_conf,
            stage=2,
            # video_class_name_to_idx=video_labels,
            # audio_class_name_to_idx=audio_labels
        ),
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
                # model_output, _, _, _ = cavmae_ft(a_input, v_input, apply_mask=True, hard_mask=True, hard_mask_ratio=0.4, adversarial=True)
                model_output, _, _, _ = cavmae_ft(
                    a_input, v_input, 
                    apply_mask=True, 
                    masking_mode='random',
                    hard_mask_ratio=0.4,
                    adversarial=False
                )
                model_output = model_output.cpu().numpy()

            for y_pred, y_true, video_path in zip(model_output, labels.numpy(), video_paths):
                data_out[video_path] = {
                    "pred": y_pred.tolist(),
                    "true": y_true.tolist(),
                }

    with open(DUMP_PATH, 'w') as f:
        json.dump(data_out, f, indent=2)
