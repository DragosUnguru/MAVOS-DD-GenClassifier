import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import numpy as np
from torch.cuda.amp import autocast
import json
from tqdm import tqdm
import datasets

from src.utilities.stats import calculate_stats
from src.mavosdd_dataset import MavosDD
from src.exddv_dataset import ExDDV
from src.custom_dataset import CustomDDV


mavos_flag = False
if mavos_flag:
    DATASET_INPUT_PATH = "/mnt/d/projects/datasets/MAVOS-DD"
    CHECKPOINT_PATH = "/home/eivor/biodeep/Detection/OpenAVFF/egs/exp/stage-3-mavos/models/best_audio_model.pth"
    CHECKPOINT_PATH = "/home/eivor/biodeep/Detection/OpenAVFF/checkpoints/stage-3.pth"
else:
    DATASET_INPUT_PATH = "/home/eivor/biodeep/xAI_deepfake/dataset5.csv"
    # CHECKPOINT_PATH = "/home/eivor/biodeep/Detection/OpenAVFF/egs/exp/stage-3-exddv-crossds/models/best_audio_model.pth"
    CHECKPOINT_PATH = "/home/eivor/biodeep/Detection/OpenAVFF/checkpoints/stage-3.pth"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
audio_model = VideoCAVMAEFT()
audio_model = torch.nn.DataParallel(audio_model)
audio_model.eval()
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
miss, unexp = audio_model.load_state_dict(ckpt, strict=False)
assert len(miss) == 0 and len(unexp) == 0 

dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}

    
if __name__ == "__main__":
    audio_model.to(device)
    
    if mavos_flag:
        mavos_dd = datasets.Dataset.load_from_disk(DATASET_INPUT_PATH)

        val_loader = torch.utils.data.DataLoader(
            MavosDD(mavos_dd.filter(lambda sample: sample['split']=="test"), DATASET_INPUT_PATH, val_audio_conf, stage=2),
            batch_size=32, shuffle=False, num_workers=0, pin_memory=False
        )
    else:
        # val_loader = torch.utils.data.DataLoader(
        #     ExDDV(DATASET_INPUT_PATH, None, val_audio_conf, stage=2),
        #     batch_size=32, shuffle=False, num_workers=0, pin_memory=False
        # )
        val_loader = torch.utils.data.DataLoader(
            CustomDDV(val_audio_conf, stage=2),
            batch_size=2, shuffle=False, num_workers=0, pin_memory=False
        )
    
    A_predictions, A_targets = [], []
    data_out = {}
    with torch.no_grad():
        for i, (a_input, v_input, labels, video_paths) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing data"):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                audio_output = audio_model(a_input, v_input).cpu().numpy()
            # probabilities = torch.sigmoid(audio_output).cpu().numpy()
            
            for y_pred,y_true,video_path in zip(audio_output,labels.numpy(),video_paths):
                data_out[video_path] = {
                    "pred": y_pred.tolist(),
                    "true": y_true.tolist(),
                }
                
    with  open('predictions_custom.json', 'w') as f:
      json.dump(data_out, f, indent=4)
