import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader
import numpy as np
from torch.cuda.amp import autocast
import json
from tqdm import tqdm
import csv
import datasets

from src.mavosdd_dataset import MavosDD
from src.utilities.stats import calculate_stats


dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}
 
    
if __name__ == "__main__":
    with open("/home/eivor/biodeep/Detection/OpenAVFF/predictions_exddv_crossmodel.json") as input_json_file:
        preds_json = json.load(input_json_file)

    y_pred = []
    y_true = []
    for path,entry in preds_json.items():
        y_pred.append(entry["pred"])
        y_true.append(entry["true"])
        
    stats = calculate_stats(torch.tensor(y_pred), torch.tensor(y_true))
    
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy
    
    print(f"Results: {mAP=}, {mAUC=}, {acc=}\n")

"""
Closed-set: mAP=0.9501837501637329, mAUC=0.9481767559252561, acc=0.8693101355452154

Open model: mAP=0.5, mAUC=-1.0, acc=0.6582830058864001

Open language: mAP=0.8560615794926437, mAUC=0.8525392515611991, acc=0.8245014245014245

Open set: mAP=0.5, mAUC=-1.0, acc=0.7689301416707377
"""
