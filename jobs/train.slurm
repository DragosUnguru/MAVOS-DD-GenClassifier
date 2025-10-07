#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH -C gmemT48
#SBATCH -c 10

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi
source /home/fl488644/miniconda3/etc/profile.d/conda.sh
conda init bash
module load cuda/11.7

conda activate AVFF
cd /home/${USER}/MAVOS-DD-GenClassifier
echo $CONDA_DEFAULT_ENV
wandb login 594bc31d98355581c73bec4991abf902a00c7ea3
export CUDA_VISIBLE_DEVICES="0"

sh finetune.sh
