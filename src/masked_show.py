import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm
from mavosdd_dataset import MavosDD
from mini_datasets import get_mini_train_set_deepfake_detection
from models.video_cav_mae import VideoCAVMAEFT

def visualize_mask_as_gif(
    video,
    mask,
    save_path="/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/trainable_mask_binary_classification_COTRAINING/examples/masked_video.gif",
    tubelet_size=2,
    patch_size=16,
    fps=2,
):
    """
    Creates an animated GIF visualizing the masking pattern over time.
    
    Args:
        video: Tensor (1, 3, T, H, W)
        mask: Tensor (1, L) binary (1 = masked)
        tubelet_size: temporal grouping size
        patch_size: spatial patch size
        fps: frames per second for the GIF
    """
    B, C, T, H, W = video.shape
    num_temporal_groups = T // tubelet_size
    num_patches = (H // patch_size, W // patch_size)  # (14, 14)

    # reshape mask -> (1, num_tubelets, H_patches, W_patches)
    mask = mask.view(1, num_temporal_groups, *num_patches)

    # expand tubelet mask to per-frame level
    mask_per_frame = mask.repeat_interleave(tubelet_size, dim=1)[0]  # (T, 14, 14)

    # upsample to pixel level via Kronecker product
    mask_pixel = torch.kron(mask_per_frame, torch.ones((patch_size, patch_size)))  # (T, 224, 224)

    frames = []
    for t in range(T):
        frame = video[0, :, t].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        frame = np.clip(frame, 0, 1)
        mask_overlay = mask_pixel[t].cpu().numpy()

        # Overlay red where mask=1
        red = np.zeros_like(frame)
        red[..., 0] = 1.0
        overlay = 0.5 * frame + 0.5 * (red * mask_overlay[..., None])

        # blend between original and overlay
        img = (overlay * 255).astype(np.uint8)
        frames.append(img)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved masked video visualization: {save_path}")


if __name__ == "__main__":
    batch_size = 1
    target_length = 1024
    freqm = 0
    timem = 0
    dataset_mean = -5.081
    dataset_std = 4.4849
    noise = False
    im_res = 224
    mae_loss_weight = 3.0
    contrast_loss_weight = 0.01

    model_weights_path = '/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/trainable_mask_binary_classification_2-STAGE-TRAINING/01-avff-frozen-mask-trainable/models/audio_model.10.pth'
    input_path = '/mnt/d/projects/datasets/MAVOS-DD'

    audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': freqm, 'timem': timem, 'mode':'train',
                'mean':dataset_mean, 'std':dataset_std, 'noise':noise, 'label_smooth': 0, 'im_res': im_res}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'mode':'eval',
                'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': im_res}

    print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(mae_loss_weight, contrast_loss_weight))

    mavos_dd = get_mini_train_set_deepfake_detection(input_path)
    test_loader = DataLoader(
        MavosDD(
            mavos_dd,
            input_path,
            audio_conf,
            stage=2,
            custom_file_path=False),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )

    # Load pre-trained AVFF model & weights
    cavmae_ft = VideoCAVMAEFT(n_classes=2)
    if not isinstance(cavmae_ft, torch.nn.DataParallel):
        cavmae_ft = torch.nn.DataParallel(cavmae_ft)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    mdl_weight = torch.load(model_weights_path, map_location=device)
    miss, unexpected = cavmae_ft.load_state_dict(mdl_weight, strict=False)

    print('Missing: ', miss)
    print('Unexpected: ', unexpected)

    cavmae_ft.eval()

    count = 0
    for (audio_input, video_input, labels, video_path) in tqdm(test_loader):
        if count >= 5:
            break
        with torch.no_grad():
            output, video_mask, ids_restore = cavmae_ft(
                audio_input.to(device),
                video_input.to(device),
                apply_mask=True,
                hard_mask=True,
                hard_mask_ratio=0.4)

            visualize_mask_as_gif(
                video_input.cpu(),
                video_mask.cpu(),
                save_path=f"/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/trainable_mask_binary_classification_2-STAGE-TRAINING/01-avff-frozen-mask-trainable/examples/masked_video_{count}.gif",
                tubelet_size=2,
                patch_size=16,
                fps=3,
            )
        count += 1
