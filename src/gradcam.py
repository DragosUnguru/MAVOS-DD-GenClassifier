import os
import argparse
import datasets
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from models.video_cav_mae import VideoCAVMAEFT
from mavosdd_dataset_multiclass import MavosDD

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from pathlib import Path
from mini_datasets import get_mini_test_set


# Custom target we want to generate the CAM for.
# Simple sum of activations as the video module has no classification layer
# (i.e. it outputs just embeddings)
class FeatureSumTarget:
    def __call__(self, model_output):
        return model_output.sum()

# Custom target we want to generate the CAM for.
# Simple argmax of activations on the last dimension
# Suitable for audio module
class FeatureArgMaxTarget:
    def __call__(self, model_output):
        return model_output.arg_max(dim=1)


def dump_cam_to_disk(visualizations, dump_file_path):
    if isinstance(visualizations, list):
        # Multiple frames. Dump video to disk
        height, width, _ = visualizations[0].shape
        out = cv2.VideoWriter(dump_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        for vis in visualizations:
            for _ in range(8): # Repeat each frame several times for better visualisation in video playback
                out.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        out.release()
    else:
        # Single frame. Dump image to disk
        cv2.imwrite(dump_file_path, visualizations)


def apply_cam_mask(
    rgb_img: np.ndarray,
    grayscale_cam: np.ndarray,
    mode: str = "soft",
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply mask to an RGB image.

    Parameters
    ----------
    rgb_img : np.ndarray
        Original RGB image, float32 in [0, 1], shape (H, W, 3).
    grayscale_cam : np.ndarray
        Grad-CAM map, float32 in [0, 1], shape (H, W).
    mode : str
        "soft" for fading mask, "hard" for binary mask.
    threshold : float
        Threshold for hard masking. Ignored if mode == "soft".

    Returns
    -------
    masked_img : np.ndarray
        Masked image, float32 in [0, 1], shape (H, W, 3).
    """

    assert rgb_img.max() <= 1.0, "rgb_img should be normalized to [0, 1]"
    assert grayscale_cam.ndim == 2, "grayscale_cam should be 2D (H, W)"
    assert mode in ("soft", "hard"), "mode must be 'soft' or 'hard'"
    assert 0 < threshold < 1, "threshold should be between (0.0, 1.0)"

    # Build mask
    if mode == "soft":
        mask = grayscale_cam
    else:  # hard
        mask = (grayscale_cam >= threshold).astype(np.float32)

    # Expand to 3 channels
    mask_3 = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Apply mask
    masked_img = rgb_img * mask_3

    # return masked_img.astype(np.float32)
    return np.uint8(np.clip(masked_img * 255, 0, 255))


'''
    Generates the CAM visualisation for the given model and model's input tensor.
    The output is a .mp4 video of the 16 frames (in AVFF's case) of the first entry from the batch
    (i.e. batch size is ignored and the CAM is generated exclusively for the first entry in the batch).
'''
def gradcam_show(
    model,
    input_video,
    input_audio,
    target_layers,
    targets,
    reshape_transform=None,
    dump_file_path='/mnt/d/projects/MAVOS-DD-GenClassifer/exp/gradcam_video.mp4',
    norm_mean=[0.4850, 0.4560, 0.4060],
    norm_std=[0.2290, 0.2240, 0.2250],
    apply_mask=False
):
    # =============================================
    # input_tensor: torch.Size([batch_size x rgb_channels x num_frames x height x width)
    # input_tensor: torch.Size([32, 3, 16, 224, 224])

    # rgb_img: (height x width x rgb_channels)
    # rgb_img: (224, 224, 3)

    # grayscale_cam: (height x width)
    # grayscale_cam: (224, 224)
    # =============================================

    temporal_dimension_size = input_video.shape[2]

    # Denormalize the input
    def denormalize_tensor(frame):
        for tensor, mean, std in zip(frame, norm_mean, norm_std):
            tensor.mul_(std).add_(mean)
        return frame

    rgb_img_collection = []
    for temporal_idx in range(temporal_dimension_size):
        rgb_img = input_video[0, :, temporal_idx, :, :].clone()
        rgb_img = denormalize_tensor(rgb_img)
        rgb_img = rgb_img.permute(1, 2, 0).cpu().numpy()    # (channels x height x width) -> (height x width x channels)
        # rgb_img = np.clip(rgb_img, 0, 1).astype(np.float32)

        rgb_img_collection.append(rgb_img)

    class ModelWrapper(nn.Module):
        def __init__(self, model, fixed_video_tensor = None, fixed_audio_tensor = None):
            super().__init__()

            assert (fixed_video_tensor is None) != (fixed_audio_tensor is None), "There must be just one fixed tensor"

            self.model = model
            self.fixed_video_tensor = fixed_video_tensor
            self.fixed_audio_tensor = fixed_audio_tensor

        def forward(self, dynamic_tensor):
            if self.fixed_audio_tensor is not None:
                return self.model(self.fixed_audio_tensor, dynamic_tensor)
            elif self.fixed_video_tensor is not None:
                return self.model(dynamic_tensor, self.fixed_video_tensor)

    # Generate CAM visualizations and save to disk
    wrapped_model = ModelWrapper(model, fixed_audio_tensor=input_audio)
    all_visualizations = []

    with GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        for temporal_idx in range(temporal_dimension_size):
            grayscale_cam = cam(input_tensor=input_video, targets=targets)
            grayscale_cam = grayscale_cam[0, temporal_idx]

            if apply_mask:
                all_visualizations.append(
                    apply_cam_mask(rgb_img=rgb_img_collection[temporal_idx], grayscale_cam=grayscale_cam, mode="hard", threshold=0.5)
                )
            else:
                all_visualizations.append(
                    show_cam_on_image(rgb_img_collection[temporal_idx], grayscale_cam, use_rgb=True)
                )

    dump_cam_to_disk(all_visualizations, dump_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video CAV-MAE')

    parser.add_argument('--batch-size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--model_weights_path', default='/mnt/d/projects/MAVOS-DD-GenClassifer/exp/stage-3/models/audio_model.8.pth', type=str, help='the path to the CAVMAEFT model weights on the MAVOS-DD dataset')
    parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)
    parser.add_argument('--dataset_mean', default=-5.081, type=float, help='the dataset audio spec mean, used for input normalization')
    parser.add_argument('--dataset_std', default=4.4849, type=float, help='the dataset audio spec std, used for input normalization')
    parser.add_argument('--noise', default=False, type=bool, help='add noise to the input')
    parser.add_argument('--mae_loss_weight', type=float, default=3.0, help='weight for mae loss')
    parser.add_argument('--contrast_loss_weight', type=float, default=0.01, help='weight for contrastive loss')

    args = parser.parse_args()

    input_path = '/mnt/d/projects/datasets/MAVOS-DD'
    im_res = 224
    audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mode':'train',
                'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'mode':'eval',
                'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

    print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))

    # Load dataset for which to generate CAMs for
    # mavos_dd = datasets.Dataset.load_from_disk(input_path).filter(lambda sample: sample['split']=='train' and sample['generative_method'] == 'liveportrait')
    # test_loader = DataLoader(
    #     MavosDD(
    #         mavos_dd,
    #         input_path,
    #         audio_conf,
    #         stage=2),
    #     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False
    # )
    test_loader = get_mini_test_set(args.batch_size, audio_conf, args.num_workers, shuffle=False)

    # Load pre-trained AVFF model & weights
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
    class_name_to_label_mapping = { **video_labels, **audio_labels }

    cavmae_ft = VideoCAVMAEFT(n_classes=len(class_name_to_label_mapping))
    if not isinstance(cavmae_ft, torch.nn.DataParallel):
        cavmae_ft = torch.nn.DataParallel(cavmae_ft)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    mdl_weight = torch.load(args.model_weights_path, map_location=device)
    miss, unexpected = cavmae_ft.load_state_dict(mdl_weight, strict=False)

    print('Missing: ', miss)
    print('Unexpected: ', unexpected)

    def reshape_transform_temporal(tensor, num_frames=16, height=224, width=224, patch_size=(2,16,16)):
        """
        Reshape transformer output (B, N, D) into (B, D, Nt, Nh, Nw),
        where Nt = num_frames / patch_size[0], Nh = H / patch_size[1], Nw = W / patch_size[2].
        """
        B, N, D = tensor.shape
        
        Nt = num_frames // patch_size[0]
        Nh = height // patch_size[1]
        Nw = width // patch_size[2]

        assert N == Nt * Nh * Nw, f"Shape mismatch: got N={N}, expected {Nt * Nh * Nw}"
        
        tensor = tensor.view(B, Nt, Nh, Nw, D)   # (B, Nt, Nh, Nw, D)
        tensor = tensor.permute(0, 4, 1, 2, 3)   # (B, D, Nt, Nh, Nw)
        return tensor


    count_to_generate = 10    
    for (audio_input, video_input, labels, video_path) in test_loader:
        unbatched_label = labels[0]
        unbatched_video_path = video_path[0]

        (language, generative_method, video_name) = unbatched_video_path.split(os.path.sep)
        
        dump_file_path = Path(f'/mnt/d/projects/MAVOS-DD-GenClassifer/exp/{language}/{generative_method}/MASKED_{video_name}')
        dump_file_path.parent.mkdir(exist_ok=True, parents=True)

        print(f'Generating {dump_file_path}...')

        audio_input = audio_input.to(device)
        video_input = video_input.to(device)

        gradcam_show(
            model=cavmae_ft.module,
            input_video=video_input,
            input_audio=audio_input,
            target_layers=[
                # target_layers=[cavmae_ft.module.visual_encoder.patch_embedding.projection], # convolutional layer
                cavmae_ft.module.visual_encoder.blocks[11].attn.proj,             # last attention projection
                cavmae_ft.module.visual_encoder.blocks[11].mlp.layers[1].linear,  # last MLP linear
                cavmae_ft.module.visual_encoder.norm                              # final LN
            ],
            targets=[ClassifierOutputTarget(class_name_to_label_mapping['liveportrait'])],
            reshape_transform=reshape_transform_temporal,
            dump_file_path=dump_file_path,
            apply_mask=True
        )

        count_to_generate -= 1
        if count_to_generate == 0:
            break
