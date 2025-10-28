import os
import argparse
import datasets
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.video_cav_mae import VideoCAVMAEFT
from mavosdd_dataset_multiclass import MavosDD

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from pathlib import Path
from mini_datasets import get_mini_test_set, get_mini_train_set_deepfake_detection


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


def dump_cam_to_disk(visualizations, dump_file_path, fps=25):
    assert visualizations.ndim in [3, 4], f"Can only dump images (H x W x Ch) or videos (T x H x W x Ch). Got shape: {visualizations.shape}"

    if visualizations.ndim == 4:
        # Multiple frames. Dump video to disk
        num_frames, height, width, _ = visualizations.shape
        out = cv2.VideoWriter(dump_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for t in range(num_frames):
            out.write(cv2.cvtColor(visualizations[t], cv2.COLOR_RGB2BGR))
    else:
        # Single frame. Dump image to disk
        cv2.imwrite(dump_file_path, visualizations)

    out.release()


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
        mask = (grayscale_cam <= 1.0 - threshold).astype(np.float32)

    # Expand to 3 channels
    mask_3 = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Apply mask
    masked_img = rgb_img * mask_3

    # return masked_img.astype(np.float32)
    return np.uint8(np.clip(masked_img * 255, 0, 255))


def pool_cam(
    rgb_video: np.ndarray,
    grayscale_cam: np.ndarray,
    kernel_size=(2, 16, 16),
    stride=(2, 16, 16),
    padding=0) -> np.ndarray:
    """
    Apply average pooling to a Grad-CAM heatmap.

    Parameters
    ----------
    grayscale_cam : np.ndarray
        Heatmap array of shape (T, H, W).

    Returns
    -------
    pooled : np.ndarray
        Downsampled heatmap of shape (T/2, H/16, W/16).
    """
    assert rgb_video.ndim == 4 and rgb_video.shape[-1] == 3, "rgb_video must have shape (T, H, W, 3)"
    assert grayscale_cam.ndim == 3, "grayscale_cam must have shape (T, H, W)"

    T, H, W, _ = rgb_video.shape
    assert grayscale_cam.shape == (T, H, W), "rgb_video and grayscale_cam must have matching (T, H, W)"

    # ---- Pool RGB video ----
    video_tensor = torch.from_numpy(rgb_video).permute(3, 0, 1, 2)  # (3, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)  # (1, 3, T, H, W)

    pooled_video = nn.functional.avg_pool3d(
        video_tensor.float(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    pooled_video = pooled_video.squeeze(0).permute(1, 2, 3, 0)  # (T/2, H/16, W/16, 3)
    pooled_video = pooled_video.cpu().numpy()

    # ---- Pool grayscale CAM ----
    # Add (batch_size, channels) dimension to grayscale CAM
    cam_tensor = torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)

    pooled_cam = nn.functional.avg_pool3d(
        cam_tensor.float(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    pooled_cam = pooled_cam.squeeze(0).squeeze(0).cpu().numpy()  # (T/2, H/16, W/16)

    return pooled_video, pooled_cam


'''
    Generates the CAM visualisation for the given model and model's input tensor.
    The output is a .mp4 video of the 16 frames (in AVFF's case) of the first entry from the batch
    (i.e. batch size is ignored and the CAM is generated exclusively for the first entry in the batch).
'''
def gradcam_show(
    model,
    input_video,
    input_audio,
    y_true,
    target_layers,
    targets,
    reshape_transform=None,
    dump_file_path=None,
    norm_mean=[0.4850, 0.4560, 0.4060],
    norm_std=[0.2290, 0.2240, 0.2250],
    apply_mask=False,
    skip_if_mislabeled=False
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
        # print(targets)
        grayscale_cams = cam(input_tensor=input_video, targets=targets)[0]

        if skip_if_mislabeled == True:
            y_pred = torch.round(torch.sigmoid(torch.Tensor(cam.outputs[0]))).cpu()

            if not torch.equal(y_pred, y_true):
                print(f"Skipping as y_pred != y_true")
                return None, None, None

        for temporal_idx in range(temporal_dimension_size):
            grayscale_cam = grayscale_cams[temporal_idx]

            if apply_mask:
                all_visualizations.append(
                    apply_cam_mask(rgb_img=rgb_img_collection[temporal_idx], grayscale_cam=grayscale_cam, mode="hard", threshold=0.4)
                )
            else:
                all_visualizations.append(
                    show_cam_on_image(rgb_img_collection[temporal_idx], grayscale_cam, use_rgb=True)
                )

    rgb_img_collection = np.stack(rgb_img_collection)
    overlayed_video = np.stack(all_visualizations)

    if dump_file_path is not None:
        dump_cam_to_disk(overlayed_video, dump_file_path)

    return overlayed_video, rgb_img_collection, grayscale_cams


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

    # mavos_dd = datasets.Dataset.load_from_disk(input_path).filter(lambda sample: sample['split']=="train" and (sample['generative_method'] != "real" or sample['audio_generative_method'] != "real"))
    mavos_dd = get_mini_train_set_deepfake_detection(input_path)
    test_loader = DataLoader(
        MavosDD(
            mavos_dd,
            input_path,
            audio_conf,
            video_class_name_to_idx=video_labels,
            audio_class_name_to_idx=audio_labels,
            stage=2),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # Load pre-trained AVFF model & weights
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

    for (audio_input, video_input, labels, video_path) in tqdm(test_loader):
        unbatched_label = labels[0]
        unbatched_video_path = video_path[0]

        (language, generative_method, video_name) = unbatched_video_path.split(os.path.sep)
        
        # dump_file_path = Path(f'/mnt/d/projects/MAVOS-DD-GenClassifer/exp/{language}/{generative_method}/POOLED_{video_name}')
        dump_file_path = Path(f'/mnt/d/projects/MAVOS-DD-GenClassifer/subset/{language}/{generative_method}/{video_name}/masked_video.mp4')
        dump_file_path.parent.mkdir(exist_ok=True, parents=True)

        print(f'Generating {dump_file_path}...')

        audio_input = audio_input.to(device)
        video_input = video_input.to(device)

        _, rgb_video, grayscale_video = gradcam_show(
            model=cavmae_ft.module,
            input_video=video_input,
            input_audio=audio_input,
            y_true=unbatched_label,
            target_layers=[
                # target_layers=[cavmae_ft.module.visual_encoder.patch_embedding.projection], # convolutional layer
                cavmae_ft.module.visual_encoder.blocks[11].attn.proj,             # last attention projection
                cavmae_ft.module.visual_encoder.blocks[11].mlp.layers[1].linear,  # last MLP linear
                cavmae_ft.module.visual_encoder.norm                              # final LN
            ],

            targets=[ClassifierOutputTarget(torch.argmax(labels, dim=1).squeeze())],

            reshape_transform=reshape_transform_temporal,
            dump_file_path=dump_file_path,
            apply_mask=True,
            skip_if_mislabeled=False
        )

        # # Overlay the gradient map over the video
        # # after pooling both (video and its heatmap)
        # if rgb_video is not None and grayscale_video is not None:
        #     rgb_video, grayscale_video = pool_cam(rgb_video, grayscale_video)
        #     overlayed_video = []

        #     for frame_idx in range(rgb_video.shape[0]):
        #         overlayed_video.append(
        #             show_cam_on_image(rgb_video[frame_idx], grayscale_video[frame_idx], use_rgb=True)
        #         )

        #     # video_name = video_name.partition(".")[0]
        #     root_file_path = Path(f'/mnt/d/projects/MAVOS-DD-GenClassifer/subset/{language}/{generative_method}/{video_name}')
        #     os.makedirs(f"/mnt/d/projects/MAVOS-DD-GenClassifer/subset/{language}/{generative_method}/{video_name}", exist_ok=True)
        #     # video_dump_path = Path(f'{root_file_path}/rgb_original_video.npy')
        #     heatmap_dump_path = Path(f'{root_file_path}/heatmap_grayscale.npy')
        #     # overlayed_dump_path = Path(f'{root_file_path}/overlayed_video.npy')
            
        #     # video_dump_path.parent.mkdir(exist_ok=True, parents=True)

        #     print(f'Generating {root_file_path}...')

        #     # np.save(video_dump_path, rgb_video)
        #     np.save(heatmap_dump_path, grayscale_video)
        #     # np.save(overlayed_dump_path, np.stack(overlayed_video))

        #     # dump_cam_to_disk(np.stack(overlayed_video), dump_file_path, fps=1)