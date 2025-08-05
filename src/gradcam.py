import argparse
import datasets
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.video_cav_mae import VideoCAVMAEFT
from mavosdd_dataset import MavosDD

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, RawScoresOutputTarget
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)

parser = argparse.ArgumentParser(description='Video CAV-MAE')

parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--model_weights_path', default='/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/avff_mavos.pth', type=str, help='the path to the CAVMAEFT model weights on the MAVOS-DD dataset')
parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--dataset_mean", default=-5.081, type=float, help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", default=4.4849, type=float, help="the dataset audio spec std, used for input normalization")
parser.add_argument("--noise", default=False, type=bool, help="add noise to the input")
parser.add_argument("--mae_loss_weight", type=float, default=3.0, help="weight for mae loss")
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")

args = parser.parse_args()

im_res = 224
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mode':'train',
            'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'mode':'eval',
            'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))


input_path = "/mnt/d/projects/datasets/MAVOS-DD"
class_name_to_label_mapping = {
    'real': 0,
    'echomimic': 1,
    'hififace': 2,
    'inswapper': 3,
    'liveportrait': 4,
    'memo': 5,
    'roop': 6,
    'sonic': 7,
}

mavos_dd = datasets.Dataset.load_from_disk(input_path)
test_loader = DataLoader(
    MavosDD(
        mavos_dd.filter(lambda sample: sample['split']=="test"),
        input_path,
        audio_conf,
        stage=2),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
)

cavmae_ft = VideoCAVMAEFT(n_classes=2)
if not isinstance(cavmae_ft, torch.nn.DataParallel):
    cavmae_ft = torch.nn.DataParallel(cavmae_ft)

mdl_weight = torch.load(args.model_weights_path, map_location='cpu')
miss, unexpected = cavmae_ft.load_state_dict(mdl_weight, strict=False)

print("Missing: ", miss)
print("Unexpected: ", unexpected)

target_layers = [cavmae_ft.module.visual_encoder.patch_embedding.projection]
audio_input, video_input, labels, video_path = next(iter(test_loader))
# Note: input_tensor can be a batch tensor with several images!

# We have to specify the target we want to generate the CAM for.
# targets = [RawScoresOutputTarget()]
class FeatureSumTarget:
    def __call__(self, model_output):
        return model_output.sum()

targets = [FeatureSumTarget()]

mid_frame_idx = video_input.shape[1] // 2
frame = video_input[0, :, mid_frame_idx, :, :].clone()

mean=[0.4850, 0.4560, 0.4060]
std=[0.2290, 0.2240, 0.2250]

for t, m, s in zip(frame, mean, std):
    t.mul_(s).add_(m)

rgb_img = frame.permute(1, 2, 0).cpu().numpy()
rgb_img = np.clip(rgb_img, 0, 1).astype(np.float32)

# Construct the CAM object once, and then re-use it on many images.
with GradCAM(model=cavmae_ft.module.visual_encoder, target_layers=target_layers) as cam:
    # video_input = video_input.permute(0, 2, 1, 3, 4)
    grayscale_cam = cam(input_tensor=video_input, targets=targets)
    cam_for_frame = grayscale_cam[0, grayscale_cam.shape[1] // 2]

    # print(grayscale_cam.shape)
    # print('===========')
    # print(grayscale_cam)

    visualization = show_cam_on_image(rgb_img, cam_for_frame, use_rgb=True)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    # You can also get the model outputs without having to redo inference
    # model_outputs = cam.outputs
    cv2.imwrite('/mnt/d/projects/MAVOS-DD-GenClassifer/exp/cam.jpg', visualization)

# print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(args.model_weights_path, len(miss), len(unexpected)))
# print(cavmae_ft)
