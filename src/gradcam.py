import argparse
import datasets
import torch
from torch.utils.data import DataLoader

from models.video_cav_mae import VideoCAVMAEFT
from mavosdd_dataset import MavosDD

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
print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(args.model_weights_path, len(miss), len(unexpected)))

print(cavmae_ft)
