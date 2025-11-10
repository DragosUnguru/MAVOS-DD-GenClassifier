import argparse
import os
import torch
import datasets
from torch.utils.data import DataLoader
from models.video_cav_mae import VideoCAVMAEFT
from traintest_ft import train
import warnings

from mavosdd_dataset import MavosDD
# from mavosdd_dataset_multiclass import MavosDD
from mini_datasets import get_mini_test_set, get_mini_train_set_deepfake_detection



parser = argparse.ArgumentParser(description='Video CAV-MAE')
parser.add_argument('--data-train', type=str, help='path to train data csv')
parser.add_argument('--data-val', type=str, help='path to val data csv')
parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
parser.add_argument("--dataset_mean", default=-5.081, type=float, help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", default=4.4849, type=float, help="the dataset audio spec std, used for input normalization")
parser.add_argument("--noise", default=False, type=bool, help="add noise to the input")

parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument('--n-epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--n_classes', default=2, type=int, help='Num of classes to be classified')
parser.add_argument('--save-dir', default='checkpoints/trainable_mask_binary_classification', type=str, help='directory to save checkpoints')
parser.add_argument('--pretrain_path', default='/mnt/d/projects/MAVOS-DD-GenClassifer/checkpoints/stage-3.pth', type=str, help='path to pretrain model')
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
parser.add_argument("--mae_loss_weight", type=float, default=3.0, help="weight for mae loss")
parser.add_argument('--save_model', default=True)
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate')
parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', default=None)
parser.add_argument("--n_print_steps", default=100, type=int)
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument('--warmup',type=bool, default=True)
parser.add_argument('--head_lr', type=int, default=50)
parser.add_argument('--mask_loss_lambda', type=float, default=0.1)
parser.add_argument('--train_mask', type=bool, default=True)
parser.add_argument('--mask_ratio', type=float, default=0.75)

parser.add_argument("--wa_start", type=int, default=1, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=10, help="which epoch to end weight averaging in finetuning")

args = parser.parse_args()

im_res = 224
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mode':'train',
            'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'mode':'eval',
            'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))

# Construct dataloader
input_path = "/mnt/d/projects/datasets/MAVOS-DD"
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

# mavos_dd = datasets.Dataset.load_from_disk(input_path)

# train_loader = DataLoader(
#     MavosDD(
#         mavos_dd.filter(lambda sample: sample['split']=="train" and (sample['generative_method'] != "real" or sample['audio_generative_method'] != "real")),
#         input_path,
#         audio_conf,
#         stage=2,
#         video_class_name_to_idx=video_labels,
#         audio_class_name_to_idx=audio_labels),
#     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
# )
mavos_dd = get_mini_train_set_deepfake_detection(input_path)
train_loader = DataLoader(
    MavosDD(
        mavos_dd,
        input_path,
        audio_conf,
        # video_class_name_to_idx=video_labels,
        # audio_class_name_to_idx=audio_labels,
        stage=2,
        custom_file_path=False),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False
)
val_loader = DataLoader(
    MavosDD(
        datasets.Dataset.load_from_disk(input_path).filter(lambda sample: sample['split']=="validation"),# and (sample['generative_method'] != "real" or sample['audio_generative_method'] != "real")),
        input_path,
        val_audio_conf,
        stage=2,
        custom_file_path=False),
        # video_class_name_to_idx=video_labels,
        # audio_class_name_to_idx=audio_labels),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True
)

print(f"Using Train: {len(train_loader)}, Eval: {len(val_loader)}")

# Load pre-trained AVFF model & weights
cavmae_ft = VideoCAVMAEFT(n_classes=args.n_classes)#len(class_name_to_label_mapping))
if not isinstance(cavmae_ft, torch.nn.DataParallel):
    cavmae_ft = torch.nn.DataParallel(cavmae_ft)

if args.pretrain_path is not None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl_weight = torch.load(args.pretrain_path, map_location=device)

    print(f'Running on {device}')

    # Ignore weights of last FC layer
    # del mdl_weight['module.mlp_head.fc3.weight']
    # del mdl_weight['module.mlp_head.fc3.bias']

    miss, unexpected = cavmae_ft.load_state_dict(mdl_weight, strict=False)

    print('Missing: ', miss)
    print('Unexpected: ', unexpected)
    print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(args.pretrain_path, len(miss), len(unexpected)))
else:
    warnings.warn("Note you are finetuning a model without any finetuning.")

print("\n Creating experiment directory: %s"%args.save_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Train model
print("Now start training for %d epochs"%args.n_epochs)
train(cavmae_ft, train_loader, val_loader, args)
