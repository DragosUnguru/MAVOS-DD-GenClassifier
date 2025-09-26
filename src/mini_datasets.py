import argparse
import datasets

from torch.utils.data import DataLoader
from mavosdd_dataset_multiclass import MavosDD


def get_mini_test_set(batch_size: int, audio_conf: object, num_workers: int = 4, shuffle: bool = False) -> DataLoader:
    input_path = '/mnt/d/projects/datasets/MAVOS-DD'

    mavos_dd = datasets.Dataset.load_from_disk(input_path).filter(
                    lambda sample: sample['split']=="train" and sample['open_set_model']==False and sample["open_set_language"]==False
                    and (sample['generative_method'] != 'real' or sample['audio_generative_method'] != 'real'))

    # mavos_dd = mavos_dd.add_column(
    #     "stratify_key",
    #     [f"{video}_{audio}" for video, audio in zip(mavos_dd["generative_method"], mavos_dd["audio_generative_method"])]
    # )

    mavos_dd = mavos_dd.train_test_split(
        test_size=0.20,
        seed=42,
        # stratify_by_column="generative_method"
    )

    mavos_dd = mavos_dd["test"]

    dataloader = DataLoader(
        MavosDD(
            mavos_dd,
            input_path,
            audio_conf,
            stage=2),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return dataloader
