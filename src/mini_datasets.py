import datasets

from torch.utils.data import DataLoader
from mavosdd_dataset_multiclass import MavosDD
from datasets import concatenate_datasets

def get_mini_test_set(batch_size: int, audio_conf: object, num_workers: int = 4, shuffle: bool = False) -> DataLoader:
    input_path = '/mnt/d/projects/datasets/MAVOS-DD'


    mavos_dd = datasets.Dataset.load_from_disk(input_path).filter(
                    lambda sample: sample['split']=="train" and sample['open_set_model']==False and sample["open_set_language"]==False
                    and (sample['generative_method'] != 'real' or sample['audio_generative_method'] != 'real'))
    mavos_dd = mavos_dd.shuffle(seed=1234)

    # mavos_dd = mavos_dd["test"].select(range(100))
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
    list_datasets = []
    for video_type in video_labels:
        mavos_dd_per_type = mavos_dd.filter(lambda sample: sample['generative_method'] == video_type)
        list_datasets.append(mavos_dd_per_type.select(range(min(len(mavos_dd_per_type), 25))))
    # for audio_type in audio_labels:
    #     mavos_dd_per_type = mavos_dd.filter(lambda sample: sample['audio_generative_method'] == audio_type)
    #     list_datasets.append(mavos_dd_per_type.select(range(min(len(mavos_dd_per_type), 20))))
    mavos_dd = concatenate_datasets(list_datasets)
    dataloader = DataLoader(
        MavosDD(
            mavos_dd,
            input_path,
            audio_conf, video_class_name_to_idx=video_labels, audio_class_name_to_idx=audio_labels,
            stage=2),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return dataloader
