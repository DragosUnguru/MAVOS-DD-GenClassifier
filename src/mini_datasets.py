import argparse
import datasets

from torch.utils.data import DataLoader
from mavosdd_dataset_multiclass import MavosDD
from datasets import concatenate_datasets
from mavosdd_dataset import MavosDD as MavosDDeepfake

def get_mini_test_set(batch_size: int, audio_conf: object, num_workers: int = 4, shuffle: bool = False) -> DataLoader:
    input_path = '/home/fl488644/datasets/MAVOS-DD'

    mavos_dd = datasets.Dataset.load_from_disk(input_path).filter(
                    lambda sample: sample['split']=="train" and sample['open_set_model']==False and sample["open_set_language"]==False
                    and (sample['generative_method'] != 'real' or sample['audio_generative_method'] != 'real'))
    mavos_dd = mavos_dd.shuffle(seed=1234)
    # mavos_dd = mavos_dd.add_column(
    #     "stratify_key",
    #     [f"{video}_{audio}" for video, audio in zip(mavos_dd["generative_method"], mavos_dd["audio_generative_method"])]
    # )

    # mavos_dd = mavos_dd.train_test_split(
    #     test_size=0.20,
    #     seed=42,
    #     # stratify_by_column="generative_method"
    # )

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


def get_mini_train_set_deepfake_detection(input_path: str) -> DataLoader:

    mavos_dd = datasets.Dataset.load_from_disk(input_path).filter(
                    lambda sample: sample['split']=="train" and sample['open_set_model']==False and sample["open_set_language"]==False)

    unique_gen_methods = list(set(mavos_dd['generative_method'])) + list(set(mavos_dd['audio_generative_method']))
    unique_languages = list(set(mavos_dd['language']))
    dict_count_gen_methods = {}
    list_datasets = []
    for language in unique_languages:
        mavos_language = mavos_dd.filter(lambda sample: sample['language'] == language)
        mavos_language = mavos_language.shuffle(seed=1234)
        list_datasets.append(mavos_language.select(range(300)))
    #     print(f"Language {language} size: {len(mavos_language)}")
    #     for gen_method in unique_gen_methods:
    #         mavos_gen_method = mavos_language.filter(lambda sample: sample['generative_method'] == gen_method or sample['audio_generative_method'] == gen_method)
    #         print(f"Language {language} generative method {gen_method} size: {len(mavos_gen_method)}")
    #         if gen_method not in dict_count_gen_methods:
    #             dict_count_gen_methods[gen_method] = len(mavos_gen_method)
    #         else:
    #             dict_count_gen_methods[gen_method] += len(mavos_gen_method)
    #     dict_count_gen_methods['real'] = len(mavos_language.filter(lambda sample: sample['label']=="real"))
    # print(dict_count_gen_methods)
    # exit()

    # for audio_type in audio_labels:
    #     mavos_dd_per_type = mavos_dd.filter(lambda sample: sample['audio_generative_method'] == audio_type)
    #     list_datasets.append(mavos_dd_per_type.select(range(min(len(mavos_dd_per_type), 20))))
    mavos_dd = concatenate_datasets(list_datasets)

    return mavos_dd