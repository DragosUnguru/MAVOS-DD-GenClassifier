import os
import random
from datasets import Dataset

MAVOS_DD_DATASET_PATH = os.path.join("..", "datasets", "MAVOS-DD")


mavos_dd = Dataset.load_from_disk(MAVOS_DD_DATASET_PATH)
index = random.randint(0, len(mavos_dd))
print("Sample: ", mavos_dd[index])
# print("Train samples: ", mavos_dd.filter(lambda sample: sample['split']=="train"))
print("Test samples open set language and model: ", mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model'] and sample["open_set_language"]))
# print("Validation samples: ", mavos_dd.filter(lambda sample: sample['split']=="validation"))
# print("Real samples: ", mavos_dd.filter(lambda sample: sample['label']=="real"))
# print("Fake samples: ", mavos_dd.filter(lambda sample: sample['label']=="fake"))
# languages = ["arabic", "english", "german", "hindi", "mandarin", "romanian", "russian", "spanish"]
# methods = ["echomimic", "hififace", "inswapper", "liveportrait", "memo", "roop", "sonic"]
# for language in languages:
#     # print(f"Language {language} real: ", mavos_dd.filter(lambda sample: sample['label']=="real" and sample["language"]==language))
#     for method in methods:
#         print(f"Language {language} method {method}: ", mavos_dd.filter(lambda sample: sample['label']=="fake" and sample["language"]==language  and sample["generative_method"]==method))