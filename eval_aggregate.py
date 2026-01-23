import numpy as np
from collections import defaultdict

runs = [
    {'closed-set': {'mAP': 0.943953991804009, 'mAUC': 0.9421049285651035, 'acc': 0.842954804405621}, 'open-model': {'mAP': 0.9138433512440727, 'mAUC': 0.9181857751381315, 'acc': 0.8406892718176765}, 'open-language': {'mAP': 0.9039551215464481, 'mAUC': 0.9031649790221143, 'acc': 0.8122041483776573}, 'open-set': {'mAP': 0.899093400864662, 'mAUC': 0.9055425753382107, 'acc': 0.8262334913978916}},
    {'closed-set': {'mAP': 0.9553312764398499, 'mAUC': 0.9544939537468986, 'acc': 0.8831181162172427}, 'open-model': {'mAP': 0.9021364647610051, 'mAUC': 0.904856910818957, 'acc': 0.7969427459699834}, 'open-language': {'mAP': 0.9170435263160028, 'mAUC': 0.9159161873595701, 'acc': 0.8550219468112574}, 'open-set': {'mAP': 0.8962099721491446, 'mAUC': 0.9008861997664952, 'acc': 0.8055414037700357}},
    {'closed-set': {'mAP': 0.9555664006342444, 'mAUC': 0.9548002205293968, 'acc': 0.8690657045195594}, 'open-model': {'mAP': 0.9140683080827607, 'mAUC': 0.9172425840777438, 'acc': 0.8284880489160645}, 'open-language': {'mAP': 0.9210486080247969, 'mAUC': 0.9205192517539786, 'acc': 0.8489542989930287}, 'open-set': {'mAP': 0.9075531170723341, 'mAUC': 0.9129021241114836, 'acc': 0.8290943292706823}},
]


metrics = defaultdict(lambda: defaultdict(list))

for run in runs:
    for dataset, values in run.items():
        for metric, value in values.items():
            metrics[dataset][metric].append(value)

print("Mean ± Std across runs\n")

for dataset in metrics:
    print(f"{dataset}:")
    for metric in metrics[dataset]:
        values = np.array(metrics[dataset][metric])
        mean = values.mean()
        std = values.std(ddof=1)  # sample standard deviation
        print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    print()
