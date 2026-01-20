import numpy as np
from collections import defaultdict

runs = [
    {
        "closed-set": {"mAP": 0.9416807983683688, "mAUC": 0.9404342218051778, "acc": 0.8541587542726927},
        "open-model": {"mAP": 0.9199387967676368, "mAUC": 0.9234657865546922, "acc": 0.8275430794886047},
        "open-language": {"mAP": 0.9052083063446446, "mAUC": 0.9064072806271114, "acc": 0.8437042774765471},
        "open-set": {"mAP": 0.9058205512338395, "mAUC": 0.9141571001264848, "acc": 0.82807540071325},
    },
    {
        "closed-set": {"mAP": 0.9171511156093652, "mAUC": 0.9150574525247712, "acc": 0.8312761109001139},
        "open-model": {"mAP": 0.8993232353396027, "mAUC": 0.9022832518767928, "acc": 0.7772651473040578},
        "open-language": {"mAP": 0.8731559066310093, "mAUC": 0.8748735836687945, "acc": 0.8026938634994406},
        "open-set": {"mAP": 0.882911806074262, "mAUC": 0.8893921907843043, "acc": 0.7752086844064742},
    },
    {
        "closed-set": {"mAP": 0.93323464505923, "mAUC": 0.9314829790369026, "acc": 0.8496961640714015},
        "open-model": {"mAP": 0.9270954521044427, "mAUC": 0.9314002509778557, "acc": 0.8358810450250139},
        "open-language": {"mAP": 0.9003697831569042, "mAUC": 0.9016069642453873, "acc": 0.843144848954299},
        "open-set": {"mAP": 0.9115712102773753, "mAUC": 0.9201355257831236, "acc": 0.8331308539405102},
    },
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
        std = values.std(ddof=0)  # sample standard deviation
        print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    print()
