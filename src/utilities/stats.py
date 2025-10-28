import numpy as np
from scipy import stats
from sklearn import metrics
import torch

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.
    
    Args:
        output: 2d array, (samples_num, classes_num), logits before activation.
        target: 2d array, (samples_num, classes_num), one-hot or multi-hot.
    
    Returns:
        stats: dict containing overall metrics + per-class metrics.
    """
    output = np.array(output)
    target = np.array(target)
    n_classes = target.shape[-1]

    # Detect mode: multi-label if any row has >1 positive
    multilabel = np.any(target.sum(axis=1) > 1)

    if multilabel:
        # ======= Multi-label: sigmoid per class =======
        probs = torch.sigmoid(torch.tensor(output)).numpy()
        preds = np.round(probs)

        # Overall accuracy (not very meaningful in multi-label)
        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(probs, 1))

        per_class_stats = []
        ap_scores, auc_scores = [], []

        for k in range(n_classes):
            avg_precision = metrics.average_precision_score(target[:, k], probs[:, k])
            ap_scores.append(avg_precision)

            try:
                auc = metrics.roc_auc_score(target[:, k], probs[:, k])
                auc_scores.append(auc)

                precisions, recalls, _ = metrics.precision_recall_curve(target[:, k], probs[:, k])
                fpr, tpr, _ = metrics.roc_curve(target[:, k], probs[:, k])

                save_every_steps = 1000
                per_class_stats.append({
                    'AP': avg_precision,
                    'AUC': auc,
                    'precisions': precisions[::save_every_steps],
                    'recalls': recalls[::save_every_steps],
                    'fpr': fpr[::save_every_steps],
                    'fnr': 1. - tpr[::save_every_steps]
                })
            except Exception:
                per_class_stats.append({
                    'AP': avg_precision,
                    'AUC': -1,
                    'precisions': -1,
                    'recalls': -1,
                    'fpr': -1,
                    'fnr': -1
                })

        # Macro/micro averages
        macro_ap = np.mean(ap_scores)
        macro_auc = np.mean(auc_scores) if len(auc_scores) > 0 else -1
        micro_ap = metrics.average_precision_score(target.ravel(), probs.ravel(), average="micro")
        try:
            micro_auc = metrics.roc_auc_score(target.ravel(), probs.ravel(), average="micro")
        except Exception:
            micro_auc = -1

        stats = {
            "mode": "multilabel",
            "accuracy": acc,
            "AP_macro": macro_ap,
            "AP_micro": micro_ap,
            "AUC_macro": macro_auc,
            "AUC_micro": micro_auc,
            "per_class": per_class_stats
        }

    else:
        # ======= Multi-class: softmax across classes =======
        probs = torch.softmax(torch.tensor(output), dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        true = np.argmax(target, axis=1)

        acc = metrics.accuracy_score(true, preds)
        f1_macro = metrics.f1_score(true, preds, average="macro")
        f1_micro = metrics.f1_score(true, preds, average="micro")

        per_class_stats = []
        ap_scores, auc_scores = [], []

        for k in range(n_classes):
            avg_precision = metrics.average_precision_score(target[:, k], probs[:, k])
            ap_scores.append(avg_precision)

            try:
                auc = metrics.roc_auc_score(target[:, k], probs[:, k])
                auc_scores.append(auc)

                precisions, recalls, _ = metrics.precision_recall_curve(target[:, k], probs[:, k])
                fpr, tpr, _ = metrics.roc_curve(target[:, k], probs[:, k])

                per_class_stats.append({
                    'AP': avg_precision,
                    'AUC': auc,
                    'precisions': precisions,
                    'recalls': recalls,
                    'fpr': fpr,
                    'fnr': 1. - tpr
                })
            except Exception:
                per_class_stats.append({
                    'AP': avg_precision,
                    'AUC': -1,
                    'precisions': -1,
                    'recalls': -1,
                    'fpr': -1,
                    'fnr': -1
                })

        # Macro/micro averages
        macro_ap = np.mean(ap_scores)
        macro_auc = np.mean(auc_scores) if len(auc_scores) > 0 else -1
        micro_ap = metrics.average_precision_score(target.ravel(), probs.ravel(), average="micro")
        try:
            micro_auc = metrics.roc_auc_score(target.ravel(), probs.ravel(), average="micro")
        except Exception:
            micro_auc = -1

        stats = {
            "mode": "multiclass",
            "accuracy": acc,
            "F1_macro": f1_macro,
            "F1_micro": f1_micro,
            "AP_macro": macro_ap,
            "AP_micro": micro_ap,
            "AUC_macro": macro_auc,
            "AUC_micro": micro_auc,
            "per_class": per_class_stats
        }

    return stats

def calculate_stats_2(output, target):
    """Calculate statistics including mAP, AUC, etc.
    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        try:
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

            # Precisions, recalls
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                target[:, k], output[:, k])

            # FPR, TPR
            (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

            save_every_steps = 1000     # Sample statistics to reduce size
            dict = {'precisions': precisions[0::save_every_steps],
                    'recalls': recalls[0::save_every_steps],
                    'AP': avg_precision,
                    'fpr': fpr[0::save_every_steps],
                    'fnr': 1. - tpr[0::save_every_steps],
                    'auc': auc,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
        except Exception as e:
            dict = {'precisions': -1,
                    'recalls': -1,
                    'AP': avg_precision,
                    'fpr': -1,
                    'fnr': -1,
                    'auc': -1,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
            print('class {:s} no true sample'.format(str(k)))
        stats.append(dict)

    return stats