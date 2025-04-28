import torch
from itertools import product

# Define the metrics to be calculated for binary segmentation problems
metrics = [
    "precision",
    "recall",
    "f1score",
    "accuracy",
    "specificity",
    "f_iou",  # Foreground IoU
    "b_iou",  # Background IoU
    "m_iou",  # Mean IoU (for a certain class vs others)
]


def calculate_metrics(outcomes_dict, num_classes, metrics=metrics):
    """
    Calculate metrics for multiclass semantic segmentation.

    Args:
        outcomes_dict (dictionary): Model outcomes: TP, TN, FP, FN
        num_classes (int): Num of classes

    Returns:
        dict: Dictionary for metrics values per class: precision, recall, f1score, accuracy, specificity, IoU
    """
    class_metrics = {
        f"class{num_class}_{metric}": 0
        for metric, num_class in product(metrics, range(num_classes))
    }
    for num_class in range(num_classes):
        # Extract TP, TN, FP, FN from outcomes_dict
        TP = outcomes_dict[f"class{num_class}_TP"]
        TN = outcomes_dict[f"class{num_class}_TN"]
        FP = outcomes_dict[f"class{num_class}_FP"]
        FN = outcomes_dict[f"class{num_class}_FN"]

        # Metrics calculation
        # Pixel classification metrics
        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        f1score = (2 * TP) / (2 * TP + FP + FN + 1e-7)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)
        specificity = TN / (TN + FP + 1e-7)
        # Semantic segmentation metrics
        f_iou = TP / (TP + FP + FN + 1e-7)
        b_iou = TN / (TN + FN + FP + 1e-7)
        m_iou = (f_iou + b_iou) / 2

        class_metrics[f"class{num_class}_precision"] = precision
        class_metrics[f"class{num_class}_recall"] = recall
        class_metrics[f"class{num_class}_f1score"] = f1score
        class_metrics[f"class{num_class}_accuracy"] = accuracy
        class_metrics[f"class{num_class}_specificity"] = specificity
        class_metrics[f"class{num_class}_f_iou"] = f_iou
        class_metrics[f"class{num_class}_b_iou"] = b_iou
        class_metrics[f"class{num_class}_m_iou"] = m_iou

    return class_metrics


def calculate_roc_metrics(outcomes_dict, num_classes):
    class_metrics = {
        f"class{num_class}_{metric}": 0
        for num_class, metric in product(range(num_classes), ["tpr", "fpr"])
    }
    for num_class in range(num_classes):
        # Extract TP, TN, FP, FN from outcomes_dict
        TP = outcomes_dict[f"class{num_class}_TP"]
        TN = outcomes_dict[f"class{num_class}_TN"]
        FP = outcomes_dict[f"class{num_class}_FP"]
        FN = outcomes_dict[f"class{num_class}_FN"]

        # Calculate TPR and FPR
        TPR = TP / (TP + FN + 1e-7)
        FPR = FP / (FP + TN + 1e-7)

        class_metrics[f"class{num_class}_tpr"] = TPR
        class_metrics[f"class{num_class}_fpr"] = FPR

    return class_metrics
