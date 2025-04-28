import torch

# Define the metrics to be calculated for binary segmentation problems
metrics = [
    "precision",
    "recall",
    "f1score",
    "accuracy",
    "specificity",
    "f_iou",  # Foreground IoU
    "b_iou",  # Background IoU
    "m_iou",  # Mean IoU
]


def calculate_metrics(outcomes_dict):
    """
    Calculate metrics for binary semantic segmentation.

    Args:
        outcomes_dict (dictionary): Model outcomes: TP, TN, FP, FN

    Returns:
        dict: Dictionary for metrics values: precision, recall, f1score, accuracy, specificity, IoU (foreground, background, mean)
    """
    # Extract TP, TN, FP, FN from outcomes_dict
    TP = outcomes_dict["TP"]
    TN = outcomes_dict["TN"]
    FP = outcomes_dict["FP"]
    FN = outcomes_dict["FN"]

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

    return {
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
        "accuracy": accuracy,
        "specificity": specificity,
        "f_iou": f_iou,
        "b_iou": b_iou,
        "m_iou": m_iou,
    }


def calculate_roc_metrics(outcomes_dict):
    # Extract TP, TN, FP, FN from outcomes_dict
    TP = outcomes_dict["TP"]
    TN = outcomes_dict["TN"]
    FP = outcomes_dict["FP"]
    FN = outcomes_dict["FN"]

    # Calculate TPR and FPR
    TPR = TP / (TP + FN + 1e-7)
    FPR = FP / (FP + TN + 1e-7)
    return {"tpr": TPR, "fpr": FPR}
