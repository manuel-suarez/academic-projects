import torch
from itertools import product

outcomes = ["TP", "TN", "FP", "FN"]


def calculate_outcomes(
    outputs, targets, num_classes, use_argmax=True, p_threshold=0.5, outcomes=outcomes
):
    """
    Calculate outcomes for multiclass semantic segmentation.

    Args:
        outputs (torch.Tensor): Model predictions (logits), shape: (batch_size, 1, H, W).
        targets (torch.Tensor): Ground truth masks, shape: (batch_size, 1, H, W).
        num_classes (int): Num of classes to predict
        use_argmax (boolean): When we need to use argmax for accuracy, etc, metrics or threshold for ROC/AUC metrics calculation
        p_threshold (float): Threshold value for ROC/AUC metrics calculation

    Returns:
        dict: Dictionary containing per class outcomes of the segmentation: TP, TN, FP, FN
    """
    # Apply softmax to get multiclass probabilities
    outputs = torch.softmax(outputs, dim=1)
    # If we are not using threshold then we use argmax to get global class predictions
    if use_argmax:
        preds = torch.argmax(outputs, dim=1)
    else:
        preds = (outputs > p_threshold).float()

    # Initialize outcomes per class
    class_outcomes = {
        f"class{num_class}_{outcome}": 0
        for outcome, num_class in product(outcomes, range(num_classes))
    }

    # Calculate TP, TN, FP, FN for each class
    for num_class in range(num_classes):
        # Flatten tensors
        if use_argmax:
            preds_class = (preds == num_class).float()
        else:
            preds_class = preds[:, num_class, :, :]
        targets = (targets == num_class).float()

        TP = (preds_class * targets).sum().item()  # True Positives
        TN = ((1 - preds_class) * (1 - targets)).sum().item()  # True Negative
        FP = (preds_class * (1 - targets)).sum().item()  # False Positives
        FN = ((1 - preds_class) * targets).sum().item()  # False Negatives

        class_outcomes[f"class{num_class}_TP"] = TP
        class_outcomes[f"class{num_class}_TN"] = TN
        class_outcomes[f"class{num_class}_FP"] = FP
        class_outcomes[f"class{num_class}_FN"] = FN

    return class_outcomes
