import torch


def calculate_metrics(outputs, targets):
    """
    Calculate metrics for binary semantic segmentation.

    Args:
        outputs (torch.Tensor): Model predictions (logits), shape: (batch_size, 1, H, W).
        targets (torch.Tensor): Ground truth masks, shape: (batch_size, 1, H, W).

    Returns:
        dict: Dictionary containing accuracy, specificity, sensitivity, dice, and IoU scores
    """
    # Binarize predictions
    threshold = torch.mean(outputs).sum()
    preds = (outputs > threshold).float()

    # Flatten tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Calculate TP, TN, FP, FN
    TP = (preds * targets).sum().item()  # True Positives
    TN = ((1 - preds) * (1 - targets)).sum().item()  # True Negative
    FP = (preds * (1 - targets)).sum().item()  # False Positives
    FN = ((1 - preds) * targets).sum().item()  # False Negatives

    # Metrics calculation
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    specificity = TN / (TN + FP + 1e-7)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-7)
    iou = TP / (TP + FP + FN + 1e-7)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "dice": dice,
        "iou": iou,
    }


def calculate_multiclass_metrics(outputs, targets, num_classes, threshold=0.5):
    """
    Calculate metrics for multiclass semantic segmentation.

    Args:
        outputs (torch.Tensor): Model predictions (logits), shape: (batch_size, num_classes, H, W).
        targets (torch.Tensor): Ground truth masks, shape: (batch_size, H, W).
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary containing metrics for each class and overall overages
    threshold (float): Threshold to binarize predictions (only for logits if necessary).
    """
    # Convert logits to predicted classes
    preds = torch.argmax(outputs, dim=1)
    metrics_per_class = {
        "TP": [],
        "TN": [],
        "FP": [],
        "FN": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "dice": [],
        "iou": [],
    }
    for c in range(num_classes):
        # One-hot encode the ground truth and predictions for class c
        pred_c = (preds == c).float()  # Predicted mask for class c
        target_c = (targets == c).float()  # Ground truth mask for class c

        # Flatten tensors
        pred_c = pred_c.view(-1)
        target_c = target_c.view(-1)

        # Calculate TP, TN, FP, FN
        TP = (pred_c * target_c).sum().item()
        TN = ((1 - pred_c) * (1 - target_c)).sum().item()
        FP = (pred_c * (1 - target_c)).sum().item()
        FN = ((1 - pred_c) * target_c).sum().item()

        # Calculate metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)
        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        specificity = TN / (TN + FP + 1e-7)
        dice = 2 * TP / (2 * TP + FP + FN + 1e-7)
        iou = TP / (TP + FP + FN + 1e-7)

        # Store metrics for this class
        metrics_per_class["TP"].append(TP)
        metrics_per_class["TN"].append(TN)
        metrics_per_class["FP"].append(FP)
        metrics_per_class["FN"].append(FN)
        metrics_per_class["accuracy"].append(accuracy)
        metrics_per_class["precision"].append(precision)
        metrics_per_class["recall"].append(recall)
        metrics_per_class["specificity"].append(specificity)
        metrics_per_class["dice"].append(dice)
        metrics_per_class["iou"].append(iou)

    # Average metrics across all classes
    metrics_avg = {
        metric: (
            sum(values) / num_classes
            if metric not in ["TP", "TN", "FP", "FN"]
            else sum(values)
        )
        for metric, values in metrics_per_class.items()
    }

    return {"per_class": metrics_per_class, "average": metrics_avg}
