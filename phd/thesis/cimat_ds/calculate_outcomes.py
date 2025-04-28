import torch

outcomes = ["TP", "TN", "FP", "FN"]


def calculate_outcomes(
    outputs, targets, use_threshold=False, use_sigmoid=False, p_threshold=0.5
):
    """
    Calculate outcomes for binary semantic segmentation.

    Args:
        outputs (torch.Tensor): Model predictions (logits), shape: (batch_size, 1, H, W).
        targets (torch.Tensor): Ground truth masks, shape: (batch_size, 1, H, W).

    Returns:
        dict: Dictionary containing outcomes of the segmentation: TP, TN, FP, FN
    """
    # Binarize predictions
    # Use sigmoid if indicated
    if use_sigmoid:
        outputs = torch.sigmoid(outputs)
    # Calculate threshold
    threshold = p_threshold if use_threshold else torch.mean(outputs)
    # Binarize
    preds = (outputs > threshold).int()
    targets = targets.int()
    assert preds.shape == targets.shape, "Mismatch predictions and targets shape!"
    b, c, h, w = targets.shape

    # Flatten tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Calculate TP, TN, FP, FN
    TP = ((preds == 1) & (targets == 1)).sum().item()  # True Positives
    TN = ((preds == 0) & (targets == 0)).sum().item()  # True Negative
    FP = ((preds == 1) & (targets == 0)).sum().item()  # False Positives
    FN = ((preds == 0) & (targets == 1)).sum().item()  # False Negatives

    total_pixels = preds.numel()
    total_outcomes = TP + TN + FP + FN
    assert total_pixels == total_outcomes, "Mismatch pixels - outcomes!"
    assert total_pixels == (b * c * h * w), "Mismatch batch input size - outcomes"
    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }
