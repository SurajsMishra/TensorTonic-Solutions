import math

def binary_focal_loss(predictions: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean binary focal loss as a float.
    """
    total_loss = 0.0
    n = len(predictions)
    for p,y in zip(predictions, targets):
        p_t = p if y== 1 else (1-p)
        focal_loss = -alpha*((1-p_t) ** gamma) * math.log(p_t)
        total_loss += focal_loss

    return float(total_loss / n)
    pass