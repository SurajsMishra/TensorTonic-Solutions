import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)
    sample_losses = np.maximum(0.0, margin - (y_true_arr * y_score_arr))
    if reduction == "mean":
        loss = np.mean(sample_losses)
    elif reduction == "sum":
        loss = np.sum(sample_losses)
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return float(loss)
    pass