import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    a = np.atleast_2d(anchor)
    p = np.atleast_2d(positive)
    n = np.atleast_2d(negative)
    d_pos = np.sum((a-p) ** 2, axis=1)
    d_neg = np.sum((a-n) ** 2, axis=1)
    losses = np.maximum(0.0, d_pos - d_neg + margin)
    return float(np.mean(losses))
    pass