import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    if a_arr.ndim == 1:
        a_arr = np.atleast_2d(a_arr)
        b_arr = np.atleast_2d(b_arr)
    d = np.linalg.norm(a_arr - b_arr, axis=1)
    loss_i = y_arr *(d**2) + (1.0 - y_arr) * (np.maximum(0.0, margin-d) ** 2)
    if reduction == "mean":
        loss = np.mean(loss_i)
    elif reduction == "sum":
        loss = np.sum(loss_i)
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return float(loss)
    pass