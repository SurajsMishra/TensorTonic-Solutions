import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    k = len(predictions)
    loss = 0.0
    for i,p in enumerate(predictions):
        if i == target:
            q = (1-epsilon)+(epsilon/k)
        else:
            q = epsilon/k
        loss -= q* math.log(p)
    return float(loss)
    pass