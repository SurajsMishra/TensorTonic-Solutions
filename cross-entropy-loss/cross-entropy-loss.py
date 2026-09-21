import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    correct_class_probs = y_pred_arr[np.arange(len(y_true_arr)), y_true_arr]
    loss = -np.log(correct_class_probs)
    return float(np.mean(loss))
    pass