import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    pred = np.asarray(y_pred, dtype=float)
    true = np.asarray(y_true, dtype=float)
    mse = np.mean((pred-true) ** 2)
    return float(mse)
    pass