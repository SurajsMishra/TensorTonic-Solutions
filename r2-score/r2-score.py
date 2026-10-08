import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)
    ss_res = np.sum((y_t - y_p)**2)
    y_mean = np.mean(y_t)
    ss_tot = np.sum((y_t - y_mean)**2)
    if ss_tot == 0:
        return 1.0 if np.array_equal(y_t, y_p) else 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2)
    pass