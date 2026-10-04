import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    X_T = X_arr.T
    w=np.linalg.inv(X_T @ X_arr) @ X_T @ y_arr
    return w.tolist()
    pass