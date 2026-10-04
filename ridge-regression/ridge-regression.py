import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    d = X_arr.shape[1]
    XTX = X_arr.T @ X_arr
    reg_matrix = XTX + lam * np.eye(d)
    w = np.linalg.inv(reg_matrix)@ X_arr.T @ y_arr
    return w.tolist()
    pass