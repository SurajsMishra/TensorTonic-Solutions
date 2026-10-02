import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    X_arr = np.array(X, dtype=np.float64)
    mean = np.mean(X_arr, axis=axis, keepdims=True)
    std = np.std(X_arr, axis=axis, keepdims=True)
    safe_std = np.where(std>eps, std, 1.0)
    z_scores = (X_arr-mean)/ safe_std
    z_scores = np.where(std>eps, z_scores, 0.0)
    return z_scores
    pass