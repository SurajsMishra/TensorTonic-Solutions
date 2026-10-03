import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    arr = np.array(X, dtype=float)
    if arr.ndim == 1:
        mask = np.isnan(arr)
        if np.all(mask):
            arr[:] = 0.0
        else:
            observed = arr[~mask]
            fill_val = np.mean(observed) if strategy == "mean" else np.median(observed)
            arr[mask] = fill_val
        return arr
    for col_idx in range(arr.shape[1]):
        col= arr[:, col_idx]
        mask = np.isnan(col)
        if np.all(mask):
            arr[:, col_idx] = 0.0
        else:
            observed = col[~mask]
            fill_val = np.mean(observed) if strategy == "mean" else np.median(observed)
            arr[mask, col_idx] = fill_val
    return arr
    pass