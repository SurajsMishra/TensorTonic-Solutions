import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    X_arr = np.array(X, dtype=float)
    N,D = X_arr.shape
    centered = X_arr- np.mean(X_arr, axis=0)
    cov = (centered.T @ centered) / (N-1)
    std = np.std(X_arr, axis= 0, ddof =1)
    denominator = np.outer(std, std)
    with np.errstate(divide = 'ignore', invalid='ignore'):
        corr = cov/denominator
    return corr
    pass