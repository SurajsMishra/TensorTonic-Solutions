import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X_c = np.array(X) - np.mean(X, axis = 0)
    return (X_c.T @ X_c) / (len(X)-1)
    pass