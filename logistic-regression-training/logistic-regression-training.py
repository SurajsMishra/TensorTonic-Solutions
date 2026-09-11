import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    N,D = X.shape
    w = np.zeros(D)
    b = 0.0
    for i in range(steps):
        z = X @ w+b
        p = _sigmoid(z)
        dw = (X.T @(p-y)) / N
        db = np.mean(p-y)
        w -= lr*dw
        b -= lr*db
    return w, float(b)
    pass