import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """
    Returns the float64 forget-gate values.
    """
    h_prev = np.array (h_prev)
    x_t = np.asarray(x_t)
    if h_prev.ndim == 1:
        concat = np.concatenate([h_prev, x_t], axis=0)
        out = sigmoid(concat @ W_f.T + b_f)
    else:
        concat = np.concatenate([h_prev, x_t], axis=-1)
        out = sigmoid(concat @ W_f.T + b_f)
    return out.astype(np.float64)
    pass