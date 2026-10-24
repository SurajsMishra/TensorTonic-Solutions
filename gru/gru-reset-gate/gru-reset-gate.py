import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
          W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """
    Returns the float64 reset-gate values.
    """
    h_prev = np.asarray(h_prev, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    concat = np.concatenate([h_prev, x_t], axis=-1)
    affine = np.dot(concat, W_r.T)+ b_r
    return sigmoid(affine).astype(np.float64)
    pass