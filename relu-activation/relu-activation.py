import numpy as np

def relu(x) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x_arr = np.array(x, dtype=float)
    return np.array(np.maximum(0.0, x_arr))
    