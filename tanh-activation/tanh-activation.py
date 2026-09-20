import numpy as np

def tanh(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x_arr = np.asarray(x, dtype=float)
    return np.tanh(x_arr)
    pass