import numpy as np

def swish(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x_arr = np.asarray(x, dtype=float)
    sigmoid = np.exp(-np.logaddexp(0.0, -x_arr))
    return x_arr *  sigmoid
    pass