import math
import numpy as np

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    arr = np.asarray(x, dtype=float)
    v_erf = np.vectorize(math.erf)
    return 0.5 * arr * (1.0 + v_erf(arr / np.sqrt(2)))
    pass