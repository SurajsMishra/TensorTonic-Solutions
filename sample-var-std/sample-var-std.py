import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    arr = np.array(x, dtype=float)
    var = float(np.var(arr, ddof=1))
    std = float(np.std(arr, ddof=1))
    return {
        "variance": var,
        "standard_deviation": std
    }
    pass