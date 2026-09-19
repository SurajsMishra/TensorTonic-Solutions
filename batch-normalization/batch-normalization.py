import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x_arr = np.array(x, dtype = np.float64)
    gamma_arr = np.array(gamma, dtype=np.float64)
    beta_arr = np.array(beta, dtype = np.float64)
    if x_arr.ndim == 2:
        axis= 0
        gamma_broadcast = gamma_arr
        beta_broadcast = beta_arr
    elif x_arr.ndim == 4:
        axis=(0,2,3)
        gamma_broadcast = gamma_arr.reshape(1,-1,1,1)
        beta_broadcast = beta_arr.reshape(1,-1,1,1)
    else:
        raise ValueError(f"Unsupported inut dimensionality: {x_arr.ndim}")

    mean = np.mean(x_arr, axis=axis, keepdims= True)
    var = np.var(x_arr, axis=axis, keepdims = True)
    x_hat = (x_arr - mean) / np.sqrt(var+eps)
    out = gamma_broadcast * x_hat + beta_broadcast
    return out
    pass