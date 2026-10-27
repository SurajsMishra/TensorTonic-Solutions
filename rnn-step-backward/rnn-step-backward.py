import numpy as np

def rnn_step_backward(dh: list, cache: list) -> dict:
    """
    Returns a dictionary with dx_t, dh_prev, dW, dU, and db.
    """
    x_t, h_prev, h_t, W, U, b = cache
    dh = np.array(dh, dtype=float)
    x_t = np.array(x_t, dtype=float)
    h_prev = np.array(h_prev, dtype=float)
    h_t = np.array(h_t, dtype=float)
    W = np.array(W, dtype=float)
    U = np.array(U, dtype=float)
    dz = dh * (1.0 - h_t ** 2)
    dx_t = W.T @ dz
    dh_prev = U.T @ dz
    dW = np.outer(dz,x_t)
    dU = np.outer(dz, h_prev)
    db = dz
    return {
        "dx_t": dx_t,
        "dh_prev": dh_prev,
        "dW": dW,
        "dU": dU,
        "db": db
    }
    pass