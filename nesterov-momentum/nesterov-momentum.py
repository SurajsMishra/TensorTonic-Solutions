import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    w_arr = np.array(w, dtype=np.float64)
    v_arr = np.array(v ,dtype=np.float64)
    grad_arr = np.array(grad, dtype=np.float64)
    new_v = momentum * v_arr + lr*grad_arr
    new_w = w_arr - new_v
    return{
        "new_w": new_w,
        "new_v": new_v
    }
    pass