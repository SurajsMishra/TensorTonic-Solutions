import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    w_arr = np.array(w)
    g_arr = np.array(g)
    s_arr = np.array(s)
    new_s = beta * s_arr + (1-beta) * (g_arr ** 2)
    new_w = w_arr - (lr/(np.sqrt(new_s) + eps)) * g_arr
    return new_w.tolist(), new_s.tolist()
    pass