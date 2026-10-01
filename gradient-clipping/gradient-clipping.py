import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    g_arr = np.array(g, dtype=float)
    norm = np.linalg.norm(g_arr)
    if norm>max_norm:
        g_arr = g_arr*(max_norm/norm)
    return g_arr
    pass