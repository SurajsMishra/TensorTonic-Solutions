import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    mask =p>0
    p_pos = p[mask]
    q_pos = q[mask]
    q_pos = np.clip(q_pos, eps, None)
    divergence = np.sum(p_pos* np.log(p_pos/q_pos))
    return float(divergence)
    pass