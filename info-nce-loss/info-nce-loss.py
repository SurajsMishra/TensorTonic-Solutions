import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    z1 = np.asarray(Z1)
    z2 = np.asarray(Z2)
    S = np.dot(z1, z2.T) / temperature
    S_max = np.max(S, axis=1, keepdims=True)
    S_shifted = S - S_max
    pos_logits = np.diag(S_shifted)
    exp_sum = np.sum(np.exp(S_shifted), axis=1)
    loss = -(pos_logits - np.log(exp_sum))
    return float(np.mean(loss))
    pass