import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    out1 = np.maximum(0, x@W1.T)
    out2 = out1 @W2.T
    res = np.maximum(0, out2+x)
    return np.round(res,4).tolist()
    pass