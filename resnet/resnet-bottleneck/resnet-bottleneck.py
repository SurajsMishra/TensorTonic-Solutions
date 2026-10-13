import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    W3 = np.array(W3, dtype=np.float64)
    if Ws is None:
        shortcut = x
    else:
        Ws = np.array(Ws, dtype=np.float64)
        shortcut = x@ Ws
    out1 = np.maximum(0, x@W1)
    out2 = np.maximum(0, out1 @ W2)
    out3 = out2 @ W3
    total = np.maximum(0, out3 + shortcut)
    return np.round(total, 4).tolist()
    pass