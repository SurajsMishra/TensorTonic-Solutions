import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    x = np.array(x, dtype=np.float64)
    conv1 = np.array(conv1, dtype=np.float64)
    W1_b1 = np.array(W1_b1, dtype=np.float64)
    W2_b1 = np.array(W2_b1, dtype=np.float64)
    W1_b2 = np.array(W1_b2, dtype=np.float64)
    W2_b2 = np.array(W2_b2, dtype=np.float64)
    Ws_b2 = np.array(Ws_b2, dtype=np.float64)
    fc = np.array(fc, dtype=np.float64)
    out = np.maximum(0, x@conv1)
    shortcut_b1 = out.copy()
    out = np.maximum(0, out@ W1_b1)
    out = out @ W2_b1
    out = np.maximum(0, out + shortcut_b1)
    shortcut_b2 = out @ Ws_b2
    out = np.maximum(0, out @ W1_b2)
    out = out @ W2_b2
    out = np.maximum(0, out+shortcut_b2)
    logits = out@fc
    return np.round(logits, 4).tolist()
    pass