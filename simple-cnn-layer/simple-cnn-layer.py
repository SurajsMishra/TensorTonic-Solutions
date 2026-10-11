import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    x = np.array(x, dtype=float)
    W = np.array(W, dtype=float)
    b = np.array(b, dtype=float)
    N,C_in, H,W_in = x.shape
    C_out, _, K_H, K_W = W.shape
    H_out = H - K_H + 1
    W_out = W_in - K_W + 1
    out = np.zeros((N, C_out, H_out, W_out), dtype=float)
    for n in range(N):
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[n, :, i:i + K_H, j:j + K_W]
                    out[n,c,i,j] = np.sum(patch * W[c]) + b[c]
    return out
    pass