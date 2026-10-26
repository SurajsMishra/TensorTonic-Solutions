import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, b_h: np.ndarray) -> dict:
    """
    Returns hidden_states and final_hidden_state as float64 arrays.
    """
    N,T,D = X.shape
    H = h_0.shape[1]
    hidden_states = np.zeros((N,T,H), dtype=np.float64)
    h = h_0.astype(np.float64)
    for t in range(T):
        x_t = X[:, t, :]
        h=np.tanh(x_t @ W_xh.T + h @W_hh.T + b_h)
        hidden_states[:, t, :] = h
    return {
        "hidden_states": hidden_states,
        "final_hidden_state": h.astype(np.float64)
    }
    pass