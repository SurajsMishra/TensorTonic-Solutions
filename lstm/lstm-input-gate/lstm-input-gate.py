import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> dict:
    """
    Returns input_gate and candidate_state as float64 arrays.
    """
    combined =np.concatenate([h_prev, x_t], axis =-1)
    i_raw = np.dot(W_i, combined.T).T + b_i if combined.ndim == 2 else W_i @ combined + b_i
    gate = sigmoid(i_raw)
    c_raw = np.dot(W_c, combined.T).T + b_c if combined.ndim == 2 else W_c @ combined + b_c
    candidate = np.tanh(c_raw)
    return{
        "input_gate":gate.astype(np.float64),
        "candidate_state": candidate.astype(np.float64)
    }
    pass