import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> dict:
    """
    Returns output_gate and hidden_state as float64 arrays.
    """
    h_prev = np.asarray(h_prev, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    C_t = np.asarray(C_t, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64)
    b_o = np.asarray(b_o, dtype=np.float64)
    concat_hx = np.concatenate([h_prev, x_t], axis=-1)
    output_gate_val = sigmoid(np.dot(concat_hx, W_o.T) + b_o)
    hidden_state_val = output_gate_val*np.tanh(C_t)
    return{
        "output_gate": output_gate_val.astype(np.float64),
        "hidden_state": hidden_state_val.astype(np.float64)
    }
    pass