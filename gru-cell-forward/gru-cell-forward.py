import numpy as np

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    # Write code here
    x = np.asarray(x,dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1,-1)
    if h_prev.ndim == 1:
        h_prev = h_prev.reshape(1,-1)
    Wz, Ur, Wh = params['Wz'], params['Ur'], params['Wh']
    Wz = params['Wz']
    Wr = params['Wr']
    Wh = params['Wh']
    Uz = params['Uz']
    Ur = params['Ur']
    Uh = params['Uh']
    bz = params['bz']
    br = params['br']
    bh = params['bh']

    def sigmoid(arr):
        return np.where(arr >= 0, 1/(1 + np.exp(-arr)), np.exp(arr) / (1+np.exp(arr)))
    z_t = sigmoid(np.dot(x,Wz) + np.dot(h_prev, Uz) + bz)
    r_t = sigmoid(np.dot(x,Wr) + np.dot(h_prev, Ur) + br)
    h_tilde = np.tanh(np.dot(x, Wh) + np.dot(r_t * h_prev, Uh) + bh)
    h_t = (1-z_t)* h_prev + z_t * h_tilde
    if is_1d:
        return h_t.squeeze(0)
    return h_t
    pass