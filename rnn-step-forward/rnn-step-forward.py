import numpy as np

def rnn_step_forward(x_t: list, h_prev: list, Wx: list, Wh: list, b: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (H,).
    """
    x = np.array(x_t)
    h = np.array(h_prev)
    Wx_arr = np.array(Wx)
    Wh_arr = np.array(Wh)
    b_arr = np.array(b)
    a_t = np.dot(x, Wx_arr) + np.dot(h, Wh_arr)+b_arr
    h_t = np.tanh(a_t)
    return h_t
    pass