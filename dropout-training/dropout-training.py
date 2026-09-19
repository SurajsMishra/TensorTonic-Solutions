import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x_arr = np.array(x, dtype = np.float64)
    if p>= 1.0:
        dropout_pattern = np.zeros_like(x_arr)
        output = np.zeros_like(x_arr)
        return output, dropout_pattern
    if rng is not None:
        rand_vals = rng.random(x_arr.shape)
    else:
        rand_vals = np.random.random(x_arr.shape)

    scale = 1.0 / (1.0 - p)
    dropout_pattern = (rand_vals >= p).astype(np.float64)* scale
    output = x_arr* dropout_pattern
    return output, dropout_pattern
    pass