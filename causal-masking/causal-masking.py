import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    arr = np.array(scores, dtype=float)
    T=arr.shape[-1]
    mask = np.triu(np.ones((T,T), dtype=bool), k=1)
    arr[..., mask] = mask_value
    return arr
    pass