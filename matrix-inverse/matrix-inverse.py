import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    aug = np.array(A, dtype=np.float64)
    n = aug.shape[0]
    if aug.ndim != 2 or aug.shape[0] != aug.shape[1]:
        return None
    aug = np.hstack([aug, np.eye(n, dtype=np.float64)])
    for col in range(n):
        max_row_idx = col+np.argmax(np.abs(aug[col:, col]))
        pivot_val = aug[max_row_idx, col]
        if np.abs(pivot_val)<1e-2:
            return None
        if max_row_idx != col:
            aug[[col, max_row_idx]] = aug[[max_row_idx, col]]
        aug[col] /= aug[col,col]
        for row in range(n):
            if row != col:
                factor = aug[row,col]
                aug[row] -= factor*aug[col]

    return aug[: , n:]
    pass