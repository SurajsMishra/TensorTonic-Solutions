import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through residual Jacobians.
    """
    result = np.asarray(x, dtype=np.float64).copy()
    for jacobian in gradients_F:
        J = np.asarray(jacobian, dtype=np.float64)
        result = result + result @ J
    return result
    pass

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through plain Jacobians.
    """
    result = np.asarray(x, dtype=np.float64)
    for jacobian in gradients_F:
        J = np.asarray(jacobian, dtype=np.float64)
        result = result @ J
    return result
    pass