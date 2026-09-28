import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    eigenvalues = np.linalg.eigvals(matrix)
    sorted_eigenvalues = np.sort(eigenvalues.real)
    return sorted_eigenvalues
    pass