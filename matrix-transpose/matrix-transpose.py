import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    N = len(A)
    M = len(A[0])
    AT = np.zeros((M,N), dtype = type(A[0][0]))
    for i in range(N):
        for j in range(M):
            AT[j][i] = A[i][j]
    return AT
