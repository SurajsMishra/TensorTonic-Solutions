import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X_arr= np.array(X, dtype=float)
    n,d=X_arr.shape
    X_centred = X_arr - np.mean(X_arr, axis=0)
    cov_matrix = np.dot(X_centred.T, X_centred)/(n-1)
    eigenvalues, eigenvectors=np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    top_k_eigenvectors = eigenvectors[:, idx[:k]]
    X_proj = np.dot(X_centred, top_k_eigenvectors)
    return X_proj.tolist()
    pass