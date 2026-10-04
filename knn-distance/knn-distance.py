import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    X_tr = np.array(X_train,dtype=float)
    X_te = np.array(X_test, dtype=float)
    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(-1,1)
    if X_te.ndim == 1:
        X_te = X_te.reshape(-1,1)
    n_train = X_tr.shape[0]
    diff = X_te[:,None,:] - X_tr[None,:,:]
    distance = np.linalg.norm(diff, axis=-1)
    sorted_indices = np.argsort(distance, axis=1)
    if k>n_train:
        pad_width = k-n_train
        return np.pad(sorted_indices, ((0,0), (0,pad_width)), mode='constant', constant_values=-1)
    else:
        return sorted_indices[:, :k]
    pass