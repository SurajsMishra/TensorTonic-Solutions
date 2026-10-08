import numpy as np

def silhouette_score(X: list, labels: list[int]) -> float:
    """
    Returns the mean Silhouette Score as a Python float.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    dists = np.linalg.norm(X[:, np.newaxis, :]- X[np.newaxis, :, :], axis=2)
    unique_labels = np.unique(labels)
    a = np.zeros(n_samples, dtype=np.float64)
    for k in unique_labels:
        mask = (labels == k)
        cluster_size = np.sum(mask)
        if cluster_size > 1:
            a[mask] = np.sum(dists[mask][:, mask], axis=1) / (cluster_size - 1)
        else:
            a[mask] = 0.0

    b = np.full(n_samples, np.inf, dtype=np.float64)
    for k in unique_labels:
        mask_other = (labels == k)
        mean_dists_to_k = np.mean(dists[:, mask_other], axis=1)
        mask_not_k = ~mask_other
        b[mask_not_k] = np.minimum(b[mask_not_k], mean_dists_to_k[mask_not_k])

    max_ab = np.maximum(a,b)
    s = np.zeros(n_samples, dtype=np.float64)
    valid_mask = max_ab>0
    s[valid_mask] = (b[valid_mask] - a[valid_mask]) / max_ab[valid_mask]
    return float(np.mean(s))
    pass