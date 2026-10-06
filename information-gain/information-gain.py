import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)
    def entropy(labels):
        if len(labels) == 0:
            return 0.0
        _,counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return float(-np.sum(probs*np.log2(probs)))

    y_L = y[split_mask]
    y_R = y[~split_mask]
    if len(y_L) == 0 or len(y_R) == 0:
        return 0.0
    n_total = len(y)
    h_parent = entropy(y)
    h_left = entropy(y_L)
    h_right = entropy(y_R)
    ig = h_parent - (len(y_L)/ n_total) * h_left - (len(y_R) / n_total) * h_right
    return float(ig)
    pass