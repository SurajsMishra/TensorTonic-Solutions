import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    def node_gini(y):
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y,return_counts = True)
        probs = counts/n
        return 1.0 - np.sum(probs**2)

    n_left = len(y_left)
    n_right = len(y_right)
    total_samples = n_left + n_right
    if total_samples == 0:
        return 0.0
    gini_left = node_gini(y_left)
    gini_right = node_gini(y_right)
    weighted_gini = (n_left / total_samples)* gini_left+(n_right / total_samples)* gini_right
    return float(weighted_gini)
    pass