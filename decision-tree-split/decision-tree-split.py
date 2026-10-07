import numpy as np
def gini_impurity(y):
    """
    Returns the best feature index and threshold.
    """
    if len(y) == 0:
        return 0.0
    _,counts = np.unique(y, return_counts=True)
    probs = counts/len(y)
    return 1.0 - np.sum(probs**2)

def decision_tree_split(X: list, y: list)-> list:
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape
    parent_gini = gini_impurity(y)
    best_gain = -1.0
    best_feature = None
    best_threshold = None
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_vals = np.sort(np.unique(feature_values))
        if len(unique_vals)<2:
            continue
        thresholds = (unique_vals[:-1]+unique_vals[1:])/2.0
        for threshold in thresholds:
            left_mask = feature_values <=threshold
            right_mask = ~left_mask
            y_left = y[left_mask]
            y_right = y[right_mask]
            n_left, n_right = len(y_left), len(y_right)
            if n_left == 0 or n_right == 0:
                continue
            split_gini = (n_left / n_samples)*gini_impurity(y_left)+(n_right/n_samples)* gini_impurity(y_right)
            info_gain = parent_gini - split_gini
            if info_gain > best_gain + 1e-12:
                best_gain = info_gain
                best_feature = feature_idx
                best_threshold = threshold

    return [int(best_feature), float(best_threshold)]
    pass