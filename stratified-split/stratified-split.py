import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    """
    Returns a dictionary with X_train, X_test, y_train, and y_test.
    """
    X_arr = np.array(X)
    y_arr = np.array(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y_arr)
    train_indices = []
    test_indices = []
    for c in classes:
        class_indices = np.flatnonzero(y_arr == c)
        n_c = len(class_indices)
        n_c_test = int(round(n_c*test_size))
        if n_c>1:
            n_c_test = min(n_c_test, n_c-1)
        shuffled_indices = rng.permutation(class_indices)
        c_test_idx = shuffled_indices[:n_c_test]
        c_train_idx = shuffled_indices[n_c_test:]
        test_indices.extend(c_test_idx)
        train_indices.extend(c_train_idx)
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    return{
        "X_train": X_arr[train_indices],
        "X_test": X_arr[test_indices],
        "y_train": y_arr[train_indices],
        "y_test": y_arr[test_indices],
    }
    
    pass