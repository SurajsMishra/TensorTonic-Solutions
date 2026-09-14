import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts = True)
    probs = counts/len(y)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)
    pass