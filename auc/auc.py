import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    fpr_arr = np.array(fpr)
    tpr_arr = np.array(tpr)
    widths = np.diff(fpr_arr)
    heights = 0.5 * (tpr_arr[:-1] + tpr_arr[1:])
    return float(np.sum(widths * heights))
    pass