import numpy as np

def roc_curve(y_true: list, y_score: list) -> dict:
    """
    Returns a dictionary with fpr, tpr, and thresholds.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    desc_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_indices]
    y_true = y_true[desc_indices]
    distinct_value_indices = np.where(y_score[:-1] != y_score[1:])[0]
    threshold_indices = np.r_[distinct_value_indices, y_true.size -1]
    tps = np.cumsum(y_true)[threshold_indices]
    fps = np.cumsum(1-y_true)[threshold_indices]
    total_positives = tps[-1]
    total_negatives = fps[-1]
    tpr = tps/total_positives
    fpr = fps/total_negatives
    thresholds = y_score[threshold_indices]
    tpr = np.r_[0.0,tpr]
    fpr = np.r_[0.0,fpr]
    thresholds = np.r_[np.inf, thresholds]
    return{
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }
    pass