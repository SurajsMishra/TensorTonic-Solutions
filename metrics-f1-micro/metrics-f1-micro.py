def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    tp = sum(t==p for t,p in zip(y_true, y_pred))
    total = len(y_true)
    f1 = tp/total
    return round(float(f1), 4)
    pass