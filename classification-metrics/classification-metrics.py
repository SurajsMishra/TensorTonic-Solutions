import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = float(np.mean(y_true == y_pred))
    classes = np.unique(np.concatenate([y_true, y_pred]))
    def safe_div(num, den):
        return num/den if den > 0 else 0.0
    if average == "micro":
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for c in classes:
            total_tp += np.sum((y_true == c) & (y_pred == c))
            total_fp += np.sum((y_true != c) & (y_pred == c))
            total_fn += np.sum((y_true == c) & (y_pred != c))
        precision = safe_div(total_tp, total_tp+total_fp)
        recall = safe_div(total_tp, total_tp+total_fn)
        f1 = safe_div(2 * precision* recall, precision+recall)
    elif average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        precision = safe_div(tp, tp+fp)
        recall = safe_div(tp, tp+fn)
        f1 = safe_div(2* precision * recall, precision+recall)
    elif average in ("macro", "weighted"):
        precisions = []
        recalls = []
        f1s = []
        supports = []

        for c in classes:
            tp = np.sum((y_true ==c)&(y_pred==c))
            fp = np.sum((y_true!=c)&(y_pred==c))
            fn = np.sum((y_true==c)*(y_pred!=c))
            support = np.sum(y_true == c)
            p_c = safe_div(tp,tp+fp)
            r_c = safe_div(tp, tp+fn)
            f1_c = safe_div(2*p_c*r_c,p_c+r_c)
            precisions.append(p_c)
            recalls.append(r_c)
            f1s.append(f1_c)
            supports.append(support)
        if average == "macro":
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)
        else:
            total_support = np.sum(supports)
            if total_support>0:
                precision = np.average(precisions, weights=supports)
                recall = np.average(recalls, weights=supports)
                f1 = np.average(f1s, weights=supports)
            else:
                precision,recall,f1 = 0.0,0.0,0.0
    return {
        "accuracy": round(accuracy,6),
        "precision": round(precision,6),
        "recall": round(recall,6),
        "f1": round(f1,6)
    }
    pass