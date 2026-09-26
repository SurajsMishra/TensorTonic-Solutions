from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    mean_val = float(np.mean(x))
    median_val = float(np.median(x))
    counts = Counter(x)
    max_freq = max(counts.values())
    modes = [val for val, freq in counts.items() if freq == max_freq]
    mode_val = float(min(modes))
    return{
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val
    }
    pass