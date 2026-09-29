import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x_arr = np.array(x, dtype=np.float64)
    n = len(x_arr)
    x_bar = np.mean(x_arr)
    s = np.std(x_arr, ddof=1)
    if s==0:
        if x_bar == mu0:
            return 0.0
        return float(np.copysign(np.inf, x_bar - mu0))
    se = s/np.sqrt(n)
    t_stat = (x_bar - mu0) / se
    return float(t_stat)
    pass