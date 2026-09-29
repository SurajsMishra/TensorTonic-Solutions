import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    x_arr = np.array(x)
    n = len(x_arr)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap,n))
    bootstrap_samples = x_arr[indices]
    bootstrap_mean = bootstrap_samples.mean(axis=1)
    overall_mean = float(np.mean(bootstrap_mean))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(bootstrap_mean, alpha))
    upper = float(np.quantile(bootstrap_mean, 1.0 - alpha))
    return {"bootstrap_mean": overall_mean, "lower": lower, "upper": upper}
    pass