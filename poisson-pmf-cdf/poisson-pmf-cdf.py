import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    current_prob = math.exp(-lam)
    cdf = current_prob
    pmf = current_prob
    if k==0:
        return {"pmf":float(pmf), "cdf":float(cdf)}
    for i in range(1,k+1):
        current_prob *= lam/i
        cdf += current_prob
        if i==k:
            pmf=current_prob

    return{"pmf":float(pmf), "cdf":float(cdf)}
    pass