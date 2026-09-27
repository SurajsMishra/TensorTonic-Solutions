import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    def pmf_i(i):
        return math.comb(n,i) * (p**i)*((1-p)**(n-i))

    pmf_val = float(pmf_i(k))
    cdf_val = float(sum(pmf_i(i) for i in range(k+1)))
    return{
        "pmf": pmf_val,
        "cdf": cdf_val
    }
    pass