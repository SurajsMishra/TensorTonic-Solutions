import math

def perplexity(prob_distributions: list, actual_tokens: list) -> float:
    """
    Returns the sequence perplexity.
    """
    log_sum = 0.0
    N = len(prob_distributions)
    for i in range(N):
        prob = prob_distributions[i][actual_tokens[i]]
        log_sum += math.log(prob)
    cross_entropy = -log_sum/N
    return round(math.exp(cross_entropy), 4)
    pass