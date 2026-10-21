import math
from collections import Counter

def bleu_score(candidate: list, reference: list, max_n: int) -> float:
    """
    Returns the unsmoothed BLEU score.
    """
    # Write code here
    if not candidate:
        return 0.0
    c = len(candidate)
    r = len(reference)
    log_precision_sum = 0.0
    for n in range(1, max_n + 1):
        cand_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate)- n+1)]
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n+1)]
        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)
        clipped_sum = 0
        total_cand = 0
        for ngram, count in cand_counts.items():
            clipped_sum += min(count, ref_counts[ngram])
            total_cand += count
        if total_cand == 0 or clipped_sum == 0:
            return 0.0
        p_n = clipped_sum / total_cand
        log_precision_sum += math.log(p_n)
    if c>= r:
        bp = 1.0
    else:
        bp = math.exp(1.0 - r/c)
    bleu = bp*math.exp(log_precision_sum/max_n)
    return float(bleu)
    pass