import numpy as np

def bigram_probabilities(tokens: list) -> dict:
    """
    Returns a dictionary with vocab, counts, and probabilities.
    """
    vocab = sorted(list(set(tokens)))
    V = len(vocab)
    token_to_idx = {token : i for i, token in enumerate(vocab)}
    counts = np.zeros((V,V), dtype=int)
    for i in range(len(tokens) - 1):
        u = token_to_idx[tokens[i]]
        v = token_to_idx[tokens[i+1]]
        counts[u,v] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probabilities = (counts + 1)/ (row_sums + V)
    return{
        "vocab":vocab,
        "counts":counts,
        "probabilities": probabilities
    }    
    pass