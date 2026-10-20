import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    N = len(docs)
    if N==0:
        return np.array([], dtype=float)
    doc_lengths = [len(doc) for doc in docs]
    avgdl = sum(doc_lengths)/N
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    unique_query_tokens = list(set(query_tokens))
    scores = np.zeros(N, dtype=float)
    for t in unique_query_tokens:
        df_t = df[t]
        idf = math.log((N- df_t +0.5)/ (df_t + 0.5) + 1.0)
        for i , doc in enumerate(docs):
            tf = doc.count(t)
            if tf>0:
                doc_len = doc_lengths[i]
                denom = tf + k1 * (1.0 - b + b *(doc_len/ avgdl))
                score_contrib = idf *(tf*(k1 + 1.0))/denom
                scores[i] += score_contrib
    return scores
    pass