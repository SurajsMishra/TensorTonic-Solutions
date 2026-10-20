import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    tokenized_docs = [doc.lower().split() for doc in documents]
    vocab_set  = set()
    for tokens in tokenized_docs:
        vocab_set.update(tokens)
    vocabulary = sorted(list(vocab_set))
    term_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
    N = len(documents)
    V = len(vocabulary)
    df = Counter()
    for tokens in tokenized_docs:
        df.update(set(tokens))
    tfidf_matrix = np.zeros((N,V), dtype=float)
    for i, tokens in enumerate(tokenized_docs):
        if not tokens:
            continue
        doc_len = len(tokens)
        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            j = term_to_idx[term]
            tf = count/doc_len
            idf = math.log(N / df[term])
            tfidf_matrix[i,j] = tf * idf
    return{
        "tfidf_matrix": tfidf_matrix,
        "vocabulary": vocabulary
    }
    pass