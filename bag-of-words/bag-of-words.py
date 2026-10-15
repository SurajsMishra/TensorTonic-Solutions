import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    vocab_map = {word: idx for idx, word in enumerate(vocab)}
    result = np.zeros(len(vocab), dtype = int)
    for token in tokens:
        if token in vocab_map:
            result[vocab_map[token]] += 1
    return result
    pass