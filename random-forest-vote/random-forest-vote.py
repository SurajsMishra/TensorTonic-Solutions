from collections import Counter
def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    """
    if not predictions or not predictions[0]:
        return []
    num_trees = len(predictions)
    num_samples = len(predictions[0])
    result = []
    for i in range(num_samples):
        sample_votes = [predictions[t][i] for t in range(num_trees)]
        counts = Counter(sample_votes)
        best_class = sorted(counts.keys(), key= lambda label: (-counts[label], label))[0]
        result.append(best_class)
    return result
    pass