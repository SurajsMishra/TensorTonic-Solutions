from collections import defaultdict 
def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    category_sums = defaultdict(float)
    category_counts = defaultdict(int)
    for cat, target in zip(categories, targets):
        category_sums[cat] += target
        category_counts[cat] += 1
    category_means = {
        cat: category_sums[cat]/category_counts[cat] for cat in category_sums
    }
    return [float(category_means[cat]) for cat in categories]
    pass