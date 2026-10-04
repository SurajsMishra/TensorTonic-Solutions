def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    assignments = []
    for p in points:
        min_dist = float('inf')
        best_centroid_idx = 0
        for idx, c in enumerate(centroids):
            dist = sum((p_d - c_d)**2 for p_d, c_d in zip(p,c))
            if dist<min_dist:
                min_dist = dist
                best_centroid_idx = idx

        assignments.append(best_centroid_idx)
    return assignments
    pass