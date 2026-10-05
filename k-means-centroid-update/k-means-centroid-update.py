def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    if not points:
        return []
    num_features = len(points[0])
    centroid_sums = [[0.0]* num_features for i in range(k)]
    cluster_counts = [0]*k
    for point, cluster_id in zip(points, assignments):
        cluster_counts[cluster_id] += 1
        for dim in range(num_features):
            centroid_sums[cluster_id][dim] += point[dim]
    centroids = []
    for cluster_id in range(k):
        count = cluster_counts[cluster_id]
        if count == 0:
            centroids.append([0.0]*num_features)
        else:
            centroids.append([val/count for val in centroid_sums[cluster_id]])
    return centroids
    pass