from classifiers.dbscan import DBSCANClassifer

def dbscan_optimal_hyper_parameters(vectors, C, dist_fn):
    # Find all distances between points.
    distances = []
    for i, vec1 in enumerate(vectors):
        for j, vec2 in enumerate(vectors):
            if i != j:
                distances.append(dist_fn(vec1, vec2))

    distances = sorted(distances)

    # Initially choose MIN_PTS to be sqrt of max possible points per cluster
    MIN_PTS = max(2, int((len(vectors) / C) ** 0.5))
    EPS = distances[0]

    best_diff = 10e9
    best_eps = EPS
    best_min_pts = MIN_PTS
    best_num_clusters = 0

    while MIN_PTS > 1:

        EPS = distances[0]

        while EPS <= distances[int(len(distances)/2) + 1]:

            DBSCAN = DBSCANClassifer(eps=EPS, minPts=MIN_PTS, dist_fn=dist_fn)

            c_labels = DBSCAN.classify(vectors)

            num_clusters = set(c_labels)
            num_clusters.discard(-1)
            num_clusters = len(num_clusters)

            if abs(C - num_clusters) < best_diff:
                best_diff = abs(C - num_clusters)
                best_eps = EPS
                best_min_pts = MIN_PTS
                best_num_clusters = num_clusters
                if best_diff == 0:
                    break

            EPS = EPS * 1.05

        if best_diff == 0:
            break

        MIN_PTS -= 1

    EPS = best_eps
    MIN_PTS = best_min_pts

    return EPS, MIN_PTS, best_num_clusters