import random


class DBSCANClassifer:
    """
    DBSCAN classifer.
    """

    def __init__(self, eps, minPts, dist_fn):
        self.eps = eps
        self.minPts = minPts
        self.dist_fn = dist_fn
        self.seed = 42

    def classify(self, vectors):
        random.seed(self.seed)

        C = 0
        current_stack = set()
        unvisited = list(range(len(vectors)))
        clusters = []
        self.label_core_idxs = {}

        while (len(unvisited) != 0):

            first_point = True
            current_stack.add(random.choice(unvisited))

            while len(current_stack) != 0:

                cur_idx = current_stack.pop()

                neighour_idxs, is_core, is_border, is_noise = self._check_core(vectors, cur_idx)

                if (is_border & first_point):
                    clusters.append((cur_idx, -1))
                    clusters.extend(list(zip(neighour_idxs, [-1 for _ in range(len(neighour_idxs))])))
                    unvisited.remove(cur_idx)
                    unvisited = [e for e in unvisited if e not in neighour_idxs]
                    continue

                unvisited.remove(cur_idx)

                neighour_idxs = set(neighour_idxs) & set(unvisited)

                if is_core:

                    if C not in self.label_core_idxs:
                        self.label_core_idxs[C] = []
                    self.label_core_idxs[C].append(cur_idx)

                    first_point = False
                    clusters.append((cur_idx, C))
                    current_stack.update(neighour_idxs)

                elif is_border:
                    clusters.append((cur_idx, C))
                    continue

                elif is_noise:
                    clusters.append((cur_idx, -1))
                    continue

            if not first_point:
                C+=1

        labels = [0] * len(vectors)

        for tuple in clusters:
            idx, label = tuple
            labels[idx] = label

        return labels


    def get_cores(self, vectors):
        """
        Returns all core vectors for each cluster label
        """
        cores = {}
        for label, core_idxs in self.label_core_idxs.items():
            cores[label] = []
            for idx in core_idxs:
                cores[label].append(vectors[idx])

        return cores


    def _check_core(self, vectors, idx):
        """
        Returns (neighbor_idxs, is_core, is_border, is_noise)
        """

        vec = vectors[idx]
        within_radius = []

        for i in range(len(vectors)):
            cur_vec = vectors[i]

            if self.dist_fn(vec, cur_vec) <= self.eps:
                within_radius.append(i)

        if len(within_radius) >= self.minPts:
            return (within_radius, True, False, False)

        elif (len(within_radius) < self.minPts) and len(within_radius) > 0:
            return (within_radius, False, True, False)

        elif len(within_radius) == 0:
            return (within_radius, False, False, True)