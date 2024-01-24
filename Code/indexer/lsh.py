import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean

class LSH:
    """
    Implementation of LSH.
    """

    def __init__(self, L, h, W, vectors):
        """
        L - number of layers.
        h - number of hashes per layer.
        W - number of partitions per hash (larger W, more false positive, smaller W, more miss).
        vectors - The vectors based on which hash ranges are determined.
        """
        self.L = L
        self.h = h
        self.W = W
        self.d = len(vectors[0])

        self.mins = []
        self.maxs = []

        self.layers = [self._create_layer(vectors) for i in range(self.L)]

        self.tables = [defaultdict(LSH._empty_list) for i in range(self.L)]


    def _empty_list():
        return []


    def _second_val(x):
        return x[2]


    def _create_layer(self, vectors):
        layer = np.random.randn(self.h, self.d)

        mins = [10e9] * self.h
        maxs = [-10e9] * self.h

        for vec in vectors:
            projections = layer @ vec
            for i, hash_val in enumerate(projections):
                mins[i] = min(hash_val, mins[i])
                maxs[i] = max(hash_val, maxs[i])

        self.mins.append(mins)
        self.maxs.append(maxs)

        return layer


    def _hash_mapper(self, val, minv, maxv):
        bucket_width = (maxv - minv) / self.W
        if val < minv:
            return 0
        return int(min(((val - minv) // bucket_width), self.W - 1))


    def generate_binary_strings(bit_count):
        binary_strings = []
        def genbin(n, bs=''):
            if len(bs) == n:
                binary_strings.append(bs)
            else:
                genbin(n, bs + '0')
                genbin(n, bs + '1')


        genbin(bit_count)
        return binary_strings


    def index(self, vector, aux_data=None):
        """
        Index a single vector into the hash tables.
        """
        vector = np.array(vector)
        val = (tuple(vector), aux_data)

        for i, table in enumerate(self.tables):

            projections = self.layers[i] @ vector
            hash = []
            for j, hash_val in enumerate(projections):
                hash.append(self._hash_mapper(hash_val, minv=self.mins[i][j], maxv=self.maxs[i][j]))

            table[tuple(hash)].append(val)


    def query(self, vector, limit):
        """
        Query for limit points closest to vector.
        Return ([(vector, aux_data, dist),...], unique_considered, total_considered)
        """
        total_considered = 0

        vector = np.array(vector)

        candidates = set()

        hashes = []

        for i, table in enumerate(self.tables):
            projections = self.layers[i] @ vector

            hash = []
            for j, hash_val in enumerate(projections):
                hash.append(self._hash_mapper(hash_val, minv=self.mins[i][j], maxv=self.maxs[i][j]))

            res = table[tuple(hash)]

            total_considered += len(res)

            candidates.update(res)

            hashes.append(hash)


        # If we didn't find enough candidates, we check adjacent buckets as well.
        delta = 0
        while len(candidates) < limit and delta < self.W:
            delta += 1
            for i, table in enumerate(self.tables):

                if len(candidates) >= limit:
                    break

                bin_strs = LSH.generate_binary_strings(len(hashes[i]))

                for bin_str in bin_strs:

                    larger_hash = []
                    smaller_hash = []

                    for j, val in enumerate(hashes[i]):
                        if bin_str[j] == '1':
                            larger_hash.append(int(min(val + delta, self.W - 1)))
                            smaller_hash.append(int(max(val - delta, 0)))
                        else:
                            larger_hash.append(val)
                            smaller_hash.append(val)

                    smaller_res = table[tuple(smaller_hash)]
                    larger_res = table[tuple(larger_hash)]

                    total_considered += len(smaller_res)
                    total_considered += len(larger_res)

                    candidates.update(smaller_res)
                    candidates.update(larger_res)

        ranked = []

        for vec in candidates:
            np_vec = np.array(vec[0])
            ranked.append((np_vec, vec[1], euclidean(vector, np_vec)))

        ranked = sorted(ranked, key=LSH._second_val)[:limit]

        return (ranked, len(candidates), total_considered)