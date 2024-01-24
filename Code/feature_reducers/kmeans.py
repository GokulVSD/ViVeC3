from numpy import array, random, argmin, sum, sqrt, mean
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from utils.vector_utils import feature_vectors_to_np_vectors


class KMeansReducer:
    """
    K-means reducer.
    """

    def __init__(self, feature_vectors, K):
        """
        Accept vector space and target dimensions, and preprocess.
        """
        self.iters = 300
        self.seed = 42

        self.K = K

        D = feature_vectors_to_np_vectors(feature_vectors)
        self.scaler = StandardScaler().fit(D)
        D = self.scaler.transform(D)

        num_objs = D.shape[0]

        random.seed(self.seed)

        # Choose K random centroids.
        centroids = []

        for k in range(K):
            centroids.append(D[random.choice(range(num_objs))])

        for iter in range(self.iters):
            clusters = self._bucket_to_clusters(D, centroids)
            previous_centroids = centroids
            centroids = self._compute_new_centroids(D, clusters)
            diff = centroids - previous_centroids
            if not diff.any():
                # Break early if centroids did not change between iterations.
                break

        self.centroids = centroids


    def _bucket_to_clusters(self, D, centroids):
        clusters = [[] for _ in range(self.K)]
        for obj_idx, obj in enumerate(D):
            closest_centroid_idx = argmin(
                sqrt(sum((obj-centroids)**2, axis=1))
            )
            clusters[closest_centroid_idx].append(obj_idx)
        return clusters


    def _compute_new_centroids(self, D, clusters):
        centroids = []
        for cluster_idxs in clusters:
            new_centroid = mean(D[cluster_idxs], axis=0)
            centroids.append(new_centroid)
        return array(centroids)


    def get_similarity_matrix(self, feature_vectors):
        """
        Get the similarity matrix. Some techniques do not expose the similarity matrix,
        for those, we just reduce the features and return.
        """
        return self.reduce_features(feature_vectors)


    def reduce_features(self, feature_vectors):
        """
        Reduce number of features in input vector space to K.
        Uses centroids generated in __init__, computes distance
        to each centroid, resulting in K features.
        """
        D = feature_vectors_to_np_vectors(feature_vectors)
        D = self.scaler.transform(D)

        res = []
        for obj in D:
            feature_vector = []
            for centroid in self.centroids:
                feature_vector.append(euclidean(obj, centroid))
            res.append(feature_vector)

        return array(res)

