from numpy import argsort, diag, sqrt, identity, ones
from numpy.linalg import eigh
from utils.vector_utils import feature_vectors_to_np_vectors
from sklearn.metrics import pairwise_distances


class MDSReducer:
    """
    Classical Multidimensional Scaling.
    """

    def __init__(self, feature_vectors, K):
        """
        Accept vector space and target dimensions, and preprocess.
        """
        self.K = K

        X = feature_vectors_to_np_vectors(feature_vectors)
        D = pairwise_distances(X)

        n = D.shape[0]

        C = identity(n) - (1/n) * ones((n, n))

        B = -0.5 * C @ (D**2) @ C

        w, v = eigh(B)
        idx   = argsort(w)[::-1]
        eigvals = w[idx]
        eigvecs = v[:,idx]

        Lambda  = diag(sqrt(eigvals[:K]))
        V  = eigvecs[:,:K]

        X  = Lambda @ V.T
        X = X.T

        self.X = X



    def get_similarity_matrix(self, feature_vectors):
        """
        Get the similarity matrix. Some techniques do not expose the similarity matrix,
        for those, we just reduce the features and return.
        """
        return self.X


    def reduce_features(self, feature_vectors):
        """
        Reduce number of features in input vector space to K.
        """

        return self.X