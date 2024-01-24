import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.vector_utils import feature_vectors_to_np_vectors


class NNMFReducer:
    """
    Non-negative Matrix Factorization reducer.
    """

    def __init__(self, feature_vectors, K):
        """
        Accept vector space and target dimensions, and preprocess.
        """
        self.iters = 200
        self.seed = 42

        self.K = K

        D = feature_vectors_to_np_vectors(feature_vectors)
        self.scaler = MinMaxScaler().fit(D)
        D = self.scaler.transform(D)

        np.random.seed(self.seed)

        W = np.abs(np.random.randn(1, D.shape[0], self.K))[0]
        H = np.abs(np.random.randn(1, self.K, D.shape[1]))[0]

        for i in range(self.iters):
            H = H * ((W.T @ D) / (W.T @ W @ H) + 1e-10)
            W = W * ((D @ H.T) / (W @ H @ H.T) + 1e-10)

        self.W = W
        self.H = H


    def get_similarity_matrix(self, feature_vectors):
        """
        Get the similarity matrix. Some techniques do not expose the similarity matrix,
        for those, we just reduce the features and return.
        """
        W = self.W
        # Delete so that the database object isn't too big.
        del self.W
        return W


    def reduce_features(self, feature_vectors):
        """
        Reduce number of features in input vector space to K.
        This uses the trained NNMF model coefficients trained during __init__
        and finds approximate W.
        """

        D = feature_vectors_to_np_vectors(feature_vectors)
        D = self.scaler.transform(D)

        np.random.seed(self.seed)
        W = np.abs(np.random.randn(1, D.shape[0], self.K))[0]

        for i in range(self.iters):
            W = W * ((D @ self.H.T) / (W @ self.H @ self.H.T) + 1e-10)

        return W