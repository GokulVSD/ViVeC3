from numpy import argsort, diag, sign, sort, sqrt, zeros, real
from numpy.linalg import eig
from utils.vector_utils import feature_vectors_to_np_vectors


class SVDReducer:
    """
    Singular-value decomposition.
    https://en.wikipedia.org/wiki/Singular_value_decomposition
    """

    def __init__(self, feature_vectors, K):
        """
        Accept vector space and target dimensions, and preprocess.
        This involves computing decomposition as D = U.S.VT.
        """
        self.K = K

        D = feature_vectors_to_np_vectors(feature_vectors)

        DT_D = D.T @ D
        D_DT = D @ D.T

        Lambda_DT_D, V = eig(DT_D)
        Lambda_D_DT, U = eig(D_DT)

        # Eigen values and vectors returned from np.linalg.eig aren't sorted. We want
        # to ensure largest Eigen values come first.
        i_1 = argsort(Lambda_DT_D)[::-1]
        i_2 = argsort(Lambda_D_DT)[::-1]

        V = V[:, i_1]
        self.U = U[:, i_2]

        Lambda = sqrt(sort(Lambda_DT_D)[::-1])

        small_dim = min(D.shape)

        # Lambda is a 1d vector representing the diagonal of a matrix. We need to
        # reconstruct the matrix in-order to perform matrix multiplication.
        S = zeros((D.shape[0], D.shape[1]))
        S[:small_dim, :small_dim] = diag(Lambda[:small_dim])


        # D @ V and U @ S should be equal, however, since we are separately finding
        # Eigen vectors for DT_D and D_DT, they can be out of sync, since the negative
        # of an Eigen vector is also an Eigen vector. We check to see if signs differ,
        # and then updates V's sign to compensate.
        same_sign = sign((D @ V)[0] * (U @ S)[0])
        self.V = V * same_sign.reshape(1, -1)

        # To get back the dataset:
        # D_latent = U[:,:self.K] @ S[0:self.K,:self.K] @ self.V.T[:self.K,:]

        self.V = self.V[:,:self.K]
        self.S = S


    def get_similarity_matrix(self, feature_vectors):
        """
        Get the similarity matrix. Some techniques do not expose the similarity matrix,
        for those, we just reduce the features and return.
        """
        U = self.U
        # Delete so that the database object isn't too big.
        del self.U
        return real(U[:,:self.K])


    def reduce_features(self, feature_vectors):
        """
        Reduce number of features in input vector space to K (truncated SVD).
        This uses the decompositions found during the __init__ function.
        """

        D = feature_vectors_to_np_vectors(feature_vectors)

        # http://infolab.stanford.edu/~ullman/mmds/ch11.pdf 11.3.5
        # Descriptive subset in latent space = D @ V.

        return real(D @ self.V)