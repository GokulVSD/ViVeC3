from numpy import zeros, array
from utils.dataset_utils import initialize_dataset
from tensorly.decomposition import parafac
from utils.vector_utils import feature_vectors_to_np_vectors


class CPDecomposition:
    """
    CP Descomposition, produces 3 mode factors:
    factor 1: image - weight
    factor 2: label - weight
    factor 3: feature - weight
    """

    def __init__(self, feature_vectors, K):
        self.K = K

        tensor = self._create_tensor(feature_vectors)

        decomposition, errors = parafac(
            tensor,
            rank=self.K,
            return_errors=True,
            init='random',
            tol=1e-4,
            n_iter_max=100,
            normalize_factors=True
        )

        weights, factors =  decomposition

        self.factors = factors


    def _create_tensor(self, feature_vectors):
        tensor = []

        all_labels = initialize_dataset().categories

        for feature_item in feature_vectors.items():
            img_id, feature_tuple = feature_item
            current_label, current_feature_vector = feature_tuple

            label_feature_distances = []

            for label in all_labels:
                if label == current_label:
                    label_feature_distances.append(current_feature_vector)
                else:
                    label_feature_distances.append(zeros(len(current_feature_vector)))

            tensor.append(label_feature_distances)

        return array(tensor)


    def get_image_weight(self):
        return self.factors[0]


    def get_label_weight(self):
        return self.factors[1]


    def get_feature_weight(self):
        return self.factors[2]


    def reduce_features(self, feature_vectors):
        """
        Reduce number of features in input vector space to K.
        This uses the decompositions found during the __init__ function.
        """

        D = feature_vectors_to_np_vectors(feature_vectors)

        # The intuition here is that CP decomposotion is a specialized
        # form of Tucker decomposition, which itself is a higher order
        # SVD. Therefore, using the same argument made in the reduce_features
        # function of SVDReducer, we can multiply D with the feature factor
        # to get an image latent semantic.

        return D @ self.factors[2]
