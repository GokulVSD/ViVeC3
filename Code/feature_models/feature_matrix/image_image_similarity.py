import numpy as np
from utils.distance_utils import get_distance_fn

FEATURE_SPACE = "image_image_similarity"
class ImageImageSimilarity:
    def __init__(self, feature_vectors):
        self.feature_vectors = feature_vectors

    def get_matrix(self):
        result = {}
        image_id_list = self.feature_vectors.keys()
        for i, item in enumerate(self.feature_vectors.items()):
            ref_image_id, (ref_image_label, ref_feature_vec) = item
            arr = np.zeros(shape=(len(image_id_list)))
            for idx, image_id in enumerate(image_id_list):
                feature_vec = self.feature_vectors[image_id][1]
                distance_fn = get_distance_fn(FEATURE_SPACE)
                arr[idx] = distance_fn(ref_feature_vec, feature_vec)
            result[ref_image_id] = (ref_image_label, arr)
        return result
