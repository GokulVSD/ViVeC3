from numpy import array
from scipy.spatial.distance import cityblock, correlation, cosine, euclidean
from numpy import ndarray, array

def cosine_similarity(vector_a, vector_b):
    return max(1 - cosine(vector_a, vector_b), 0)


# Distance functions for latent semantics were chosen based on subjective analysis of the
# results.
DISTANCE_MAP = {
    # Manhattan distance (City-block distance): This distance can be imagined as the length needed
    # to move between two points in a grid where you can only move up, down, left or right. This
    # measure is sensitive to differences along each dimension of the feature vectors, capturing
    # the total absolute difference between corresponding moments, which makes it great for
    # assessing differences in color distribution in terms of absolute deviations. Since it
    # does not square the differences like Euclidean distance, it is less susceptible outliers
    # when many dimensions differ.
    "color": cityblock,
    "t7LS1_color": correlation,
    "t8LS1_color": cityblock,
    "t9LS1_color": cityblock,
    "t10LS1_color": euclidean,
    # Correlation similarity: This measures the linear relationship between HOG feature vectors,
    # and performs well with features that have a strong linear correlation or dependency in
    # their gradient orientation distributions, taking into account directionality.
    "hog": correlation,
    "t7LS2_hog": cosine,
    "t8LS2_hog": cosine,
    "t9LS2_hog": cosine,
    "t10LS2_hog": correlation,
    # Cosine distance: We use Cosine since the output of ResNet layers are
    # normalized, reducing the importance of magnitude. It is good at discerning
    # semantic or structural similarity rather than their absolute feature values.
    # It also works well with sparse feature spaces such as those found in CNNs.
    "avgpool": cosine,
    "t7LS3_avgpool": cosine,
    "t8LS3_avgpool": cosine,
    "t9LS3_avgpool": cosine,
    "t10LS3_avgpool": cosine,
    "layer3": cosine,
    "t7LS4_layer3": cosine,
    "t8LS4_layer3": cosine,
    "t9LS4_layer3": cosine,
    "t10LS4_layer3": cosine,
    "fc": cosine,
    "resnet": cosine,
    "t7LS1_resnet": euclidean,
    "t8LS1_resnet": euclidean,
    "t9LS1_resnet": cosine,
    "t10LS1_resnet": euclidean,
    "t7LS2_resnet": cityblock,
    "t8LS2_resnet": cityblock,
    "t9LS2_resnet": cosine,
    "t10LS2_resnet": cityblock,
    "t7LS3_resnet": correlation,
    "t8LS3_resnet": correlation,
    "t9LS3_resnet": cosine,
    "t10LS3_resnet": cosine,
    "t7LS4_resnet": cosine,
    "t8LS4_resnet": cosine,
    "t9LS4_resnet": cosine,
    "t10LS4_resnet": cosine,
    # Cosine similarity is used when we try to construct label_label or image_image similarity
    # matrices. It is good at discerning semantic or structural similarity rather
    # than their absolute feature values.
    "label_label_similarity": cosine_similarity,
    "image_image_similarity": cosine_similarity,
}


def get_distance_fn(key):
    """
    Retrieve the distance function chosen for the specific feature space.
    """
    # If we have not defined a distance function for a latent semantic, use the
    # distance function corresponding to the feature space.
    if key not in DISTANCE_MAP:
        return DISTANCE_MAP['_'.join(key.split('_')[1:])]
    return DISTANCE_MAP[key]


def top_k_distance_ranker(k, query_vector, feature_vectors, distance_fn):
    return all_distance_ranker(query_vector, feature_vectors, distance_fn)[:k]


def top_k_min_distance_ranker(k, query_vectors, feature_vectors, distance_fn):
    return min_distance_ranker(query_vectors, feature_vectors, distance_fn)[:k]


def top_k_unique_label_distance_ranker(k, query_vector, feature_vectors, distance_fn):
    """
    Same as top k distance ranker but only includes the distance to the closest entry
    in feature vectors for a particular label.
    """
    return all_unique_label_distance_ranker(query_vector, feature_vectors, distance_fn)[:k]


def all_unique_label_distance_ranker(query_vector, feature_vectors, distance_fn):
    """
    Same as all distance ranker but only includes the distance to the closest entry
    in feature vectors for a particular label.
    """
    distances = all_distance_ranker(query_vector, feature_vectors, distance_fn)

    unique_label_distances = []
    found_labels = set()

    for dist_tuple in distances:
        dist, img_id, label = dist_tuple

        if label not in found_labels:
            found_labels.add(label)
            unique_label_distances.append(dist_tuple)

    return unique_label_distances


def all_distance_ranker(query_vector, feature_vectors, distance_fn):
    """
    Computes distance of query vector to all vectors in feature_vectors,
    returns sorted by distance (distance, img_id, label)
    """
    distances = []

    for img_id, feature_tuple in feature_vectors.items():
        label, feature_vector = feature_tuple
        distances.append((distance_fn(query_vector, feature_vector), img_id, label))

    distances = sorted(distances)

    return distances


def min_distance_ranker(query_vectors, feature_vectors, distance_fn):
    """
    Computes distance of each vector in feature_vectors to each vector
    in query_vectors. Includes only the smallest distance to a query
    vector in the result (one query vector is selected for every
    feature_vector, which happens to have the min distance to it).
    Returns sorted by distance (distance, img_id, label)
    """
    distances = []

    for img_id, feature_tuple in feature_vectors.items():
        label, feature_vector = feature_tuple

        query_distances = []
        for query_vector in query_vectors:
            query_distances.append((distance_fn(query_vector, feature_vector), img_id, label))

        query_distances = sorted(query_distances)

        distances.append(query_distances[0])

    distances = sorted(distances)

    return distances


def feature_vectors_to_np_vectors(feature_vectors):
    if isinstance(feature_vectors, ndarray):
        return feature_vectors

    vectors = []

    for label, vector in feature_vectors.values():
        vectors.append(vector)

    return array(vectors)

def distance_matrix(feature_vectors, distance_fn):
    """
    Computes an N x N matrix of distance between every image in the given vector space
    """
    feature_vectors = feature_vectors_to_np_vectors(feature_vectors)
    dm = []
    for query_vector in feature_vectors:
        d = []
        for target_vector in feature_vectors:
            d.append(distance_fn(query_vector, target_vector))
        dm.append(d)

    return array(dm)


def make_candidates_hashable_and_update_images(vector, candidates, images):
    for i in range(len(candidates)):
        candidate = candidates[i]
        vec, img_id, dist = tuple(candidate[0]), candidate[1], candidate[2]
        images[img_id] = candidate[0]
        candidates[i] = (vec, img_id, euclidean(candidate[0], vector))

    return set(candidates)