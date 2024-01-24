from numpy import array, ndarray, abs
from sklearn.cluster import KMeans
from pandas import DataFrame
from utils.distance_utils import all_unique_label_distance_ranker, cosine_similarity


def rep_label_vectors_to_feature_vectors(rep_label_vectors):
    """
    This function takes in representative label vectors and outputs
    feature vectors, in order to facilitate usage with distance utils.
    Input: dict: label -> list of representative vectors
    Output: dict: Index -> (label, representative vector)

    IMPORTANT: Since we are creating multiple dictionary entries for one
    dictionary entry of rep_label_vectors, the Index is meaningless and
    should not be used as IMG_ID.
    """
    feature_vectors = {}
    index = 0

    for label, rep_vectors in rep_label_vectors.items():
        for rep_vector in rep_vectors:
            feature_vectors[index] = (label, rep_vector)
            index += 1

    return feature_vectors


def feature_vectors_to_np_vectors(feature_vectors):
    if isinstance(feature_vectors, ndarray):
        return feature_vectors

    vectors = []

    for label, vector in feature_vectors.values():
        vectors.append(vector)

    return array(vectors)


def get_latent_feature_vectors(feature_vectors, reducer):
    latent_vectors = reducer.reduce_features(feature_vectors)

    latent_feature_vectors = {}
    for i, feature_item in enumerate(feature_vectors.items()):
        img_id, feature_tuple = feature_item
        label, _ = feature_tuple
        latent_feature_vectors[img_id] = (label, latent_vectors[i])

    return latent_feature_vectors


def flatten_feature_vectors(feature_vectors):
    """
    Input Format - dictionary: {img_id: (label, vector), ... }
    Output Format - list: [[img_id, label, vector], ... ]
    """
    flattened_feature_vectors = []
    for img_id, feature_tuple in feature_vectors.items():
        label, vector = feature_tuple
        flattened_feature_vectors.append([img_id, label, vector])

    return flattened_feature_vectors


def get_representative_vectors_using_kmeans(vectors, K):
    """
    Clusters the provided vectors into K clusters, and returns the list of K cluster
    centers. if K is 1, returns just a vector.
    """
    cluster_centers = []
    kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto").fit(array(vectors))
    for cluster_center in kmeans.cluster_centers_:
        cluster_centers.append(cluster_center)

    if K == 1:
        return cluster_centers[0]

    return cluster_centers


def get_representative_vectors_for_labels(feature_vectors, all_labels, K):

    rep_label_vectors = {}

    df = DataFrame(flatten_feature_vectors(feature_vectors), columns=["img_id", "label", "vector"])

    for label in all_labels:
        label_df = df.loc[df['label'] == label]
        rep_label_vectors[label] = get_representative_vectors_using_kmeans(label_df["vector"].tolist(), K)

    return rep_label_vectors


def get_image_label_similarity_vector(vector, rep_label_vectors, all_labels):
    """
    Given an image vector, and the representative label vectors under the same vector
    space, constructs a similarity vector using the same order as all_labels.
    """
    rep_feature_vectors = rep_label_vectors_to_feature_vectors(rep_label_vectors)

    similarities = all_unique_label_distance_ranker(vector, rep_feature_vectors, cosine_similarity)

    label_similarity = {}
    for i in range(len(similarities)):
        similarity, dummy_img_id, label = similarities[i]
        label_similarity[label] = similarity

    image_label_similarity = []
    for label in all_labels:
        image_label_similarity.append(label_similarity[label])

    return array(image_label_similarity)


def get_image_image_similarity_vector(vector, feature_vectors):
    """
    Given an image vector, and the feature space, generate an image-image similarity
    vector.
    """
    image_label_similarity = []

    for img_id, feature_tuple in feature_vectors.items():
        label, feature_vector = feature_tuple
        image_label_similarity.append(cosine_similarity(vector, feature_vector))

    return array(image_label_similarity)


def get_vectors_for_labels(feature_vectors, all_labels):

    label_vectors = {}

    df = DataFrame(flatten_feature_vectors(feature_vectors), columns=["img_id", "label", "vector"])

    for label in all_labels:
        label_df = df.loc[df['label'] == label]
        label_vectors[label] = list(zip(label_df["img_id"].tolist(), label_df["vector"].tolist()))

    return label_vectors


def get_img_ids_and_vectors(img_vectors):
    img_ids = [img_vector[0] for img_vector in img_vectors]
    vectors = [img_vector[1] for img_vector in img_vectors]

    return img_ids, array(vectors)