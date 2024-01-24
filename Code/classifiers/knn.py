'''
1. We have the images and labels in a particular range
2. For a given test set we loop through each image and:
    1. For each image in the test set we take distances from all images with k and store it
    2. After sorting the array we find the k nearest neighbors from the new image
    3. Depending on which type of class the majority of neighbors belong to we assign that label to the new item
3. Generate metrics to calculate precision, recall and F1 score
'''

from scipy.spatial.distance import euclidean

class KNNClassifier:
    def __init__(self, K, train_vectors, dist_fn = euclidean):
        self.K = K
        self.dist_fn = dist_fn
        self.train_vectors = train_vectors

    def find_max_repeated_label(self, distance_arr):
        max_labels = {}
        for label, _ in distance_arr:
            max_labels[label] = max_labels.get(label, 0) + 1
        return max(max_labels, key=max_labels.get)


    def classify(self, test_vectors):
        output_labels = []
        if self.K > len(self.train_vectors):
            print("K value is greater than length of training vectors provided")
            return
        for i, feature_item in enumerate(test_vectors.items()):
            image_id, feature_tuple = feature_item
            test_label = feature_tuple[0]
            test_vector = feature_tuple[1]

            distance_arr = []
            for train_label, train_vector in self.train_vectors.values():
                distance = self.dist_fn(test_vector, train_vector)
                distance_arr.append((train_label, distance))
            distance_arr = sorted(distance_arr, key=lambda x: x[1])[:self.K]
            predicted_label = self.find_max_repeated_label(distance_arr)
            output_labels.append({"test_label": test_label, "predicted_label": predicted_label})
            print(f"IMG_ID: {image_id}\tTrue label: {test_label}\tPred label: {predicted_label}")
        return output_labels
