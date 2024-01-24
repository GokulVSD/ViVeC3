'''
1. We have the images and labels in a particular range
2. For a given test set we loop through each image and:
    1. For each image in the test set we calculate the PPR with each cluster representative
    2. After sorting the array we find the nearest cluster representative from the new image
3. Generate metrics to calculate precision, recall and F1 score
'''
from utils.distance_utils import get_distance_fn, distance_matrix
from scipy.spatial.distance import euclidean
from utils.database_utils import exists, compressed_retrieve, compressed_store, retrieve
import numpy as np
import sys
distance_fn = euclidean
np.set_printoptions(threshold=sys.maxsize)
FEATURE_SPACE_NAME = "fc"
class PPNClassifier:
    def __init__(self, N, M, train_vectors, feature_space="resnet", damping_factor=0.4):
        print("Computing distance matrix")
        self.train_vectors = train_vectors
        pt_filename = feature_space + "_" + "page_rank_classification.pt"
        if exists(pt_filename):
            self.distance_matrix = compressed_retrieve(pt_filename)
        else:
            self.distance_matrix = distance_matrix(train_vectors, distance_fn)
            compressed_store(self.distance_matrix, pt_filename)
        self.distance_matrix = np.array(self.distance_matrix)
        self.N = N
        self.M = M
        self.damping_factor = damping_factor
        print("Distance Matrix computed")

    def find_max_repeated_label(self, distance_arr):
        max_labels = {}
        for label, _ in distance_arr:
            max_labels[label] = max_labels.get(label, 0) + 1
        return max(max_labels, key=max_labels.get)

    def classify(self, test_vectors):
        output_labels = []
        for i, feature_item in enumerate(test_vectors.items()):
            image_id, feature_tuple = feature_item
            test_label = feature_tuple[0]
            test_value = feature_tuple[1]

            new_distance_matrix = np.pad(self.distance_matrix, ((0,1),(0,1)), mode='constant', constant_values=0)
            index = 0
            for train_label, train_value in self.train_vectors.values():
                distance = distance_fn(test_value, train_value)
                new_distance_matrix[-1][index] = distance
                new_distance_matrix[index][-1] = distance
                index += 1
            similarity_graph = np.zeros((len(new_distance_matrix), len(new_distance_matrix)), dtype=float)
            for i, row in enumerate(new_distance_matrix):
                similarity_graph[i][np.argsort(row)[:self.N]] = 1

            transition_matrix = (similarity_graph / self.N).T

            # Set the damping factor for the PageRank algorithm
            # Changed to make Damping Factor Configurable
            # damping_factor = 0.4

            # Initialize the PageRank vector with equal probabilities for all nodes
            pr = np.full(len(similarity_graph), 1/len(similarity_graph), dtype=float)

            s = np.zeros(len(similarity_graph), dtype=float)

            s[-1] = 1

            iterations = 20

            # Perform PageRank iterations to update node probabilities
            for i in range(iterations):
                # Update PageRank values using the PageRank formula with the transition matrix and 's'.
                pr = ((self.damping_factor * transition_matrix) @ pr) + ((1 - self.damping_factor) * s)
            feature_vectors = retrieve(f'{FEATURE_SPACE_NAME}.pt')

            dist_arr = []
            for idx in np.argsort(pr)[:-self.M-1:-1]:
                # the last element does not have a label so we want to skip that element
                if idx == 4339: continue
                imgid = idx * 2
                label = feature_vectors[imgid][0]
                pr_score = pr[idx]
                dist_arr.append((label, pr_score))
            predicted_label = self.find_max_repeated_label(dist_arr)
            output_labels.append({"test_label":test_label, "predicted_label": predicted_label, "score": dist_arr, "img_id": image_id})
            print(f"IMG_ID: {image_id}\tTrue label: {test_label}\tPred labels: {predicted_label}")
        return output_labels
