from utils.database_utils import retrieve
import numpy as np
import math
RELEVANCE_MAP = {
    'very_relevant': 'R+',
    'relevant': 'R',
    'irrelevant': 'I',
    'very_irrelevant': 'I-'
}
from scipy.spatial.distance import euclidean

class ProbabilisticReranker:
    def __init__(self, FEATURE_SPACE, distance_fn = euclidean):
        self.feature_vectors = retrieve(f'{FEATURE_SPACE}.pt')
        self.vector_length = len(self.feature_vectors[0][1])
        self.probability_map = {
            'very_relevant': {'total': 1, "vector": np.zeros((self.vector_length))},
            'relevant': {'total': 1, "vector": np.zeros((self.vector_length))},
            'irrelevant': {'total': 1, "vector": np.zeros((self.vector_length))},
            'very_irrelevant': {'total': 1, "vector": np.zeros((self.vector_length))},
        }
        self.distance_fn = distance_fn

    def initialize_candidates(self, candidates):
        self.candidates = candidates

    def update_candidates(self, new_candidates):
        self.probability_map = {
            'very_relevant': {'total': 1, "vector": np.zeros((self.vector_length))},
            'relevant': {'total': 1, "vector": np.zeros((self.vector_length))},
            'irrelevant': {'total': 1, "vector": np.zeros((self.vector_length))},
            'very_irrelevant': {'total': 1, "vector": np.zeros((self.vector_length))},
        }
        for new_candidate in new_candidates:
            for index, candidate in enumerate(self.candidates):
                if new_candidate[0] == candidate[0] and new_candidate[1] != "-":
                    image_id, relevance, distance = new_candidate
                    self.candidates[index] = (image_id, 0, relevance, distance)

    def get_significance(self, query_vector, candidates, feedbacks):
        feedbacks = feedbacks[:]
        # Convert query vector to binary
        bin_query_vector = np.where(query_vector > 0.5, 1, 0)

        # Convert feedback images to binary
        candidates = []
        for index, feedback in enumerate(feedbacks):
            image_id, relevance = feedback
            candidate_vector = self.feature_vectors[image_id][1]
            binary_candidate = np.where(candidate_vector > 0.5, 1, 0)
            candidates.append((image_id, relevance, 0, binary_candidate))
            feedbacks[index] = (image_id, relevance, binary_candidate)

        # Find significance of every feature using formula
        for candidate in candidates:
            # Based on relavance and irrelavance of results we calculate probability of strong relavance, relavance, irrelevance and very irrelevance for a given result
            image_id, relevance, distance, _ = candidate
            for feedback in feedbacks:
                feedback_image_id, feedback_relevance, _ = feedback
                if feedback_image_id == image_id:
                    vector = self.feature_vectors[image_id][1]
                    if RELEVANCE_MAP['very_relevant'] == relevance:
                        self.probability_map['very_relevant']['total'] += 1
                    if RELEVANCE_MAP['relevant'] == relevance:
                        self.probability_map['relevant']['total'] += 1
                    if RELEVANCE_MAP['irrelevant'] == relevance:
                        self.probability_map['irrelevant']['total'] += 1
                    if RELEVANCE_MAP['very_irrelevant'] == relevance:
                        self.probability_map['very_irrelevant']['total'] += 1

        feature_count = []
        size_features = len(self.feature_vectors[0][1])
        for index in range(size_features):
            feature_count.append({
                'very_relevant': 0,
                'relevant': 0,
                'irrelevant': 0,
                'very_irrelevant': 0
            })
        
        for feedback in feedbacks:
            feedback_image_id, feedback_relevance, binary_feature = feedback
            features = self.feature_vectors[image_id][1]
            
            for index, feature in enumerate(binary_feature):
                if feature == 0.0: continue
                if RELEVANCE_MAP['very_relevant'] == feedback_relevance:
                    feature_count[index]['very_relevant'] += 1
                if RELEVANCE_MAP['relevant'] == feedback_relevance:
                    feature_count[index]['relevant'] += 1
                if RELEVANCE_MAP['irrelevant'] == feedback_relevance:
                    feature_count[index]['irrelevant'] += 1
                if RELEVANCE_MAP['very_irrelevant'] == feedback_relevance:
                    feature_count[index]['very_irrelevant'] += 1

        significance = np.zeros((size_features))
        # For every feature we want to calculate significance
        for index in range(size_features):
            P_feature_given_vrel = feature_count[index]['very_relevant'] / self.probability_map['very_relevant']['total']
            P_feature_given_rel = feature_count[index]['relevant'] / self.probability_map['relevant']['total']
            P_feature_given_irrel = feature_count[index]['irrelevant'] / self.probability_map['irrelevant']['total']
            P_feature_given_virrel = feature_count[index]['very_irrelevant'] / self.probability_map['very_irrelevant']['total']
            alpha = 1.2
            denominator_fraction = 0.0001
            numerator = (P_feature_given_rel/ (1 - P_feature_given_rel + denominator_fraction)) + alpha * (P_feature_given_vrel/(1- P_feature_given_vrel + denominator_fraction)) + denominator_fraction
            denominator = (P_feature_given_irrel/ (1 - P_feature_given_irrel + denominator_fraction)) + alpha * (P_feature_given_virrel/(1- P_feature_given_virrel + denominator_fraction)) + denominator_fraction
            
            feature_significance = math.log(numerator/denominator)

            significance[index] = feature_significance

        # Multiply significance with query vector
        return np.array(query_vector) + 0.3 * significance * np.array(query_vector)
        # return significance
    
    def sort_results(self, query_vector, candidates):
        reranked_results = []
        # threshold = np.where(significance > 0, 1, 0)
        # negative_threshold = np.where(significance < 0, 1, 0)

        for candidate in candidates:
            image_id, relevance, distance = candidate
            candidate_vector = self.feature_vectors[image_id][1]
            new_distance = self.distance_fn(np.array(query_vector), np.array(candidate_vector))
            # negative_distance = self.distance_fn(np.array(query_vector) * threshold, threshold * np.array(candidate_vector))
            # new_distance = positive_distance + negative_distance
            reranked_results.append((image_id, new_distance, new_distance, relevance))
        reranked_results.sort(key=lambda x: x[2])
        return reranked_results

    def rerank_roccios(self, query_vector, new_candidates): #new_candidates = [(image_id, relevance, distance), (4200, 1, 0.0), (4222, 2, 5.7801666259765625)]
        self.update_candidates(new_candidates)
        for candidate in self.candidates:
            # Based on relavance and irrelavance of results we calculate probability of strong relavance, relavance, irrelevance and very irrelevance for a given result
            image_id, _, relevance, distance = candidate
            vector = self.feature_vectors[image_id][1]
            if RELEVANCE_MAP['very_relevant'] == relevance:
                self.probability_map['very_relevant']['total'] += 1
                self.probability_map['very_relevant']['vector'] = np.add(self.probability_map['very_relevant']['vector'], self.feature_vectors[image_id][1])
            if RELEVANCE_MAP['relevant'] == relevance:
                self.probability_map['relevant']['total'] += 1
                self.probability_map['relevant']['vector'] = np.add(self.probability_map['relevant']['vector'], self.feature_vectors[image_id][1])
            if RELEVANCE_MAP['irrelevant'] == relevance:
                self.probability_map['irrelevant']['total'] += 1
                self.probability_map['irrelevant']['vector'] = np.add(self.probability_map['irrelevant']['vector'], self.feature_vectors[image_id][1])
            if RELEVANCE_MAP['very_irrelevant'] == relevance:
                self.probability_map['very_irrelevant']['total'] += 1
                self.probability_map['very_irrelevant']['vector'] = np.add(self.probability_map['very_irrelevant']['vector'], self.feature_vectors[image_id][1])

        total_candidates = len(new_candidates)
        P_vrel = self.probability_map['very_relevant']['total']/total_candidates
        P_rel = self.probability_map['relevant']['total']/total_candidates
        P_irrel = self.probability_map['irrelevant']['total']/total_candidates
        P_virrel = self.probability_map['very_irrelevant']['total']/total_candidates
        alpha = 0.95
        beta = 0.9
        P_vrel= alpha*P_vrel + beta*(1/self.probability_map['very_relevant']['total'])*self.probability_map['very_relevant']['vector']
        P_rel= alpha*P_rel + beta*(1/self.probability_map['relevant']['total'])*self.probability_map['relevant']['vector']
        P_irrel= alpha*P_irrel + beta*(1/self.probability_map['irrelevant']['total'])*self.probability_map['irrelevant']['vector']
        P_virrel= alpha*P_virrel + beta*(1/self.probability_map['very_irrelevant']['total'])*self.probability_map['very_irrelevant']['vector']

        P_vrel = np.array(P_vrel)
        P_rel = np.array(P_rel)
        P_irrel = np.array(P_irrel)
        P_virrel = np.array(P_virrel)
        query_lambda = 0.8 # Higher lambda, more it favors the initial config
        strong_weights = 1.5
        # We then update the query vector using the probabilities and fetch the reranked results 
        new_query_vector = query_lambda * np.array(query_vector) + (1-query_lambda)*np.array((strong_weights*P_vrel + P_rel - P_irrel - strong_weights*P_virrel))

        return new_query_vector
