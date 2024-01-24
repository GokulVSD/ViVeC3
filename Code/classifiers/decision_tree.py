import math

class Node:
    def __init__(self, feature_index=None, feature_value=None, l_branch=None, r_branch=None, predicted_value=None):
        self.feature_index = feature_index
        self.feature_value = feature_value
        self.l_branch = l_branch
        self.r_branch = r_branch
        self.predicted_value = predicted_value

class DecisionTreeClassifier:
    def __init__(self, mode='gini', mimimum_samples_leaf=1, max_depth=None):
        self.root = None
        self.mode = mode
        self.mimimum_samples_leaf = mimimum_samples_leaf
        if max_depth is None or max_depth == 0:
            self.max_depth = float('inf')
        else:
            self.max_depth = max_depth
        
        


    def count_per_label(self, dataset):
        records_per_label = {}
        for row in dataset:
            y = row[-1]
            if y in records_per_label:
                records_per_label[y] += 1
            else:
                records_per_label[y] = 1
        return records_per_label


    def calculate_uncertainty(self, dataset):

        records_per_label = self.count_per_label(dataset)
        record_count = len(dataset)
        if self.mode == 'gini':
            impurity = 1
            for k, v in records_per_label.items():
                prob_label = v/record_count
                impurity -= prob_label ** 2
            return impurity
        if self.mode == 'entropy':
            entropy = 0
            for k, v in records_per_label.items():
                prob_label = v/record_count
                entropy += -prob_label * math.log2(prob_label)
            return entropy


    def calculate_gain(self, l_part, r_part, current_uncertainty):

        p = len(l_part)/(len(l_part) + len(r_part))
        return current_uncertainty - p * self.calculate_uncertainty(l_part) - (1-p) * self.calculate_uncertainty(r_part)


    def find_best_split(self, dataset):
        best_gain = 0
        best_feature = None
        best_value = None
        current_uncertainty = self.calculate_uncertainty(dataset)
        n_features = len(dataset[0]) - 1

        for feature_index in range(1, n_features):
            feature_values = set([row[feature_index] for row in dataset])
            for value in feature_values:
                l_part, r_part = self.partition(dataset, feature_index, value)
                
                if len(l_part) == 0 or len(r_part) == 0:
                    continue
                
                tmp_gain = self.calculate_gain(l_part, r_part, current_uncertainty)

                if tmp_gain >= best_gain:
                    best_gain = tmp_gain
                    best_feature = feature_index
                    best_value = value
        
        return best_gain, best_feature, best_value


    def partition(self, dataset, feature_index, feature_value):
        l_part = []
        r_part = []

        for row in dataset:
            if row[feature_index] >= feature_value:
                r_part.append(row)
            else:
                l_part.append(row)
        
        return l_part, r_part

    def calculate_leaf_value(self, dataset):
        l = [row[-1] for row in dataset]
        return max(l, key=l.count)


    def build_tree(self, dataset, current_depth=0):
        if len(dataset) >= self.mimimum_samples_leaf and current_depth < self.max_depth:
            gain, feature_index, feature_value = self.find_best_split(dataset)

            if gain == 0:
                leaf_value = self.calculate_leaf_value(dataset)
                return Node(predicted_value=leaf_value)
            
            l_part, r_part = self.partition(dataset, feature_index, feature_value)

            current_depth += 1

            l_branch = self.build_tree(l_part, current_depth)
            r_branch = self.build_tree(r_part, current_depth)

            return Node(feature_index=feature_index, 
                        feature_value=feature_value, 
                        l_branch=l_branch,
                        r_branch=r_branch,
                        predicted_value=None)
        else:
            leaf_value = self.calculate_leaf_value(dataset)
            return Node(predicted_value=leaf_value)


    def format_database(self, database):
        dataset = []
        for k,v in database.items():
            row = []
            image_id = k
            image_label, feature_vector = v
            row.append(image_id)
            # row.extend(feature_vector.tolist())
            fv = [round(e,2) for e in feature_vector.tolist()]
            row.extend(fv)
            row.append(image_label)
            dataset.append(row)
        return dataset

    def fit(self, database):
        dataset = self.format_database(database)
        self.root = self.build_tree(dataset)


    def classify(self, database):
        dataset = self.format_database(database)
        prediction_result = []
        for row in dataset:
            predicted_class = self.predict_one(row, self.root)
            prediction_result.append({"test_label": row[-1], "predicted_label": predicted_class})
            print(f"IMG_ID: {row[0]}\tTrue label: {row[-1]}\tPred label: {predicted_class}")
        return prediction_result
    
    def predict_one(self, row, node):
        if node.predicted_value != None:
            return node.predicted_value
        else:
            feature_index = node.feature_index
            feature_value = node.feature_value

            if row[feature_index] >= feature_value:
                return self.predict_one(row, node.r_branch)
            else:
                return self.predict_one(row, node.l_branch)
    
    def print_tree(self, node=None, indent = " "):
        if not node:
            node = self.root

        if node.predicted_value is not None:
            print(node.predicted_value)
        else:
            print("featureIdx_"+str(node.feature_index), "<", node.feature_value)
            print("%sleft:" % (indent), end="")
            self.print_tree(node.l_branch, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(node.r_branch, indent + indent)

    def tree_depth(self, node=None):
        if not node:
            node = self.root

        if node.predicted_value != None:
            return 0
        else:
            l_depth = self.tree_depth(node.l_branch)
            r_depth = self.tree_depth(node.r_branch)
        
            if l_depth > r_depth:
                return l_depth + 1
            else:
                return r_depth + 1