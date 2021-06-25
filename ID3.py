import csv
import numpy as np
import math

NUM_FEATURES = 30


#  sorted_data = unsorted_data[unsorted_data[:, 1].argsort()]

class ID3:
    def __init__(self):
        pass

    @staticmethod
    def read_train(file):
        """
        Reads the data given in the file for train, returns the features and the labels
        :param file: the file name
        :return: tuple of two numpy arrays, features and labels
        """
        raw_data = np.genfromtxt(file, delimiter=',', dtype="S1," + "f8," * (NUM_FEATURES - 1) + "f8")
        features = np.zeros((raw_data.shape[0], NUM_FEATURES))
        labels = np.zeros((raw_data.shape[0]), dtype=np.long)
        for sample in range(raw_data.shape[0]):
            labels[sample] = (1 if raw_data[sample][0] == b'M' else 0)
            for feature in range(30):
                features[sample, feature] = raw_data[sample][feature + 1]
        return features, labels

    @staticmethod
    def compute_entropy(x):
        """
        Computes the entropy of one dimensional binary vector x
        :param x: one dimensional vector, each element is either 0 or 1
        :return: the entropy
        """
        pop_count = np.sum(x)
        p = pop_count / x.shape[0]
        return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)

    @staticmethod
    def construct_tree(train_features, train_labels, parent_majority=0) -> int:

        num_samples = train_features.shape[0]

        if train_labels.shape[0] == 0:
            # TODO: return leaf according to parent_majority
            return None

        if np.min(train_labels) == np.max(train_labels):
            # TODO: return leaf according to the value
            return None

        max_entropy_idx = 0
        max_entropy = 0
        chosen_t = None
        parent_entropy = ID3.compute_entropy(train_labels)
        for feature_idx in range(NUM_FEATURES):

            # Retrieve the feature values
            feature = train_features[:, feature_idx]

            # Sort the feature values and labels (according to feature values)
            sorted_feature = feature[feature.argsort()]
            sorted_labels = train_labels[feature.argsort()]

            # Choose the t that maximizes entropy
            max_entropy_t_idx = 0
            max_entropy_t = 0
            for t_idx in range(num_samples - 1):

                # Divide according to to the given t
                less_features = sorted_feature[0:t_idx + 1]
                less_labels = sorted_labels[0:t_idx + 1]
                greater_features = sorted_feature[t_idx + 1:]
                greater_labels = sorted_labels[t_idx + 1:]

                # Compute entropy gain of less/greater division
                entropy = parent_entropy - \
                    (1 / num_samples) * (less_labels.shape[0] * ID3.compute_entropy(less_labels)) - \
                    (1 / num_samples) * (greater_labels.shape[0] * ID3.compute_entropy(greater_labels))

                if entropy > max_entropy_t:
                    max_entropy_t = entropy
                    max_entropy_t_idx = t_idx

            if max_entropy_t > max_entropy:
                max_entropy = max_entropy_t
                max_entropy_idx = feature_idx
                chosen_t = max_entropy_t_idx

        # Choose feature max_entropy_idx and division chosen_t
        chosen_feature = train_features[max_entropy_idx]
        sorted_chosen_feature = chosen_feature[chosen_feature.argsort()]
        tval = (sorted_chosen_feature[chosen_t] + sorted_chosen_feature[chosen_t + 1]) / 2

        left_idxs = chosen_feature.argsort()[0:chosen_t + 1]
        right_idxs = chosen_feature.argsort()[chosen_t + 1:]

        left_features = train_features[left_idxs, :]
        left_labels = train_labels[left_idxs, :]
        right_features = train_features[right_idxs, :]
        right_labels = train_labels[right_idxs, :]

        # Run recursively on left and right, splitting according to feature_{max_entropy_idx} >= tval
        left = ID3.construct_tree(left_features, left_labels)
        right = ID3.construct_tree(right_features, right_labels)

        # Add node with feature_{max_entropy_idx} < tval connecting to left and
        # feature_{max_entropy_idx} >= tval connecting to right

        return None

    @staticmethod
    def fit_predict(train, test):

        # Read the train data
        train_features, train_labels = ID3.read_train(train)

        # Construct tree given the train data
        tree = ID3.construct_tree(train_features, train_labels)

        # TODO: Use the tree to infer on the test set


if '__main__' == __name__:
    ID3.fit_predict("train.csv", None)
