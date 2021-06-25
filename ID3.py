import csv
import numpy as np
import math

NUM_FEATURES = 30


class Node:
    """
    Represents a single node in the computation graph
    """

    def __init__(self, left=None, right=None, feature=-1, t=-1, leaf=-1):
        """
        Initializes the node
        :param left: the node to the left
        :param right: the node to the right
        :param feature: the feature by which this node divides
        :param t: the division value
        :param leaf: whether or not the node is a leaf. If this value is -1, then the node is not a leaf. Otherwise,
        this value is the value assigned to the leaf.
        """
        self.left = left
        self.right = right
        self.feature = feature
        self.t = t
        self.leaf = leaf


class ID3:
    def __init__(self):
        pass

    @staticmethod
    def compute_entropy(x):
        """
        Computes the entropy of one dimensional binary vector x
        :param x: one dimensional vector, each element is either 0 or 1
        :return: the entropy
        """
        pop_count = np.sum(x)
        p = pop_count / x.shape[0]
        if p == 0 or p == 1:
            return 0
        return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)

    @staticmethod
    def construct_tree(train_features, train_labels, set_size_pruning) -> Node:

        num_samples = train_features.shape[0]

        if np.min(train_labels) == np.max(train_labels):
            # Return leaf according to the value
            return Node(leaf=np.min(train_labels))

        if num_samples <= set_size_pruning:
            # Return leaf according to majority
            n = Node(leaf=(1 if (np.sum(train_labels) >= (num_samples/2)) else 0))
            return n

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

            if sorted_feature[0] == sorted_feature[-1]:
                continue

            # Choose the t that maximizes entropy
            max_entropy_t_idx = 0
            max_entropy_t = 0
            for t_idx in range(num_samples - 1):

                # If two identical values, skip
                if sorted_feature[t_idx] == sorted_feature[t_idx+1]:
                    continue

                # Divide according to to the given t
                less_labels = sorted_labels[0:t_idx + 1]
                greater_labels = sorted_labels[t_idx + 1:]

                # Compute entropy gain of less/greater division
                entropy = parent_entropy - \
                    (1 / num_samples) * (less_labels.shape[0] * ID3.compute_entropy(less_labels)) - \
                    (1 / num_samples) * (greater_labels.shape[0] * ID3.compute_entropy(greater_labels))

                if entropy >= max_entropy_t:
                    max_entropy_t = entropy
                    max_entropy_t_idx = t_idx

            if max_entropy_t >= max_entropy:
                max_entropy = max_entropy_t
                max_entropy_idx = feature_idx
                chosen_t = max_entropy_t_idx

        # Choose feature max_entropy_idx and division chosen_t
        chosen_feature = train_features[:, max_entropy_idx]
        sorted_chosen_feature = chosen_feature[chosen_feature.argsort()]
        tval = (sorted_chosen_feature[chosen_t] + sorted_chosen_feature[chosen_t + 1]) / 2

        left_idxs = chosen_feature.argsort()[0:chosen_t + 1]
        right_idxs = chosen_feature.argsort()[chosen_t + 1:]

        left_features = train_features[left_idxs, :]
        left_labels = train_labels[left_idxs]
        right_features = train_features[right_idxs, :]
        right_labels = train_labels[right_idxs]

        # Run recursively on left and right, splitting according to feature_{max_entropy_idx} >= tval
        left = ID3.construct_tree(left_features, left_labels, set_size_pruning)
        right = ID3.construct_tree(right_features, right_labels, set_size_pruning)

        return Node(left=left, right=right, feature=max_entropy_idx, t=tval)

    @staticmethod
    def depth(node) -> int:
        if node.leaf == -1:
            return max(ID3.depth(node.left), ID3.depth(node.right)) + 1
        return 0

    @staticmethod
    def fit_predict(train, test, set_size_pruning=1):
        """
        Fits according to train and predicts on test
        :param train: (num samples) x (num features + 1) vector where first column is label (1 for sick, 0 for not)
        :param test: (num samples) x (num features) vector
        :param set_size_pruning: the minimal set size for a node to not be a leaf (default 1 = no pruning)
        :return: (num_samples) vector for predictions on test
        """

        # Construct tree given the train data
        tree = ID3.construct_tree(train[:, 1:], train[:, 0], set_size_pruning)

        num_samples = test.shape[0]
        predictions = np.zeros(num_samples)
        for sample in range(num_samples):

            # Retrieve the sample
            x = test[sample]

            # Traverse the tree according to the sample
            node = tree
            while node.leaf == -1:
                if x[node.feature] < node.t:
                    node = node.left
                else:
                    node = node.right

            # Save the prediction
            predictions[sample] = node.leaf

        return predictions
