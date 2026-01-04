import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def entropy(y):
    """
    Compute entropy of a label array y.
    y: array-like of shape (n_samples,)
    """
    # Count occurrences of each class
    counts = np.bincount(y)
    probabilities = counts[counts > 0] / len(y)

    # H = -sum p * log2(p)
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(y, y_left, y_right):
    """
    Compute information gain of a split:
    IG = H(parent) - (n_L/n * H(left) + n_R/n * H(right))
    """
    H_parent = entropy(y)

    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 or n_right == 0:
        # No actual split
        return 0

    H_left = entropy(y_left)
    H_right = entropy(y_right)

    weighted_child_entropy = (n_left / n) * H_left + (n_right / n) * H_right

    return H_parent - weighted_child_entropy

class TreeNode:
    def __init__(self,
                 feature_index=None,
                 threshold=None,
                 left=None,
                 right=None,
                 value=None):
        """
        If value is not None, this is a leaf node.
        Otherwise, it is a decision node, with feature_index and threshold.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # class label for leaf

class DecisionTreeClassifierScratch:
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_impurity_decrease=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None

    def fit(self, X, y):
        """
        Build the decision tree from training data.
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        """
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        # ipdb.set_trace(context=10)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        """
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping conditions:
        # 1. Pure node (all labels the same)
        # 2. Not enough samples to split
        # 3. Reached max depth
        if (num_labels == 1 or
            num_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth)):
            leaf_value = self._most_common_label(y)
            # ipdb.set_trace(context = 10)
            return TreeNode(value=leaf_value)

        # Find the best split: feature and threshold
        best_feature, best_threshold, best_gain = self._best_split(X, y)

        # If no useful split (gain too small), create leaf
        if best_gain < self.min_impurity_decrease:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        # Split
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # Recursively build children
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        # Return decision node
        return TreeNode(feature_index=best_feature,
                        threshold=best_threshold,
                        left=left_child,
                        right=right_child)

    def _best_split(self, X, y):
        """
        Try all features and candidate thresholds and return the best split.
        """
        num_samples, num_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_index in range(num_features):
            # Consider unique values of this feature as candidate thresholds
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            # Optionally, we could consider midpoints between sorted unique values.
            # For simplicity, let's use unique_values directly as thresholds.
            for threshold in unique_values:
                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold

                y_left = y[left_indices]
                y_right = y[right_indices]

                gain = information_gain(y, y_left, y_right)

                if gain > best_gain:
                    # ipdb.set_trace(context=10)
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _most_common_label(self, y):
        """
        Return the most frequent class label in y.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the tree from the root to a leaf for sample x.
        """
        # If leaf
        if node.value is not None:
            return node.value

        # Go left or right depending on threshold
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class RandomForestClassifierScratch:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features # Number of features to consider at each split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            # 1. Bootstrapping: Sample indices with replacement
            n_samples = X.shape[0]
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            
            # Note: For true Feature Randomness, you would modify the _best_split 
            # in the DecisionTree class to only iterate over a random subset of indices.
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Gather predictions from every tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # tree_preds shape: (n_trees, n_samples). Transpose to (n_samples, n_trees)
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # 2. Majority Voting
        y_pred = [Counter(sample_preds).most_common(1)[0][0] for sample_preds in tree_preds]
        return np.array(y_pred)

from sklearn.datasets import make_moons

# 1. Generate synthetic data
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# 2. Train the Random Forest
clf = RandomForestClassifierScratch(n_trees=10, max_depth=5)
clf.fit(X, y)

# 3. Create a mesh grid for plotting
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict over the mesh grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap='RdBu')
plt.title("Random Forest Classification Boundary (Scratch)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
