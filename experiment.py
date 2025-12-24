import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Value to split at
        self.left = left            # Left child node
        self.right = right          # Right child node
        self.value = value          # Predicted value (for leaf nodes)

class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(value=np.mean(y))

        best_feat, best_thresh = self._best_split(X, y, n_features)
        if best_feat is None:
            return Node(value=np.mean(y))

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx
        
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, n_features):
        best_mse = float('inf')
        split_feat, split_thresh = None, None
        
        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_y = y[X[:, feat] <= thresh]
                right_y = y[X[:, feat] > thresh]
                
                if len(left_y) > 0 and len(right_y) > 0:
                    mse = self._calculate_mse(left_y, right_y)
                    if mse < best_mse:
                        best_mse, split_feat, split_thresh = mse, feat, thresh
        return split_feat, split_thresh

    def _calculate_mse(self, left_y, right_y):
        # Weighted MSE of child nodes
        n = len(left_y) + len(right_y)
        mse_l = np.var(left_y) * len(left_y)
        mse_r = np.var(right_y) * len(right_y)
        return (mse_l + mse_r) / n

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Generate synthetic data
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Train from-scratch model
model = DecisionTreeRegressorScratch(max_depth=4)
model.fit(X, y)
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = model.predict(X_test)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_pred, color="cornflowerblue", label="prediction", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression (From Scratch)")
plt.legend()
plt.show()
