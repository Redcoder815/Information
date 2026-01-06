import numpy as np

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """Calculate the mean and standard deviation for each column."""
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        # Using ddof=0 to match scikit-learn's default population std
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """Apply the scaling: (X - mean) / std."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transforming.")
        
        X = np.array(X)
        # Prevent division by zero if std is 0
        scale_adj = np.where(self.scale_ == 0, 1, self.scale_)
        return (X - self.mean_) / scale_adj

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

# Example Usage
if __name__ == "__main__":
    data = np.array([[10, 2], [20, 4], [30, 6]])
    
    scaler = CustomStandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    print("Mean per column:", scaler.mean_)
    print("Std per column:", scaler.scale_)
    print("Scaled Data:\n", scaled_data)



