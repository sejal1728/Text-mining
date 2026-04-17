from sklearn.datasets import make_blobs
import numpy as np

X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

print("Data points:", X.shape[0])
print("Features:", X.shape[1])
print("Clusters:", len(set(y)))

print("\nFirst 5 samples:\n", X[:5])
print("\nMean:", np.mean(X, axis=0))
