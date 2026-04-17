from sklearn.datasets import make_blobs
from sklearn.cluster import Birch

X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

model = Birch(n_clusters=3)
labels = model.fit_predict(X)

print("BIRCH Labels:\n", labels)
