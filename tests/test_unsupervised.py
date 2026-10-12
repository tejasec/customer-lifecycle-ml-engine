import numpy as np
from mlo.unsupervised.clustering import CustomerKMeans, CustomerHierarchicalClustering
from mlo.unsupervised.density import CustomerDBSCAN
from mlo.unsupervised.dim_reduction import CustomerPCA, CustomerTSNE

def test_kmeans(sample_arrays):
    X, _, _ = sample_arrays
    km = CustomerKMeans(n_clusters=3, random_state=42).fit(X)
    labels = km.predict(X)
    assert len(labels) == len(X)
    assert set(np.unique(labels)).issubset({0, 1, 2})
    assert km.centroids.shape == (3, X.shape[1])
    assert km.inertia > 0

def test_hierarchical(sample_arrays):
    X, _, _ = sample_arrays
    hier = CustomerHierarchicalClustering(n_clusters=3)
    labels = hier.fit_predict(X)
    assert len(labels) == len(X)
    assert len(set(labels)) == 3

def test_dbscan(sample_arrays):
    X, _, _ = sample_arrays
    db = CustomerDBSCAN(eps=1.5, min_samples=3)
    labels = db.fit_predict(X)
    stats = db.get_cluster_stats()
    assert "n_clusters" in stats
    assert "n_noise_points" in stats
    assert 0.0 <= stats["noise_ratio"] <= 1.0

def test_dim_reduction(sample_arrays):
    X, _, _ = sample_arrays
    pca = CustomerPCA(n_components=2)
    X_pca = pca.fit_transform(X)
    assert X_pca.shape == (len(X), 2)
    assert 0.0 < pca.total_variance_explained <= 1.0

    tsne = CustomerTSNE(n_components=2, perplexity=10.0, random_state=42)
    X_tsne = tsne.fit_transform(X)
    assert X_tsne.shape == (len(X), 2)
