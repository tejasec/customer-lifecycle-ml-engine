from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

class CustomerKMeans:
    """
    K-Means behavioral clustering using iterative centroid recalculation.
    """
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def centroids(self) -> np.ndarray:
        return self.model.cluster_centers_

    @property
    def inertia(self) -> float:
        return float(self.model.inertia_)

class CustomerHierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering with Ward/Average linkage and Dendrogram support.
    """
    def __init__(self, n_clusters: int = 3, linkage_method: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)

    def plot_dendrogram(self, X: np.ndarray, max_d: Optional[float] = None, save_path: Optional[Path] = None):
        """
        Computes linkage matrix and renders hierarchical dendrogram.
        """
        Z = linkage(X, method=self.linkage_method)
        plt.figure(figsize=(10, 6))
        dendrogram(Z, truncate_mode="lastp", p=20, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
        plt.title(f"Hierarchical Clustering Dendrogram ({self.linkage_method.capitalize()} Linkage)")
        plt.xlabel("Cluster Size / Sample Index")
        plt.ylabel("Distance")
        if max_d:
            plt.axhline(y=max_d, c="r", linestyle="--", label=f"Cut-off distance = {max_d}")
            plt.legend()
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
