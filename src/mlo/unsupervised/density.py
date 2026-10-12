from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.cluster import DBSCAN

class CustomerDBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Identifies core clusters and flags anomalies/fraud as noise points (-1).
    """
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_: Optional[np.ndarray] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    def get_cluster_stats(self) -> Dict[str, Any]:
        if self.labels_ is None:
            raise ValueError("DBSCAN model has not been fitted yet.")
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = int(np.sum(self.labels_ == -1))
        noise_ratio = float(n_noise / len(self.labels_))
        return {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_ratio": noise_ratio
        }
