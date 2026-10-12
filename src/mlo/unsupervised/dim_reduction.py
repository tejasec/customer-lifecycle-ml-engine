from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class CustomerPCA:
    """
    Principal Component Analysis (PCA) for linear dimensionality reduction
    and variance preservation.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.model.explained_variance_ratio_

    @property
    def total_variance_explained(self) -> float:
        return float(np.sum(self.model.explained_variance_ratio_))

class CustomerTSNE:
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE) for non-linear manifold learning.
    """
    def __init__(self, n_components: int = 2, perplexity: float = 30.0, random_state: int = 42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self.model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(X)
