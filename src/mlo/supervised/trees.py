from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

class CustomerDecisionTree:
    """
    Decision Tree architecture supporting:
    - Splitting criteria: Gini Index vs. Entropy (Information Gain)
    - Pre-pruning: max_depth, min_samples_split, min_samples_leaf
    - Post-pruning: Minimal Cost-Complexity Pruning (ccp_alpha)
    """
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        ccp_alpha: float = 0.0,
        random_state: int = 42
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            ccp_alpha=ccp_alpha,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))

    def get_rules(self, feature_names: Optional[List[str]] = None) -> str:
        return export_text(self.model, feature_names=feature_names)

    @property
    def depth(self) -> int:
        return int(self.model.get_depth())

    @property
    def n_leaves(self) -> int:
        return int(self.model.get_n_leaves())

def compare_tree_architectures(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compares Unpruned Gini Tree vs. Pre-Pruned Entropy Tree.
    """
    unpruned = CustomerDecisionTree(criterion="gini", max_depth=None).fit(X_train, y_train)
    pruned = CustomerDecisionTree(criterion="entropy", max_depth=3, min_samples_split=5).fit(X_train, y_train)

    return {
        "unpruned_gini": {
            "model": unpruned,
            "train_acc": unpruned.score(X_train, y_train),
            "test_acc": unpruned.score(X_test, y_test),
            "depth": unpruned.depth,
            "n_leaves": unpruned.n_leaves
        },
        "pre_pruned_entropy": {
            "model": pruned,
            "train_acc": pruned.score(X_train, y_train),
            "test_acc": pruned.score(X_test, y_test),
            "depth": pruned.depth,
            "n_leaves": pruned.n_leaves,
            "rules": pruned.get_rules(feature_names=feature_names)
        }
    }
