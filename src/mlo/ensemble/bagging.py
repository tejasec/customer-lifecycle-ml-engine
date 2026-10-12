from typing import Optional, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class CustomerRandomForest:
    """
    Bagging ensemble: Random Forest aggregating decision trees over bootstrap samples.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 5,
        bootstrap: bool = True,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=bootstrap,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomerRandomForest":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_
