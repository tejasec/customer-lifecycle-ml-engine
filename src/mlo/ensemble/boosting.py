from typing import Optional, Dict, Any
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb

class CustomerAdaBoost:
    """
    AdaBoost: Sequentially adjusts sample weights on hard misclassified observations.
    """
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomerAdaBoost":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class CustomerGradientBoosting:
    """
    Gradient Boosting: Optimizes arbitrary differentiable loss function via residual fitting.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomerGradientBoosting":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class CustomerXGBoost:
    """
    XGBoost: Highly scalable, regularized gradient boosted decision tree engine.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            eval_metric="logloss"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomerXGBoost":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
