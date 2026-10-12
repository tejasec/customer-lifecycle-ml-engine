from typing import List, Tuple, Any, Optional
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class CustomerStackingClassifier:
    """
    Two-Level Stacking Classifier:
    - Level 0 Base Estimators: Decision Tree, KNN, Random Forest
    - Level 1 Meta-Learner: Logistic Regression (optimally blends predictions)
    """
    def __init__(
        self,
        base_estimators: Optional[List[Tuple[str, Any]]] = None,
        final_estimator: Optional[Any] = None,
        cv: int = 5,
        random_state: int = 42
    ):
        self.base_estimators = base_estimators or [
            ('dt', DecisionTreeClassifier(max_depth=4, random_state=random_state)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=4, random_state=random_state))
        ]
        self.final_estimator = final_estimator or LogisticRegression(random_state=random_state)
        self.cv = cv
        self.model = StackingClassifier(
            estimators=self.base_estimators,
            final_estimator=self.final_estimator,
            cv=self.cv
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomerStackingClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))
