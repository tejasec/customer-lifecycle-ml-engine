from typing import Dict, Any, Optional
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_hyperparameters(
    model: Any,
    param_grid: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    method: str = "grid",
    cv: int = 5,
    scoring: str = "f1",
    n_iter: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    if method == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
    elif method == "random":
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown tuning method: {method}")

    search.fit(X, y)
    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_
    }
