from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

def perform_cross_validation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    scoring: str = "r2",
    random_state: int = 42
) -> Dict[str, Any]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    return {
        "scoring": scoring,
        "n_splits": n_splits,
        "fold_scores": scores.tolist(),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores))
    }

def perform_stratified_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    scoring: str = "f1",
    random_state: int = 42
) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    return {
        "scoring": scoring,
        "n_splits": n_splits,
        "fold_scores": scores.tolist(),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores))
    }
