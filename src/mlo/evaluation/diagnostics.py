from typing import Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import learning_curve

def compute_learning_curves(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
    cv: int = 5,
    scoring: str = "neg_mean_squared_error"
) -> Dict[str, Any]:
    sizes, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring, n_jobs=-1
    )
    return {
        "train_sizes": sizes.tolist(),
        "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
        "train_scores_std": np.std(train_scores, axis=1).tolist(),
        "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
        "test_scores_std": np.std(test_scores, axis=1).tolist()
    }

def diagnose_bias_variance(
    train_score: float,
    test_score: float,
    benchmark_score: float = 0.85,
    variance_threshold: float = 0.10
) -> Dict[str, Any]:
    """
    Diagnoses learning state:
    - High Bias (Underfitting): Both train and test scores fall significantly below benchmark
    - High Variance (Overfitting): High train score but significant divergence from test score
    - Balanced: Both scores are high and close together
    """
    gap = abs(train_score - test_score)
    if train_score < benchmark_score and test_score < benchmark_score:
        diagnosis = "High Bias (Underfitting)"
        remedy = "Increase model capacity, engineer richer features, or reduce regularization."
    elif gap > variance_threshold:
        diagnosis = "High Variance (Overfitting)"
        remedy = "Add regularization (L1/L2), gather more training samples, or prune model depth."
    else:
        diagnosis = "Balanced Fit"
        remedy = "Model generalizes well across training and validation splits."

    return {
        "train_score": train_score,
        "test_score": test_score,
        "generalization_gap": gap,
        "diagnosis": diagnosis,
        "recommendation": remedy
    }
