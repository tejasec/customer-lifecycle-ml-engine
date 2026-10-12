import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from mlo.evaluation.metrics import calculate_regression_metrics, calculate_classification_metrics
from mlo.evaluation.validation import perform_cross_validation, perform_stratified_cv
from mlo.evaluation.diagnostics import diagnose_bias_variance
from mlo.evaluation.tuning import tune_hyperparameters

def test_metrics():
    y_true_reg = np.array([10.0, 20.0, 30.0])
    y_pred_reg = np.array([11.0, 19.0, 30.0])
    reg_m = calculate_regression_metrics(y_true_reg, y_pred_reg)
    assert reg_m["mse"] > 0
    assert reg_m["rmse"] > 0
    assert reg_m["r2"] > 0.9

    y_true_cls = np.array([0, 1, 1, 0, 1])
    y_pred_cls = np.array([0, 1, 1, 0, 0])
    y_prob_cls = np.array([0.1, 0.9, 0.8, 0.2, 0.4])
    cls_m = calculate_classification_metrics(y_true_cls, y_pred_cls, y_prob_cls)
    assert cls_m["accuracy"] == 0.8
    assert "confusion_matrix" in cls_m
    assert cls_m["roc_auc"] is not None

def test_cross_validation(sample_arrays):
    X, _, y_cls = sample_arrays
    lr = LogisticRegression(random_state=42)
    cv_res = perform_stratified_cv(lr, X, y_cls, n_splits=3, scoring="accuracy")
    assert cv_res["n_splits"] == 3
    assert len(cv_res["fold_scores"]) == 3
    assert 0.0 <= cv_res["mean_score"] <= 1.0

def test_bias_variance_diagnosis():
    diag_underfit = diagnose_bias_variance(0.50, 0.52, benchmark_score=0.85)
    assert "Underfitting" in diag_underfit["diagnosis"]

    diag_overfit = diagnose_bias_variance(0.95, 0.70, benchmark_score=0.85)
    assert "Overfitting" in diag_overfit["diagnosis"]

    diag_balanced = diagnose_bias_variance(0.90, 0.88, benchmark_score=0.85)
    assert "Balanced" in diag_balanced["diagnosis"]

def test_hyperparameter_tuning(sample_arrays):
    X, _, y_cls = sample_arrays
    dt = DecisionTreeClassifier(random_state=42)
    grid = {'max_depth': [2, 4], 'criterion': ['gini', 'entropy']}
    res = tune_hyperparameters(dt, grid, X, y_cls, method="grid", cv=3, scoring="accuracy")
    assert "best_params" in res
    assert res["best_score"] > 0
