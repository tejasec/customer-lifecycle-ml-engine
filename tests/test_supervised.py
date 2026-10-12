import numpy as np
from mlo.supervised.regression import train_regression_suite, LinearRegressionModel, RidgeModel, LassoModel
from mlo.supervised.classification import train_classification_suite, ClassificationSuite
from mlo.supervised.trees import CustomerDecisionTree, compare_tree_architectures

def test_regression_suite(sample_arrays):
    X, y_reg, _ = sample_arrays
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y_reg[:80], y_reg[80:]

    res = train_regression_suite(X_train, y_train, X_test, y_test)
    assert "Linear Regression" in res
    assert "Polynomial Regression" in res
    assert "Ridge Regression (L2)" in res
    assert "Lasso Regression (L1)" in res
    assert res["Linear Regression"]["r2"] > 0.5
    assert "zeroed_features" in res["Lasso Regression (L1)"]

def test_classification_suite(sample_arrays):
    X, _, y_cls = sample_arrays
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y_cls[:80], y_cls[80:]

    suite, res_df = train_classification_suite(X_train, y_train, X_test, y_test)
    assert len(res_df) == 5
    assert set(['Accuracy', 'Precision', 'Recall', 'F1-Score']).issubset(res_df.columns)

def test_decision_tree_architectures(sample_arrays):
    X, _, y_cls = sample_arrays
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y_cls[:80], y_cls[80:]

    results = compare_tree_architectures(X_train, y_train, X_test, y_test)
    assert results["pre_pruned_entropy"]["depth"] <= 3
    assert len(results["pre_pruned_entropy"]["rules"]) > 0
