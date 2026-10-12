import numpy as np
from mlo.ensemble.bagging import CustomerRandomForest
from mlo.ensemble.boosting import CustomerAdaBoost, CustomerGradientBoosting, CustomerXGBoost
from mlo.ensemble.stacking import CustomerStackingClassifier

def test_random_forest(sample_arrays):
    X, _, y_cls = sample_arrays
    rf = CustomerRandomForest(n_estimators=20, max_depth=3).fit(X, y_cls)
    preds = rf.predict(X)
    assert len(preds) == len(y_cls)
    assert len(rf.feature_importances) == X.shape[1]

def test_boosting_suite(sample_arrays):
    X, _, y_cls = sample_arrays
    ada = CustomerAdaBoost(n_estimators=10).fit(X, y_cls)
    assert len(ada.predict(X)) == len(y_cls)

    gb = CustomerGradientBoosting(n_estimators=10, max_depth=2).fit(X, y_cls)
    assert len(gb.predict(X)) == len(y_cls)

    xgb_clf = CustomerXGBoost(n_estimators=10, max_depth=2).fit(X, y_cls)
    assert len(xgb_clf.predict(X)) == len(y_cls)

def test_stacking_classifier(sample_arrays):
    X, _, y_cls = sample_arrays
    stack = CustomerStackingClassifier().fit(X, y_cls)
    preds = stack.predict(X)
    probs = stack.predict_proba(X)
    assert len(preds) == len(y_cls)
    assert probs.shape == (len(y_cls), 2)
    assert stack.score(X, y_cls) > 0.5
