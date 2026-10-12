from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class RidgeModel:
    def __init__(self, alpha: float = 10.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class LassoModel:
    def __init__(self, alpha: float = 5.0):
        self.model = Lasso(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def zeroed_features_count(self) -> int:
        return int(np.sum(self.model.coef_ == 0))

def train_regression_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ridge_alpha: float = 10.0,
    lasso_alpha: float = 5.0
) -> Dict[str, Dict[str, Any]]:
    """
    Fits and evaluates:
    1. Linear Regression
    2. Polynomial Regression (degree 2)
    3. Ridge Regression (L2)
    4. Lasso Regression (L1 feature selection)
    """
    # 1. Linear
    lin = LinearRegression().fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)

    # 2. Polynomial
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_tr_poly = poly.fit_transform(X_train)
    X_te_poly = poly.transform(X_test)
    poly_reg = LinearRegression().fit(X_tr_poly, y_train)
    y_pred_poly = poly_reg.predict(X_te_poly)

    # 3. Ridge
    ridge = Ridge(alpha=ridge_alpha).fit(X_tr_poly, y_train)
    y_pred_ridge = ridge.predict(X_te_poly)

    # 4. Lasso
    lasso = Lasso(alpha=lasso_alpha).fit(X_tr_poly, y_train)
    y_pred_lasso = lasso.predict(X_te_poly)

    return {
        "Linear Regression": {
            "model": lin,
            "mse": float(mean_squared_error(y_test, y_pred_lin)),
            "r2": float(r2_score(y_test, y_pred_lin)),
        },
        "Polynomial Regression": {
            "model": poly_reg,
            "poly_transformer": poly,
            "mse": float(mean_squared_error(y_test, y_pred_poly)),
            "r2": float(r2_score(y_test, y_pred_poly)),
        },
        "Ridge Regression (L2)": {
            "model": ridge,
            "mse": float(mean_squared_error(y_test, y_pred_ridge)),
            "r2": float(r2_score(y_test, y_pred_ridge)),
        },
        "Lasso Regression (L1)": {
            "model": lasso,
            "mse": float(mean_squared_error(y_test, y_pred_lasso)),
            "r2": float(r2_score(y_test, y_pred_lasso)),
            "zeroed_features": int(np.sum(lasso.coef_ == 0)),
            "total_features": len(lasso.coef_),
        },
    }
