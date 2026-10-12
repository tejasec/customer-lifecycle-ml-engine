from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

class DataPreprocessor:
    """
    Standard preprocessing workflows:
    - Train / test splitting
    - Feature scaling (StandardScaler or MinMaxScaler)
    - Polynomial feature expansions
    """
    def __init__(self, scaler_type: str = "standard", poly_degree: Optional[int] = None):
        self.scaler_type = scaler_type
        self.poly_degree = poly_degree
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False) if poly_degree else None

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X
        if self.poly:
            X_out = self.poly.fit_transform(X_out)
        return self.scaler.fit_transform(X_out)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X
        if self.poly:
            X_out = self.poly.transform(X_out)
        return self.scaler.transform(X_out)
