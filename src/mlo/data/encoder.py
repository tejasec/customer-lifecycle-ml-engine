from typing import Dict, List, Optional
import pandas as pd

class CategoricalEncoder:
    """
    Handles categorical encoding transformations:
    - Ordinal encoding: Map ordered categories to integers (e.g., PlanType)
    - One-hot encoding: Dummy encoding for nominal categories (e.g., Gender)
    """
    def __init__(self, ordinal_mappings: Optional[Dict[str, Dict[str, int]]] = None):
        self.ordinal_mappings = ordinal_mappings or {
            'PlanType': {'Basic': 0, 'Standard': 1, 'Premium': 2}
        }
        self.nominal_columns: List[str] = ['Gender']
        self.feature_columns_: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        for col, mapping in self.ordinal_mappings.items():
            if col in df_encoded.columns:
                df_encoded[f"{col}_Encoded"] = df_encoded[col].map(mapping).fillna(0).astype(int)
                df_encoded = df_encoded.drop(columns=[col])

        # One-hot encode nominal columns
        present_nominal = [c for c in self.nominal_columns if c in df_encoded.columns]
        if present_nominal:
            df_encoded = pd.get_dummies(df_encoded, columns=present_nominal, drop_first=True, dtype=int)

        self.feature_columns_ = [c for c in df_encoded.columns if c not in ['LTV', 'Churn', 'IsAnomaly']]
        return df_encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        for col, mapping in self.ordinal_mappings.items():
            if col in df_encoded.columns:
                df_encoded[f"{col}_Encoded"] = df_encoded[col].map(mapping).fillna(0).astype(int)
                df_encoded = df_encoded.drop(columns=[col])

        present_nominal = [c for c in self.nominal_columns if c in df_encoded.columns]
        if present_nominal:
            df_encoded = pd.get_dummies(df_encoded, columns=present_nominal, drop_first=True, dtype=int)

        # Ensure all fitted feature columns exist
        for col in self.feature_columns_:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        return df_encoded
