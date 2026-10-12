import pytest
import numpy as np
import pandas as pd
from mlo.data.generator import generate_customer_dataset
from mlo.data.encoder import CategoricalEncoder
from mlo.data.preprocessor import DataPreprocessor

@pytest.fixture
def sample_df():
    return generate_customer_dataset(n_samples=100, random_state=42)

@pytest.fixture
def sample_encoded(sample_df):
    encoder = CategoricalEncoder()
    df_encoded = encoder.fit_transform(sample_df)
    return encoder, df_encoded

@pytest.fixture
def sample_arrays(sample_encoded):
    encoder, df_encoded = sample_encoded
    X = df_encoded[encoder.feature_columns_].values
    y_reg = df_encoded['LTV'].values
    y_cls = df_encoded['Churn'].values
    preproc = DataPreprocessor(scaler_type="standard")
    X_scaled = preproc.fit_transform(X)
    return X_scaled, y_reg, y_cls
