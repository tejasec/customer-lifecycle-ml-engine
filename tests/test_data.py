import numpy as np
import pandas as pd
from mlo.data.generator import generate_customer_dataset
from mlo.data.encoder import CategoricalEncoder
from mlo.data.preprocessor import DataPreprocessor

def test_generate_customer_dataset():
    df = generate_customer_dataset(n_samples=50, random_state=123)
    assert len(df) == 50
    assert set(['Age', 'MonthlySpend', 'AccountAgeMonths', 'SupportCalls', 'PlanType', 'Gender', 'LTV', 'Churn', 'IsAnomaly']).issubset(df.columns)
    assert df['Churn'].isin([0, 1]).all()
    assert df['IsAnomaly'].isin([0, 1]).all()

def test_categorical_encoder(sample_df):
    encoder = CategoricalEncoder()
    df_encoded = encoder.fit_transform(sample_df)
    assert 'PlanType_Encoded' in df_encoded.columns
    assert 'Gender_Male' in df_encoded.columns
    assert 'PlanType' not in df_encoded.columns
    assert 'Gender' not in df_encoded.columns
    assert len(encoder.feature_columns_) > 0

    # Test transform on new data
    df_test = sample_df.iloc[:5]
    df_test_enc = encoder.transform(df_test)
    assert set(encoder.feature_columns_).issubset(df_test_enc.columns)

def test_data_preprocessor(sample_encoded):
    encoder, df_encoded = sample_encoded
    X = df_encoded[encoder.feature_columns_]
    y = df_encoded['Churn']

    preproc = DataPreprocessor(scaler_type="standard", poly_degree=2)
    X_train, X_test, y_train, y_test = preproc.split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 80
    assert len(X_test) == 20

    X_train_sc = preproc.fit_transform(X_train.values)
    X_test_sc = preproc.transform(X_test.values)
    assert X_train_sc.shape[0] == 80
    assert X_test_sc.shape[0] == 20
    assert X_train_sc.shape[1] > X.shape[1]  # Expanded by polynomial degree 2
