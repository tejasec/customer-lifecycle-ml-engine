import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from mlo.pipeline.lifecycle import ProductionPipeline
from mlo.pipeline.monitoring import detect_feature_drift
from mlo.pipeline.serving import app

def test_production_pipeline_run(tmp_path):
    pipeline = ProductionPipeline()
    report = pipeline.run()
    assert "regression_metrics" in report
    assert "classification_metrics" in report
    assert report["regression_metrics"]["r2"] > 0.5
    assert report["classification_metrics"]["accuracy"] > 0.6

def test_feature_drift_detection(sample_df):
    baseline = sample_df.copy()
    current = sample_df.copy()

    # Artificially shift age distribution
    current['Age'] = current['Age'] + 50

    drift_report = detect_feature_drift(baseline, current, feature_cols=['Age', 'MonthlySpend'])
    assert drift_report["drift_detected"] is True
    assert "Age" in drift_report["drifted_features"]

def test_api_serving():
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # 2. LTV prediction
    payload = {
        "Age": 42,
        "MonthlySpend": 115.0,
        "AccountAgeMonths": 24,
        "SupportCalls": 1,
        "PlanType": "Standard",
        "Gender": "Female"
    }
    res_ltv = client.post("/predict/ltv", json=payload)
    assert res_ltv.status_code == 200
    assert "predicted_ltv" in res_ltv.json()
    assert isinstance(res_ltv.json()["predicted_ltv"], float)

    # 3. Churn prediction
    res_churn = client.post("/predict/churn", json=payload)
    assert res_churn.status_code == 200
    churn_data = res_churn.json()
    assert churn_data["churn_prediction"] in [0, 1]
    assert 0.0 <= churn_data["churn_probability"] <= 1.0
    assert churn_data["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]
