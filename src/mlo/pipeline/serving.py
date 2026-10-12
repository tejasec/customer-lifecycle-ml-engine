from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlo.config import ROOT_DIR
from mlo.pipeline.lifecycle import ProductionPipeline

app = FastAPI(
    title="Customer Lifecycle & Risk Prediction Engine",
    description="Real-time inference API for LTV forecasting and Churn classification",
    version="0.1.0"
)

# In-memory artifact cache
_artifacts: Dict[str, Any] = {}

def get_artifacts():
    global _artifacts
    if not _artifacts:
        models_dir = ROOT_DIR / "artifacts" / "models"
        reg_path = models_dir / "ltv_regressor.joblib"
        cls_path = models_dir / "churn_classifier.joblib"
        enc_path = models_dir / "encoder.joblib"
        scl_path = models_dir / "scaler.joblib"

        if not (reg_path.exists() and cls_path.exists() and enc_path.exists() and scl_path.exists()):
            # Run pipeline once if models haven't been exported yet
            pipe = ProductionPipeline()
            pipe.run()

        _artifacts["regressor"] = joblib.load(reg_path)
        _artifacts["classifier"] = joblib.load(cls_path)
        _artifacts["encoder"] = joblib.load(enc_path)
        _artifacts["scaler"] = joblib.load(scl_path)
    return _artifacts

class CustomerPayload(BaseModel):
    Age: int = Field(..., ge=18, le=100, json_schema_extra={"example": 35})
    MonthlySpend: float = Field(..., ge=0.0, json_schema_extra={"example": 120.50})
    AccountAgeMonths: int = Field(..., ge=0, json_schema_extra={"example": 24})
    SupportCalls: int = Field(..., ge=0, json_schema_extra={"example": 2})
    PlanType: str = Field(..., json_schema_extra={"example": "Standard"})
    Gender: str = Field(..., json_schema_extra={"example": "Female"})

class LTVResponse(BaseModel):
    predicted_ltv: float

class ChurnResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str

@app.get("/health")
def health():
    return {"status": "healthy", "service": "mlo-prediction-engine"}

@app.post("/predict/ltv", response_model=LTVResponse)
def predict_ltv(customer: CustomerPayload):
    try:
        arts = get_artifacts()
        df = pd.DataFrame([customer.model_dump()])
        df_encoded = arts["encoder"].transform(df)
        X = df_encoded[arts["encoder"].feature_columns_]
        X_scaled = arts["scaler"].transform(X)
        ltv = float(arts["regressor"].predict(X_scaled)[0])
        return LTVResponse(predicted_ltv=round(ltv, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/churn", response_model=ChurnResponse)
def predict_churn(customer: CustomerPayload):
    try:
        arts = get_artifacts()
        df = pd.DataFrame([customer.model_dump()])
        df_encoded = arts["encoder"].transform(df)
        X = df_encoded[arts["encoder"].feature_columns_]
        X_scaled = arts["scaler"].transform(X)

        pred = int(arts["classifier"].predict(X_scaled)[0])
        prob = float(arts["classifier"].predict_proba(X_scaled)[0, 1])

        if prob >= 0.7:
            risk = "High Risk"
        elif prob >= 0.4:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return ChurnResponse(
            churn_prediction=pred,
            churn_probability=round(prob, 4),
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
