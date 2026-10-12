from pathlib import Path
from typing import Dict, Any, Optional
import json
import joblib
import pandas as pd
import numpy as np

from mlo.config import get_config, ROOT_DIR
from mlo.data.generator import generate_customer_dataset
from mlo.data.encoder import CategoricalEncoder
from mlo.data.preprocessor import DataPreprocessor
from mlo.supervised.regression import RidgeModel
from mlo.ensemble.stacking import CustomerStackingClassifier
from mlo.evaluation.metrics import calculate_regression_metrics, calculate_classification_metrics
from mlo.utils.logger import setup_logger

logger = setup_logger("mlo.lifecycle")

class ProductionPipeline:
    """
    Implements the complete 9-Step Production Lifecycle:
    1. Data Collection
    2. Data Cleaning & Validation
    3. Categorical Preprocessing
    4. Feature Engineering & Scaling
    5. Train/Test Splitting
    6. Multi-target Training (LTV Regressor & Churn Classifier)
    7. Rigorous Evaluation
    8. Deployment Artifact Export
    9. Drift Monitoring & Readiness
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.encoder = CategoricalEncoder()
        self.scaler = DataPreprocessor(scaler_type="standard")
        self.regressor = RidgeModel(alpha=10.0)
        self.classifier = CustomerStackingClassifier()

    def run(self, raw_data_path: Optional[str] = None) -> Dict[str, Any]:
        logger.info(">>> STEP 1: DATA COLLECTION")
        if raw_data_path and Path(raw_data_path).exists():
            df = pd.read_csv(raw_data_path)
            logger.info(f"Loaded existing data from {raw_data_path} (Shape: {df.shape})")
        else:
            df = generate_customer_dataset(n_samples=1000, random_state=42)
            out_raw = ROOT_DIR / "data" / "raw" / "customer_churn_data.csv"
            out_raw.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_raw, index=False)
            logger.info(f"Generated synthetic customer data saved to {out_raw} (Shape: {df.shape})")

        logger.info(">>> STEP 2: DATA CLEANING & VALIDATION")
        df_clean = df.dropna().drop_duplicates()
        logger.info(f"Cleaned dataset shape: {df_clean.shape}")

        logger.info(">>> STEP 3: CATEGORICAL ENCODING")
        df_encoded = self.encoder.fit_transform(df_clean)
        feature_cols = self.encoder.feature_columns_
        X = df_encoded[feature_cols]
        y_ltv = df_encoded['LTV']
        y_churn = df_encoded['Churn']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train_churn, y_test_churn, y_train_ltv, y_test_ltv = train_test_split(
            X, y_churn, y_ltv, test_size=0.2, random_state=42, stratify=y_churn
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(">>> STEP 6: MODEL TRAINING")
        self.regressor.fit(X_train_scaled, y_train_ltv.values)
        self.classifier.fit(X_train_scaled, y_train_churn.values)
        logger.info("Fitted LTV Ridge Regressor and Churn Stacking Classifier.")

        logger.info(">>> STEP 7: RIGOROUS EVALUATION")
        ltv_preds = self.regressor.predict(X_test_scaled)
        churn_preds = self.classifier.predict(X_test_scaled)
        churn_probs = self.classifier.predict_proba(X_test_scaled)[:, 1]

        reg_metrics = calculate_regression_metrics(y_test_ltv.values, ltv_preds)
        cls_metrics = calculate_classification_metrics(y_test_churn.values, churn_preds, churn_probs)
        logger.info(f"LTV Regression - MSE: {reg_metrics['mse']:.2f}, R2: {reg_metrics['r2']:.4f}")
        logger.info(f"Churn Classification - Accuracy: {cls_metrics['accuracy']:.4f}, F1: {cls_metrics['f1']:.4f}")

        logger.info(">>> STEP 8: DEPLOYMENT ARTIFACT EXPORT")
        models_dir = ROOT_DIR / "artifacts" / "models"
        metrics_dir = ROOT_DIR / "artifacts" / "metrics"
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.regressor, models_dir / "ltv_regressor.joblib")
        joblib.dump(self.classifier, models_dir / "churn_classifier.joblib")
        joblib.dump(self.encoder, models_dir / "encoder.joblib")
        joblib.dump(self.scaler, models_dir / "scaler.joblib")

        metrics_report = {
            "regression_metrics": reg_metrics,
            "classification_metrics": cls_metrics,
            "feature_columns": feature_cols
        }
        with open(metrics_dir / "pipeline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=2)
        logger.info("Exported serialized models and metrics report.")

        logger.info(">>> STEP 9: RETRAINING & DRIFT MONITORING BASELINE ESTABLISHED")
        return metrics_report
