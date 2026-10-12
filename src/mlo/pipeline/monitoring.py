from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def detect_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
    p_value_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Performs Kolmogorov-Smirnov (KS) two-sample test between baseline distribution
    and incoming current distribution to detect covariate shift / data drift.
    """
    drift_report: Dict[str, Any] = {}
    drifted_features: List[str] = []

    for col in feature_cols:
        if col in baseline_df.columns and col in current_df.columns:
            stat, p_val = ks_2samp(baseline_df[col].dropna(), current_df[col].dropna())
            is_drift = bool(p_val < p_value_threshold)
            drift_report[col] = {
                "ks_statistic": float(stat),
                "p_value": float(p_val),
                "drift_detected": is_drift
            }
            if is_drift:
                drifted_features.append(col)

    retrain_recommended = len(drifted_features) > 0
    return {
        "drift_detected": retrain_recommended,
        "drifted_features": drifted_features,
        "features_tested": len(drift_report),
        "feature_details": drift_report
    }
