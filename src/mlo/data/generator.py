import numpy as np
import pandas as pd
from typing import Optional

def generate_customer_dataset(
    n_samples: int = 1000,
    random_state: int = 42,
    anomaly_ratio: float = 0.05
) -> pd.DataFrame:
    """
    Generates synthetic customer records for lifecycle, risk, and anomaly analysis.
    
    Includes:
    - Demographic & Behavioral: Age, MonthlySpend, AccountAgeMonths, SupportCalls, PlanType, Gender
    - Supervised Targets: LTV (continuous), Churn (binary)
    - Unsupervised Anomaly: TransactionAnomaly (binary flag for DBSCAN fraud evaluation)
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 70, size=n_samples)
    monthly_spend = rng.uniform(20.0, 200.0, size=n_samples)
    account_age_months = rng.integers(1, 60, size=n_samples)
    support_calls = rng.integers(0, 10, size=n_samples)
    plan_types = rng.choice(['Basic', 'Standard', 'Premium'], size=n_samples, p=[0.5, 0.3, 0.2])
    gender = rng.choice(['Male', 'Female'], size=n_samples)

    # LTV Target (regression target)
    ltv = (
        (monthly_spend * account_age_months * 0.85)
        + (age * 12.5)
        - (support_calls * 45.0)
        + rng.normal(0, 50.0, n_samples)
    )

    # Churn Target (classification target)
    logit = -1.5 + (0.45 * support_calls) - (0.05 * account_age_months) + (0.012 * monthly_spend) - (0.02 * age)
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (churn_prob > 0.5).astype(int)

    # Anomaly indicator for DBSCAN / Outlier detection
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_flags = np.zeros(n_samples, dtype=int)
    if n_anomalies > 0:
        anomaly_indices = rng.choice(n_samples, size=n_anomalies, replace=False)
        anomaly_flags[anomaly_indices] = 1
        # Inject anomalous behavioral spikes
        monthly_spend[anomaly_indices] += rng.uniform(250.0, 500.0, size=n_anomalies)
        support_calls[anomaly_indices] += rng.integers(10, 25, size=n_anomalies)

    df = pd.DataFrame({
        'Age': age,
        'MonthlySpend': np.round(monthly_spend, 2),
        'AccountAgeMonths': account_age_months,
        'SupportCalls': support_calls,
        'PlanType': plan_types,
        'Gender': gender,
        'LTV': np.round(ltv, 2),
        'Churn': churn,
        'IsAnomaly': anomaly_flags
    })
    return df
