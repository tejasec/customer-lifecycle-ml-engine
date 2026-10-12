# Customer Lifecycle & Risk Prediction Engine

An end-to-end, unified Machine Learning and Mathematical Optimization pipeline designed to model customer value, churn risk, behavioral clusters, and custom gradient descent optimization engines.

---

## 🎯 Architecture Overview

This project is structured into five modular phases:

```
Raw Customer Records (Age, Spend, Tenancy, Support Calls, Plan, Anomalies)
       │
       ▼
Phase 1: Supervised Predictors ───► LTV Regression, Churn Classification, Decision Trees
       │
       ▼
Phase 2: Discovery Engine     ───► K-Means, DBSCAN Outliers, PCA & t-SNE
       │
       ▼
Phase 3: Validation Framework ───► Stratified K-Fold, Grid Search, Bias-Variance Diagnostics
       │
       ▼
Phase 4: Custom Optimizer     ───► Scratch GD Engine (BGD, SGD, Adam, Schedulers)
       │
       ▼
Phase 5: Production Pipeline   ───► Random Forest, XGBoost, Stacking Meta-Learner & API
```

---

## 📂 Repository Structure

```
MLO/
├── configs/
│   ├── config.yaml                    # Global paths and pipeline settings
│   └── model_configs.yaml             # Hyperparameters for all models
├── data/
│   ├── raw/                           # Raw synthetic / external datasets
│   └── processed/                     # Preprocessed, scaled, train/test splits
├── artifacts/
│   ├── models/                        # Serialized models (.joblib / .pkl)
│   ├── metrics/                       # Evaluation reports (.json / .csv)
│   └── plots/                         # Evaluation, cluster, & loss curves (.png)
├── docs/
│   └── concept_mapping.md             # Architecture and concept mapping table
├── notebooks/
│   ├── 01_phase1_supervised_predictors.ipynb
│   ├── 02_phase2_unsupervised_discovery.ipynb
│   ├── 03_phase3_validation_and_tuning.ipynb
│   ├── 04_phase4_custom_optimizers.ipynb
│   └── 05_phase5_ensemble_production_pipeline.ipynb
├── scripts/
│   ├── generate_dataset.py            # CLI data generation utility
│   ├── run_phase.py                   # Multi-phase runner CLI (--phase 1..5)
│   ├── run_pipeline.py                # 9-Step production lifecycle runner
│   └── serve.py                       # FastAPI model server runner
├── src/
│   └── mlo/
│       ├── config.py                  # Path and config loader
│       ├── data/                      # Generator, Encoders, Preprocessors
│       ├── supervised/                # Regressors, Classifiers, Trees
│       ├── unsupervised/              # K-Means, DBSCAN, PCA, t-SNE
│       ├── evaluation/                # Metrics, K-Fold, Diagnostics, GridSearch
│       ├── optimization/              # Scratch Losses, Optimizers, Schedulers
│       ├── ensemble/                  # Random Forest, Boosting, Stacking
│       ├── pipeline/                  # 9-Step Lifecycle, Serving API, Drift Monitoring
│       └── utils/                     # Logger and plotting utilities
└── tests/                             # Full automated pytest suite
```

---

## 🚀 Quickstart & Setup

### 1. Initialize Virtual Environment with Python 3.12
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 2. Install Project Dependencies
```bash
uv pip install -e ".[dev]"
```

### 3. Generate Synthetic Customer Dataset
```bash
python scripts/generate_dataset.py --samples 1000 --output data/raw/customer_churn_data.csv
```

### 4. Run Any Milestone Phase
```bash
python scripts/run_phase.py --phase 1   # Phase 1: Supervised Learning
python scripts/run_phase.py --phase 2   # Phase 2: Unsupervised Learning
python scripts/run_phase.py --phase 3   # Phase 3: Evaluation & Tuning
python scripts/run_phase.py --phase 4   # Phase 4: Optimization Engine
python scripts/run_phase.py --phase 5   # Phase 5: Ensembles & Production
```

### 5. Run the 9-Step Production Pipeline
```bash
python scripts/run_pipeline.py
```

### 6. Launch the Real-Time Inference API
```bash
python scripts/serve.py --host 0.0.0.0 --port 8000
```
Interactive API docs available at `http://localhost:8000/docs`.

### 7. Run Test Suite
```bash
pytest tests/ -v
```
