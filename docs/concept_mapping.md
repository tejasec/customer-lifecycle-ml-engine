# Concept to Codebase Mapping Matrix

This document maps every concept, mathematical formulation, and algorithm in **Machine Learning & Optimization** to its implementation in this repository.

---

## 📌 Module Mapping Table

| Module / Topic | Concepts & Math | Code Implementation Location | CLI / Script Runner | Notebook Demo |
| :--- | :--- | :--- | :--- | :--- |
| **Supervised Learning** | Categorical Encoding (Label & One-Hot) | `src/mlo/data/encoder.py` | `python scripts/run_phase.py --phase 1` | `notebooks/01_phase1_supervised_predictors.ipynb` |
| | Linear & Polynomial Regression ($y = b_0 + \sum b_i x_i$) | `src/mlo/supervised/regression.py` | | |
| | Regularization: Ridge ($L_2$) & Lasso ($L_1$) | `src/mlo/supervised/regression.py` | | |
| | Classification: Logistic Regression ($\sigma(z)$), KNN, SVM (Linear, Poly, RBF) | `src/mlo/supervised/classification.py` | | |
| | Decision Trees (Gini $1 - \sum p_i^2$, Entropy $-\sum p_i \log_2 p_i$, Pruning) | `src/mlo/supervised/trees.py` | | |
| **Unsupervised Learning** | K-Means & Agglomerative Hierarchical Clustering | `src/mlo/unsupervised/clustering.py` | `python scripts/run_phase.py --phase 2` | `notebooks/02_phase2_unsupervised_discovery.ipynb` |
| | Dendrogram Plotting & Ward Linkage | `src/mlo/unsupervised/clustering.py` | | |
| | DBSCAN Density Clustering ($\epsilon, \text{MinPts}$) & Outlier / Fraud Detection | `src/mlo/unsupervised/density.py` | | |
| | Dimensionality Reduction: PCA & t-SNE | `src/mlo/unsupervised/dim_reduction.py` | | |
| **Evaluation & Validation** | Metrics: MSE, RMSE, Confusion Matrix, Accuracy, Precision, Recall, F1 | `src/mlo/evaluation/metrics.py` | `python scripts/run_phase.py --phase 3` | `notebooks/03_phase3_validation_and_tuning.ipynb` |
| | Cross-Validation: K-Fold & Stratified K-Fold | `src/mlo/evaluation/validation.py` | | |
| | Bias-Variance Diagnosis & Learning Curves | `src/mlo/evaluation/diagnostics.py` | | |
| | Hyperparameter Tuning: Grid Search CV | `src/mlo/evaluation/tuning.py` | | |
| **Optimization in ML** | Scratch Loss Functions (MSE, Binary Cross-Entropy / Log Loss) & Gradients | `src/mlo/optimization/losses.py` | `python scripts/run_phase.py --phase 4` | `notebooks/04_phase4_custom_optimizers.ipynb` |
| | Gradient Descent Variants: BGD, SGD, Mini-Batch GD | `src/mlo/optimization/optimizers.py` | | |
| | Adaptive Optimizers: Adagrad, RMSprop, Adam ($\beta_1=0.9, \beta_2=0.999$) | `src/mlo/optimization/optimizers.py` | | |
| | Learning Rate Schedules: Step, Exponential, Cosine, Warmup, Cyclical | `src/mlo/optimization/schedulers.py` | | |
| | Custom Engine Loop & Convergence Comparison | `src/mlo/optimization/engine.py` | | |
| **Ensemble Learning & Production** | Bagging: Random Forest with Bootstrap Sampling | `src/mlo/ensemble/bagging.py` | `python scripts/run_phase.py --phase 5` | `notebooks/05_phase5_ensemble_production_pipeline.ipynb` |
| | Boosting: AdaBoost, Gradient Boosting, XGBoost | `src/mlo/ensemble/boosting.py` | | |
| | Stacking: Two-Level Stacking Classifier & Meta-Learner | `src/mlo/ensemble/stacking.py` | | |
| | 9-Step Production Lifecycle Orchestration | `src/mlo/pipeline/lifecycle.py` | `python scripts/run_pipeline.py` | |
| | Real-Time Serving API (FastAPI & Pydantic) | `src/mlo/pipeline/serving.py` | `python scripts/serve.py` | |
| | Data Drift Monitoring & Retraining Triggers | `src/mlo/pipeline/monitoring.py` | | |
