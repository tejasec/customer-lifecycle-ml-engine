import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from mlo.config import ROOT_DIR
from mlo.data.generator import generate_customer_dataset
from mlo.data.encoder import CategoricalEncoder
from mlo.data.preprocessor import DataPreprocessor
from mlo.supervised.regression import train_regression_suite
from mlo.supervised.classification import train_classification_suite
from mlo.supervised.trees import compare_tree_architectures
from mlo.unsupervised.clustering import CustomerKMeans, CustomerHierarchicalClustering
from mlo.unsupervised.density import CustomerDBSCAN
from mlo.unsupervised.dim_reduction import CustomerPCA, CustomerTSNE
from mlo.evaluation.metrics import calculate_regression_metrics, calculate_classification_metrics
from mlo.evaluation.validation import perform_stratified_cv, perform_cross_validation
from mlo.evaluation.diagnostics import diagnose_bias_variance
from mlo.evaluation.tuning import tune_hyperparameters
from mlo.optimization.optimizers import (
    BatchGradientDescent,
    StochasticGradientDescent,
    MiniBatchGradientDescent,
    AdagradOptimizer,
    RMSpropOptimizer,
    AdamOptimizer
)
from mlo.optimization.schedulers import CosineAnnealingScheduler
from mlo.optimization.engine import OptimizationEngine
from mlo.ensemble.bagging import CustomerRandomForest
from mlo.ensemble.boosting import CustomerAdaBoost, CustomerGradientBoosting, CustomerXGBoost
from mlo.ensemble.stacking import CustomerStackingClassifier
from mlo.utils.visualization import plot_loss_curves, plot_cluster_scatter

def run_phase_1():
    print("=" * 80)
    print("PHASE 1: SUPERVISED PREDICTORS")
    print("=" * 80)
    df = generate_customer_dataset(n_samples=500, random_state=42)
    encoder = CategoricalEncoder()
    df_encoded = encoder.fit_transform(df)
    features = df_encoded[encoder.feature_columns_]

    # 1. Regression (LTV)
    preproc = DataPreprocessor(scaler_type="standard")
    X_train_r, X_test_r, y_train_r, y_test_r = preproc.split(features, df_encoded['LTV'], test_size=0.2, random_state=42)
    reg_results = train_regression_suite(X_train_r.values, y_train_r.values, X_test_r.values, y_test_r.values)
    print("\n--- Regression Suite (LTV Target) ---")
    for name, res in reg_results.items():
        print(f"{name:25s} | MSE: {res['mse']:10.2f} | R2: {res['r2']:.4f}")

    # 2. Classification (Churn)
    X_train_c, X_test_c, y_train_c, y_test_c = preproc.split(features, df_encoded['Churn'], test_size=0.2, random_state=42, stratify=df_encoded['Churn'])
    X_tr_c_sc = preproc.fit_transform(X_train_c.values)
    X_te_c_sc = preproc.transform(X_test_c.values)

    suite, cls_results = train_classification_suite(X_tr_c_sc, y_train_c.values, X_te_c_sc, y_test_c.values)
    print("\n--- Classification Suite (Churn Target) ---")
    print(cls_results.to_string(index=False))

    # 3. Decision Trees
    tree_results = compare_tree_architectures(X_tr_c_sc, y_train_c.values, X_te_c_sc, y_test_c.values, feature_names=encoder.feature_columns_)
    print("\n--- Decision Tree Architecture & Pruning ---")
    print(f"Unpruned Gini Tree Test Accuracy: {tree_results['unpruned_gini']['test_acc']:.4f} (Depth: {tree_results['unpruned_gini']['depth']})")
    print(f"Pre-pruned Entropy Tree Test Accuracy: {tree_results['pre_pruned_entropy']['test_acc']:.4f} (Depth: {tree_results['pre_pruned_entropy']['depth']})")
    print("Pre-pruned Tree Rules:\n", tree_results['pre_pruned_entropy']['rules'])

def run_phase_2():
    print("=" * 80)
    print("PHASE 2: DISCOVERY ENGINE")
    print("=" * 80)
    df = generate_customer_dataset(n_samples=500, random_state=42, anomaly_ratio=0.06)
    encoder = CategoricalEncoder()
    df_encoded = encoder.fit_transform(df)
    X = df_encoded[encoder.feature_columns_].values
    preproc = DataPreprocessor(scaler_type="standard")
    X_scaled = preproc.fit_transform(X)

    # 1. K-Means
    kmeans = CustomerKMeans(n_clusters=3, random_state=42).fit(X_scaled)
    labels_km = kmeans.predict(X_scaled)
    print(f"K-Means (k=3) Fitted | Inertia: {kmeans.inertia:.2f} | Centroids Shape: {kmeans.centroids.shape}")

    # 2. Hierarchical
    hier = CustomerHierarchicalClustering(n_clusters=3, linkage_method="ward")
    labels_hier = hier.fit_predict(X_scaled)
    dendro_path = ROOT_DIR / "artifacts" / "plots" / "hierarchical_dendrogram.png"
    hier.plot_dendrogram(X_scaled[:50], max_d=7.0, save_path=dendro_path)
    print(f"Hierarchical Clustering Fitted | Dendrogram saved to: {dendro_path}")

    # 3. DBSCAN
    dbscan = CustomerDBSCAN(eps=1.2, min_samples=4)
    labels_db = dbscan.fit_predict(X_scaled)
    db_stats = dbscan.get_cluster_stats()
    print(f"DBSCAN Density Clustering | Clusters: {db_stats['n_clusters']} | Noise/Anomalies Detected: {db_stats['n_noise_points']} ({db_stats['noise_ratio']*100:.1f}%)")

    # 4. PCA & t-SNE
    pca = CustomerPCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA 2D Projection | Variance Explained: {pca.total_variance_explained*100:.2f}% (Ratios: {pca.explained_variance_ratio})")

    tsne = CustomerTSNE(n_components=2, perplexity=20.0, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_plot_path = ROOT_DIR / "artifacts" / "plots" / "tsne_clusters.png"
    plot_cluster_scatter(X_tsne[:, 0], X_tsne[:, 1], labels_km, title="t-SNE Customer Clusters (K-Means Labels)", save_path=tsne_plot_path)
    print(f"t-SNE 2D Manifold Scatter saved to: {tsne_plot_path}")

def run_phase_3():
    print("=" * 80)
    print("PHASE 3: VALIDATION FRAMEWORK")
    print("=" * 80)
    df = generate_customer_dataset(n_samples=500, random_state=42)
    encoder = CategoricalEncoder()
    df_encoded = encoder.fit_transform(df)
    X = df_encoded[encoder.feature_columns_].values
    y = df_encoded['Churn'].values
    preproc = DataPreprocessor(scaler_type="standard")
    X_scaled = preproc.fit_transform(X)

    # 1. Stratified K-Fold
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(random_state=42)
    cv_res = perform_stratified_cv(lr, X_scaled, y, n_splits=5, scoring="f1")
    print(f"Stratified 5-Fold CV (Logistic Regression F1): Mean = {cv_res['mean_score']:.4f} (+/- {cv_res['std_score']:.4f})")
    print(f"Fold Scores: {[round(s, 4) for s in cv_res['fold_scores']]}")

    # 2. Bias-Variance Diagnosis
    lr.fit(X_scaled[:400], y[:400])
    train_score = float(lr.score(X_scaled[:400], y[:400]))
    test_score = float(lr.score(X_scaled[400:], y[400:]))
    diagnosis = diagnose_bias_variance(train_score, test_score, benchmark_score=0.80)
    print(f"\nBias-Variance Diagnosis:")
    print(f"Train Score: {train_score:.4f} | Test Score: {test_score:.4f} | Gap: {diagnosis['generalization_gap']:.4f}")
    print(f"Diagnosis: {diagnosis['diagnosis']} -> {diagnosis['recommendation']}")

    # 3. Hyperparameter Tuning
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=42)
    param_grid = {'max_depth': [2, 3, 5, 8], 'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 5, 10]}
    tuning_res = tune_hyperparameters(dt, param_grid, X_scaled, y, method="grid", cv=5, scoring="f1")
    print(f"\nGridSearchCV Best Hyperparameters: {tuning_res['best_params']} (Best F1: {tuning_res['best_score']:.4f})")

def run_phase_4():
    print("=" * 80)
    print("PHASE 4: CUSTOM OPTIMIZATION ENGINE")
    print("=" * 80)
    df = generate_customer_dataset(n_samples=600, random_state=42)
    encoder = CategoricalEncoder()
    df_encoded = encoder.fit_transform(df)
    X = df_encoded[encoder.feature_columns_].values
    y = df_encoded['Churn'].values
    preproc = DataPreprocessor(scaler_type="standard")
    X_scaled = preproc.fit_transform(X)

    epochs = 60
    optimizers = {
        "Batch GD": BatchGradientDescent(learning_rate=0.1),
        "SGD (Momentum)": StochasticGradientDescent(learning_rate=0.01, momentum=0.9),
        "Mini-Batch GD": MiniBatchGradientDescent(learning_rate=0.05, batch_size=32),
        "Adagrad": AdagradOptimizer(learning_rate=0.1),
        "RMSprop": RMSpropOptimizer(learning_rate=0.01),
        "Adam": AdamOptimizer(learning_rate=0.02)
    }

    loss_trajectories = {}
    print(f"Training custom optimizers from scratch for {epochs} epochs:")
    for name, opt in optimizers.items():
        scheduler = CosineAnnealingScheduler(initial_lr=0.05, total_epochs=epochs) if name == "Adam" else None
        engine = OptimizationEngine(optimizer=opt, loss_type="bce", epochs=epochs, scheduler=scheduler)
        engine.fit(X_scaled, y)
        loss_trajectories[name] = engine.loss_history
        preds = engine.predict(X_scaled)
        acc = np.mean(preds == y)
        print(f"{name:16s} | Final Log Loss: {engine.loss_history[-1]:.4f} | Training Acc: {acc:.4f}")

    plot_path = ROOT_DIR / "artifacts" / "plots" / "optimizer_loss_curves.png"
    plot_loss_curves(loss_trajectories, title="Custom Optimizer Convergence (Binary Cross-Entropy)", save_path=plot_path)
    print(f"\nConvergence loss plot saved to: {plot_path}")

def run_phase_5():
    print("=" * 80)
    print("PHASE 5: ENSEMBLE PRODUCTION PIPELINE")
    print("=" * 80)
    df = generate_customer_dataset(n_samples=800, random_state=42)
    encoder = CategoricalEncoder()
    df_encoded = encoder.fit_transform(df)
    X = df_encoded[encoder.feature_columns_].values
    y = df_encoded['Churn'].values

    preproc = DataPreprocessor(scaler_type="standard")
    X_train, X_test, y_train, y_test = preproc.split(pd.DataFrame(X), pd.Series(y), test_size=0.2, random_state=42, stratify=pd.Series(y))
    X_train_sc = preproc.fit_transform(X_train.values)
    X_test_sc = preproc.transform(X_test.values)

    ensembles = {
        "Random Forest (Bagging)": CustomerRandomForest(n_estimators=100, max_depth=5),
        "AdaBoost": CustomerAdaBoost(n_estimators=50),
        "Gradient Boosting": CustomerGradientBoosting(n_estimators=80, learning_rate=0.1),
        "XGBoost": CustomerXGBoost(n_estimators=80, learning_rate=0.1),
        "Stacking Meta-Learner": CustomerStackingClassifier()
    }

    records = []
    for name, model in ensembles.items():
        model.fit(X_train_sc, y_train.values)
        preds = model.predict(X_test_sc)
        metrics = calculate_classification_metrics(y_test.values, preds)
        records.append({
            "Ensemble Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1"]
        })

    print(pd.DataFrame(records).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Run MLO Project Phases (Phases 1 - 5).")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5], help="Phase number to run (1 to 5).")
    args = parser.parse_args()

    phase_funcs = {
        1: run_phase_1,
        2: run_phase_2,
        3: run_phase_3,
        4: run_phase_4,
        5: run_phase_5,
    }
    phase_funcs[args.phase]()

if __name__ == "__main__":
    main()
