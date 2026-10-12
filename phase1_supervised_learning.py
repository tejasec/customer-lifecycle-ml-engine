"""
================================================================================
Phase 1: Supervised Learning — Customer Lifecycle & Risk Prediction Engine
================================================================================

This module implements Phase 1 of the complete Machine Learning & Optimization project:
1. Data Preprocessing & Categorical Encoding (Label vs. One-Hot Encoding)
2. Linear & Polynomial Regression for LTV Forecasting
3. Regularization: Ridge (L2) vs. Lasso (L1) Feature Selection
4. Classification Suite: Logistic Regression, KNN (Euclidean Distance), SVM (Linear, RBF)
5. Decision Tree Architecture: Gini Index vs. Entropy Splitting & Pre-Pruning
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

def main():
    print("=" * 80)
    print("STEP 1: SYNTHETIC DATASET GENERATION")
    print("=" * 80)
    np.random.seed(42)
    n_samples = 500

    age = np.random.randint(18, 70, size=n_samples)
    monthly_spend = np.random.uniform(20.0, 200.0, size=n_samples)
    account_age_months = np.random.randint(1, 60, size=n_samples)
    support_calls = np.random.randint(0, 10, size=n_samples)
    plan_types = np.random.choice(['Basic', 'Standard', 'Premium'], size=n_samples, p=[0.5, 0.3, 0.2])
    gender = np.random.choice(['Male', 'Female'], size=n_samples)

    # Customer LTV Target
    ltv = (monthly_spend * account_age_months * 0.85) + (age * 12.5) - (support_calls * 45) + np.random.normal(0, 50, n_samples)
    
    # Churn Binary Target
    logit = -1.5 + (0.4 * support_calls) - (0.05 * account_age_months) + (0.01 * monthly_spend) - (0.02 * age)
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (churn_prob > 0.5).astype(int)

    df = pd.DataFrame({
        'Age': age,
        'MonthlySpend': monthly_spend,
        'AccountAgeMonths': account_age_months,
        'SupportCalls': support_calls,
        'PlanType': plan_types,
        'Gender': gender,
        'LTV': ltv,
        'Churn': churn
    })
    print("Dataset Shape:", df.shape)
    print(df.head(), "\n")

    print("=" * 80)
    print("STEP 2: CATEGORICAL FEATURE ENCODING")
    print("=" * 80)
    df_encoded = df.copy()
    plan_mapping = {'Basic': 0, 'Standard': 1, 'Premium': 2}
    df_encoded['PlanType_Encoded'] = df_encoded['PlanType'].map(plan_mapping)
    df_encoded = pd.get_dummies(df_encoded, columns=['Gender'], drop_first=True, dtype=int)
    
    features_df = df_encoded.drop(columns=['PlanType', 'LTV', 'Churn'])
    print("Encoded Feature Set Columns:", list(features_df.columns))
    print(features_df.head(), "\n")

    print("=" * 80)
    print("STEP 3: LINEAR & POLYNOMIAL REGRESSION FOR LTV PREDICTION")
    print("=" * 80)
    X_reg = features_df
    y_reg = df_encoded['LTV']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    lin_reg = LinearRegression().fit(X_train_r, y_train_r)
    y_pred_lin = lin_reg.predict(X_test_r)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_r)
    X_test_poly = poly.transform(X_test_r)

    poly_reg = LinearRegression().fit(X_train_poly, y_train_r)
    y_pred_poly = poly_reg.predict(X_test_poly)

    print(f"Linear Regression MSE: {mean_squared_error(y_test_r, y_pred_lin):.2f} | R2: {r2_score(y_test_r, y_pred_lin):.4f}")
    print(f"Polynomial Regression MSE: {mean_squared_error(y_test_r, y_pred_poly):.2f} | R2: {r2_score(y_test_r, y_pred_poly):.4f}\n")

    print("=" * 80)
    print("STEP 4: REGULARIZATION — RIDGE (L2) VS. LASSO (L1)")
    print("=" * 80)
    scaler = StandardScaler()
    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler.transform(X_test_poly)

    ridge = Ridge(alpha=10.0).fit(X_train_poly_scaled, y_train_r)
    y_pred_ridge = ridge.predict(X_test_poly_scaled)

    lasso = Lasso(alpha=5.0).fit(X_train_poly_scaled, y_train_r)
    y_pred_lasso = lasso.predict(X_test_poly_scaled)

    print(f"Ridge (L2) MSE: {mean_squared_error(y_test_r, y_pred_ridge):.2f} | R2: {r2_score(y_test_r, y_pred_ridge):.4f}")
    print(f"Lasso (L1) MSE: {mean_squared_error(y_test_r, y_pred_lasso):.2f} | R2: {r2_score(y_test_r, y_pred_lasso):.4f}")
    print(f"Lasso Zeroed Features Count: {np.sum(lasso.coef_ == 0)} / {len(lasso.coef_)}\n")

    print("=" * 80)
    print("STEP 5: CLASSIFICATION SUITE FOR CHURN PREDICTION")
    print("=" * 80)
    X_cls = features_df
    y_cls = df_encoded['Churn']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    scaler_cls = StandardScaler()
    X_train_c_scaled = scaler_cls.fit_transform(X_train_c)
    X_test_c_scaled = scaler_cls.transform(X_test_c)

    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'KNN (K=5, Euclidean)': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        'SVM (Linear Kernel)': SVC(kernel='linear'),
        'SVM (RBF Kernel)': SVC(kernel='rbf')
    }

    results = []
    for name, clf in classifiers.items():
        clf.fit(X_train_c_scaled, y_train_c)
        preds = clf.predict(X_test_c_scaled)
        results.append({
            'Classifier': name,
            'Accuracy': accuracy_score(y_test_c, preds),
            'Precision': precision_score(y_test_c, preds),
            'Recall': recall_score(y_test_c, preds),
            'F1-Score': f1_score(y_test_c, preds)
        })
    print(pd.DataFrame(results).to_string(index=False), "\n")

    print("=" * 80)
    print("STEP 6: DECISION TREE ARCHITECTURE & PRE-PRUNING")
    print("=" * 80)
    dt_unpruned = DecisionTreeClassifier(criterion='gini', random_state=42).fit(X_train_c, y_train_c)
    acc_unpruned = dt_unpruned.score(X_test_c, y_test_c)

    dt_pruned = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=5, random_state=42).fit(X_train_c, y_train_c)
    acc_pruned = dt_pruned.score(X_test_c, y_test_c)

    print(f"Unpruned Gini Tree Accuracy: {acc_unpruned:.4f} (Depth: {dt_unpruned.get_depth()})")
    print(f"Pre-pruned Entropy Tree Accuracy: {acc_pruned:.4f} (Depth: {dt_pruned.get_depth()})\n")
    print("--- Pre-Pruned Tree Rules ---")
    print(export_text(dt_pruned, feature_names=list(features_df.columns)))

if __name__ == '__main__':
    main()
