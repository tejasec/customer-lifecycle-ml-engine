from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassificationSuite:
    """
    Trains and compares:
    - Logistic Regression
    - K-Nearest Neighbors (Euclidean Distance)
    - Support Vector Machine (Linear, Poly, RBF kernels)
    """
    def __init__(
        self,
        knn_neighbors: int = 5,
        svm_c: float = 1.0,
        random_state: int = 42
    ):
        self.models: Dict[str, Any] = {
            'Logistic Regression': LogisticRegression(random_state=random_state),
            'KNN (K=5, Euclidean)': KNeighborsClassifier(n_neighbors=knn_neighbors, metric='euclidean'),
            'SVM (Linear Kernel)': SVC(kernel='linear', C=svm_c, probability=True, random_state=random_state),
            'SVM (Polynomial Kernel)': SVC(kernel='poly', degree=3, C=svm_c, probability=True, random_state=random_state),
            'SVM (RBF Kernel)': SVC(kernel='rbf', C=svm_c, gamma='scale', probability=True, random_state=random_state),
        }
        self.fitted_models: Dict[str, Any] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for name, model in self.fitted_models.items():
            preds = model.predict(X_test)
            records.append({
                'Classifier': name,
                'Accuracy': float(accuracy_score(y_test, preds)),
                'Precision': float(precision_score(y_test, preds, zero_division=0)),
                'Recall': float(recall_score(y_test, preds, zero_division=0)),
                'F1-Score': float(f1_score(y_test, preds, zero_division=0))
            })
        return pd.DataFrame(records)

def train_classification_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    knn_neighbors: int = 5,
    svm_c: float = 1.0
) -> Tuple[ClassificationSuite, pd.DataFrame]:
    suite = ClassificationSuite(knn_neighbors=knn_neighbors, svm_c=svm_c)
    suite.fit(X_train, y_train)
    results_df = suite.evaluate(X_test, y_test)
    return suite, results_df
