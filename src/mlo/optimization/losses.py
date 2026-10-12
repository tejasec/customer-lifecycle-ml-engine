from typing import Tuple
import numpy as np

class MSELoss:
    """
    Mean Squared Error (MSE) Loss function:
    L(y_true, y_pred) = (1 / (2 * m)) * sum((y_pred - y_true)^2)
    """
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = len(y_true)
        return float(np.mean((y_pred - y_true) ** 2) / 2.0)

    @staticmethod
    def backward(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes analytical gradients w.r.t weights and bias.
        grad_W = (1 / m) * X.T @ (y_pred - y_true)
        grad_b = (1 / m) * sum(y_pred - y_true)
        """
        m = len(y_true)
        diff = y_pred - y_true
        grad_W = (1.0 / m) * np.dot(X.T, diff)
        grad_b = float(np.mean(diff))
        return grad_W, grad_b

class BinaryCrossEntropyLoss:
    """
    Binary Cross-Entropy (Log Loss) function:
    L(y_true, y_pred) = - (1 / m) * sum(y * log(p + eps) + (1 - y) * log(1 - p + eps))
    """
    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def forward(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        p = np.clip(y_pred_prob, self.eps, 1.0 - self.eps)
        loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        return float(loss)

    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred_prob: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Analytical gradients for sigmoid activated logistic regression.
        grad_W = (1 / m) * X.T @ (p - y)
        grad_b = (1 / m) * sum(p - y)
        """
        m = len(y_true)
        diff = y_pred_prob - y_true
        grad_W = (1.0 / m) * np.dot(X.T, diff)
        grad_b = float(np.mean(diff))
        return grad_W, grad_b
