from typing import Dict, Any, List, Optional
import numpy as np
from .losses import MSELoss, BinaryCrossEntropyLoss
from .optimizers import (
    BaseOptimizer,
    BatchGradientDescent,
    StochasticGradientDescent,
    MiniBatchGradientDescent,
    AdagradOptimizer,
    RMSpropOptimizer,
    AdamOptimizer
)

class OptimizationEngine:
    """
    Executes iterative parameter optimization across custom GD algorithms.
    Supports continuous regression (MSE) and binary classification (BCE).
    """
    def __init__(
        self,
        optimizer: BaseOptimizer,
        loss_type: str = "mse",
        epochs: int = 100,
        batch_size: int = 32,
        scheduler: Optional[Any] = None,
        random_state: int = 42
    ):
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.random_state = random_state

        self.W: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OptimizationEngine":
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        self.W = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0
        self.loss_history = []

        loss_fn = MSELoss() if self.loss_type == "mse" else BinaryCrossEntropyLoss()

        for epoch in range(self.epochs):
            lr = self.scheduler.get_lr(epoch) if self.scheduler else None

            # Mini-batch / Stochastic / Full batch handling
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            if isinstance(self.optimizer, StochasticGradientDescent):
                # Update per single observation
                for idx in indices:
                    x_i = X[idx:idx+1]
                    y_i = y[idx:idx+1]
                    linear_output = np.dot(x_i, self.W) + self.b
                    pred_i = self._sigmoid(linear_output) if self.loss_type == "bce" else linear_output
                    grad_W, grad_b = loss_fn.backward(x_i, y_i, pred_i)
                    self.W, self.b = self.optimizer.step(self.W, self.b, grad_W, grad_b, current_lr=lr)
            elif isinstance(self.optimizer, MiniBatchGradientDescent):
                for start_idx in range(0, n_samples, self.batch_size):
                    batch_idx = indices[start_idx : start_idx + self.batch_size]
                    X_b = X[batch_idx]
                    y_b = y[batch_idx]
                    linear_output = np.dot(X_b, self.W) + self.b
                    pred_b = self._sigmoid(linear_output) if self.loss_type == "bce" else linear_output
                    grad_W, grad_b = loss_fn.backward(X_b, y_b, pred_b)
                    self.W, self.b = self.optimizer.step(self.W, self.b, grad_W, grad_b, current_lr=lr)
            else:
                # Full Batch or Adaptive (BGD, Adagrad, RMSprop, Adam)
                linear_output = np.dot(X, self.W) + self.b
                preds = self._sigmoid(linear_output) if self.loss_type == "bce" else linear_output
                grad_W, grad_b = loss_fn.backward(X, y, preds)
                self.W, self.b = self.optimizer.step(self.W, self.b, grad_W, grad_b, current_lr=lr)

            # Record epoch loss
            full_preds = self._sigmoid(np.dot(X, self.W) + self.b) if self.loss_type == "bce" else (np.dot(X, self.W) + self.b)
            epoch_loss = loss_fn.forward(y, full_preds)
            self.loss_history.append(float(epoch_loss))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise ValueError("Model is not fitted yet.")
        linear_output = np.dot(X, self.W) + self.b
        if self.loss_type == "bce":
            probs = self._sigmoid(linear_output)
            return (probs > 0.5).astype(int)
        return linear_output

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W is None or self.loss_type != "bce":
            raise ValueError("Predict proba is only available for classification.")
        return self._sigmoid(np.dot(X, self.W) + self.b)
