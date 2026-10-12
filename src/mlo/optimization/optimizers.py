from typing import Tuple, Dict, Any, Optional
import numpy as np

class BaseOptimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def step(
        self,
        W: np.ndarray,
        b: float,
        grad_W: np.ndarray,
        grad_b: float,
        current_lr: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

class BatchGradientDescent(BaseOptimizer):
    """
    Standard Batch Gradient Descent:
    theta_new = theta_old - alpha * grad
    """
    def step(
        self,
        W: np.ndarray,
        b: float,
        grad_W: np.ndarray,
        grad_b: float,
        current_lr: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        lr = current_lr if current_lr is not None else self.lr
        new_W = W - lr * grad_W
        new_b = b - lr * grad_b
        return new_W, new_b

class StochasticGradientDescent(BaseOptimizer):
    """
    Stochastic Gradient Descent (SGD) with optional momentum.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v_W: Optional[np.ndarray] = None
        self.v_b: float = 0.0

    def step(
        self,
        W: np.ndarray,
        b: float,
        grad_W: np.ndarray,
        grad_b: float,
        current_lr: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        lr = current_lr if current_lr is not None else self.lr
        if self.v_W is None:
            self.v_W = np.zeros_like(W)

        self.v_W = self.momentum * self.v_W + lr * grad_W
        self.v_b = self.momentum * self.v_b + lr * grad_b

        new_W = W - self.v_W
        new_b = b - self.v_b
        return new_W, new_b

class MiniBatchGradientDescent(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.01, batch_size: int = 32):
        super().__init__(learning_rate)
        self.batch_size = batch_size

    def step(
        self,
        W: np.ndarray,
        b: float,
        grad_W: np.ndarray,
        grad_b: float,
        current_lr: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        lr = current_lr if current_lr is not None else self.lr
        return W - lr * grad_W, b - lr * grad_b

class AdagradOptimizer(BaseOptimizer):
    """
    Adagrad: Adaptive gradient algorithm scaling coordinates by sum of historical squared gradients.
    """
    def __init__(self, learning_rate: float = 0.01, eps: float = 1e-8):
        super().__init__(learning_rate)
        self.eps = eps
        self.G_W: Optional[np.ndarray] = None
        self.G_b: float = 0.0

    def step(
        self,
        W: np.ndarray,
        b: float,
        grad_W: np.ndarray,
        grad_b: float,
        current_lr: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        lr = current_lr if current_lr is not None else self.lr
        if self.G_W is None:
            self.G_W = np.zeros_like(W)

        self.G_W += grad_W ** 2
        self.G_b += grad_b ** 2

        new_W = W - (lr / (np.sqrt(self.G_W) + self.eps)) * grad_W
        new_b = b - (lr / (np.sqrt(self.G_b) + self.eps)) * grad_b
        return new_W, new_b

class RMSpropOptimizer(BaseOptimizer):
    """
    RMSprop: Exponential moving average of squared gradients.
    """
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.eps = eps
        self.v_W: Optional[np.ndarray] = None
        self.v_b: float = 0.0

    def step(
        self,
        W: np.ndarray,
        b: float,
        grad_W: np.ndarray,
        grad_b: float,
        current_lr: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        lr = current_lr if current_lr is not None else self.lr
        if self.v_W is None:
            self.v_W = np.zeros_like(W)

        self.v_W = self.beta * self.v_W + (1.0 - self.beta) * (grad_W ** 2)
        self.v_b = self.beta * self.v_b + (1.0 - self.beta) * (grad_b ** 2)

        new_W = W - (lr / (np.sqrt(self.v_W) + self.eps)) * grad_W
        new_b = b - (lr / (np.sqrt(self.v_b) + self.eps)) * grad_b
        return new_W, new_b

class AdamOptimizer(BaseOptimizer):
    """
    Adam: Adaptive Moment Estimation combining momentum and squared gradient scaling with bias correction.
    """
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_W: Optional[np.ndarray] = None
        self.m_b: float = 0.0
        self.v_W: Optional[np.ndarray] = None
        self.v_b: float = 0.0
        self.t: int = 0

    def step(
        self,
        W: np.ndarray,
        b: float,
        grad_W: np.ndarray,
        grad_b: float,
        current_lr: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        lr = current_lr if current_lr is not None else self.lr
        self.t += 1
        if self.m_W is None:
            self.m_W = np.zeros_like(W)
            self.v_W = np.zeros_like(W)

        # 1st moment
        self.m_W = self.beta1 * self.m_W + (1.0 - self.beta1) * grad_W
        self.m_b = self.beta1 * self.m_b + (1.0 - self.beta1) * grad_b

        # 2nd moment
        self.v_W = self.beta2 * self.v_W + (1.0 - self.beta2) * (grad_W ** 2)
        self.v_b = self.beta2 * self.v_b + (1.0 - self.beta2) * (grad_b ** 2)

        # Bias corrections
        m_W_hat = self.m_W / (1.0 - self.beta1 ** self.t)
        m_b_hat = self.m_b / (1.0 - self.beta1 ** self.t)
        v_W_hat = self.v_W / (1.0 - self.beta2 ** self.t)
        v_b_hat = self.v_b / (1.0 - self.beta2 ** self.t)

        new_W = W - (lr / (np.sqrt(v_W_hat) + self.eps)) * m_W_hat
        new_b = b - (lr / (np.sqrt(v_b_hat) + self.eps)) * m_b_hat
        return new_W, new_b
