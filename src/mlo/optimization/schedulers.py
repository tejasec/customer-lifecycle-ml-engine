import numpy as np

class StepDecayScheduler:
    def __init__(self, initial_lr: float, step_size: int = 10, gamma: float = 0.5):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        return float(self.initial_lr * (self.gamma ** (epoch // self.step_size)))

class ExponentialDecayScheduler:
    def __init__(self, initial_lr: float, decay_rate: float = 0.05):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def get_lr(self, epoch: int) -> float:
        return float(self.initial_lr * np.exp(-self.decay_rate * epoch))

class CosineAnnealingScheduler:
    def __init__(self, initial_lr: float, total_epochs: int, min_lr: float = 1e-5):
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def get_lr(self, epoch: int) -> float:
        return float(
            self.min_lr
            + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + np.cos(np.pi * epoch / self.total_epochs))
        )

class WarmupScheduler:
    def __init__(self, target_lr: float, warmup_epochs: int = 5):
        self.target_lr = target_lr
        self.warmup_epochs = max(1, warmup_epochs)

    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return float(self.target_lr * ((epoch + 1) / self.warmup_epochs))
        return float(self.target_lr)

class CyclicalLRScheduler:
    def __init__(self, base_lr: float = 1e-4, max_lr: float = 1e-2, step_size: int = 10):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size

    def get_lr(self, epoch: int) -> float:
        cycle = np.floor(1 + epoch / (2 * self.step_size))
        x = np.abs(epoch / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        return float(lr)
