import numpy as np
from mlo.optimization.losses import MSELoss, BinaryCrossEntropyLoss
from mlo.optimization.optimizers import (
    BatchGradientDescent,
    StochasticGradientDescent,
    MiniBatchGradientDescent,
    AdagradOptimizer,
    RMSpropOptimizer,
    AdamOptimizer
)
from mlo.optimization.schedulers import (
    StepDecayScheduler,
    ExponentialDecayScheduler,
    CosineAnnealingScheduler,
    WarmupScheduler
)
from mlo.optimization.engine import OptimizationEngine

def test_loss_gradients():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([5.0, 11.0])
    y_pred = np.array([4.0, 10.0])

    # MSE backward
    grad_W, grad_b = MSELoss.backward(X, y, y_pred)
    assert grad_W.shape == (2,)
    assert isinstance(grad_b, float)

    # BCE backward
    y_cls = np.array([0.0, 1.0])
    prob = np.array([0.2, 0.8])
    bce = BinaryCrossEntropyLoss()
    loss_val = bce.forward(y_cls, prob)
    assert loss_val > 0
    grad_W_bce, grad_b_bce = bce.backward(X, y_cls, prob)
    assert grad_W_bce.shape == (2,)

def test_optimizers_step():
    W = np.array([1.0, 2.0])
    b = 0.5
    grad_W = np.array([0.1, -0.2])
    grad_b = 0.05

    opts = [
        BatchGradientDescent(learning_rate=0.1),
        StochasticGradientDescent(learning_rate=0.1),
        MiniBatchGradientDescent(learning_rate=0.1),
        AdagradOptimizer(learning_rate=0.1),
        RMSpropOptimizer(learning_rate=0.1),
        AdamOptimizer(learning_rate=0.1),
    ]

    for opt in opts:
        new_W, new_b = opt.step(W, b, grad_W, grad_b)
        assert new_W.shape == W.shape
        assert isinstance(new_b, float)
        assert not np.isnan(new_W).any()

def test_schedulers():
    step_sch = StepDecayScheduler(initial_lr=0.1, step_size=5, gamma=0.5)
    assert step_sch.get_lr(0) == 0.1
    assert step_sch.get_lr(5) == 0.05

    cosine_sch = CosineAnnealingScheduler(initial_lr=0.1, total_epochs=10, min_lr=0.01)
    assert abs(cosine_sch.get_lr(0) - 0.1) < 1e-4
    assert abs(cosine_sch.get_lr(10) - 0.01) < 1e-4

    warm_sch = WarmupScheduler(target_lr=0.1, warmup_epochs=5)
    assert warm_sch.get_lr(0) < 0.1
    assert warm_sch.get_lr(5) == 0.1

def test_optimization_engine_convergence(sample_arrays):
    X, _, y_cls = sample_arrays
    opt = AdamOptimizer(learning_rate=0.05)
    engine = OptimizationEngine(optimizer=opt, loss_type="bce", epochs=40)
    engine.fit(X, y_cls)

    assert len(engine.loss_history) == 40
    # Confirm optimization successfully minimized the objective
    assert engine.loss_history[-1] < engine.loss_history[0]
    preds = engine.predict(X)
    assert len(preds) == len(y_cls)
