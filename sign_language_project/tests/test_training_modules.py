"""Deterministic tests for Person 5 modules using a tiny test-only model."""
import tempfile
from pathlib import Path
import numpy as np

from training.loss import cross_entropy_loss, cross_entropy_gradient
from training.optimizer import SGD
from training.backward import backward
from training.train import train, evaluate
from training.checkpoints import save_checkpoint, load_checkpoint
from training.schedule import StepDecay, ExponentialDecay, CosineDecay


class TinyDense:
    """Test-only single affine layer; not part of the production CNN."""
    def __init__(self, input_dim=2, classes=2):
        self.W = np.array([[0.2, -0.1], [-0.3, 0.4]], dtype=float)
        self.b = np.zeros(classes, dtype=float)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward_logits(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW[...] = self.x.T @ grad
        self.db[...] = grad.sum(axis=0)
        return grad @ self.W.T


class TinyModel:
    """Test fixture with the interface expected by Person 5."""
    def __init__(self):
        self.cfg = type('Config', (), {'num_classes': 2})()
        self.dense = TinyDense()
        self.layers = [self.dense]

    def forward(self, X):
        logits = self.dense.forward_logits(np.asarray(X).reshape(len(X), -1))
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def parameters(self):
        return [(self.dense.W, self.dense.dW), (self.dense.b, self.dense.db)]


def test_loss_and_gradient():
    p = np.array([[0.8, 0.2], [0.25, 0.75]])
    y = np.array([0, 1], dtype=np.int64)
    expected = -(np.log(0.8) + np.log(0.75)) / 2
    np.testing.assert_allclose(cross_entropy_loss(p, y), expected)
    grad = cross_entropy_gradient(p, y)
    np.testing.assert_allclose(grad, [[-0.1, 0.1], [0.125, -0.125]])
    assert grad.shape == p.shape
    try:
        cross_entropy_loss(p, np.array([0, 2]))
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid label was accepted')


def test_sgd_update():
    model = TinyModel()
    model.dense.dW[:] = 1
    model.dense.db[:] = 2
    old_w, old_b = model.dense.W.copy(), model.dense.b.copy()
    SGD(0.1).step(model)
    np.testing.assert_allclose(model.dense.W, old_w - 0.1)
    np.testing.assert_allclose(model.dense.b, old_b - 0.2)


def test_backward_shapes():
    model = TinyModel()
    X = np.array([[1., 0.], [0., 1.]])
    y = np.array([0, 1])
    p = model.forward(X)
    dx = backward(model, p, y)
    assert dx.shape == X.shape
    assert np.isfinite(dx).all()


def test_training_and_evaluation():
    model = TinyModel()
    X = np.array([[[[1.]], [[0.]]], [[[0.]], [[1.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    batches = [(X, y)]
    initial = evaluate(model, batches)['loss']
    history = train(model, batches, val_batches=batches, epochs=8, learning_rate=0.1, verbose=False)
    final = evaluate(model, batches)['loss']
    assert len(history['train_loss']) == 8
    assert len(history['val_accuracy']) == 8
    assert np.isfinite(history['train_loss']).all()
    assert final < initial


def test_checkpoint_round_trip():
    model = TinyModel()
    X = np.array([[1., 0.], [0., 1.]])
    before = model.forward(X)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'nested' / 'model.npz'
        save_checkpoint(model, path, epoch=3, history={'train_loss': [0.5]})
        restored = TinyModel()
        metadata = load_checkpoint(restored, path)
        np.testing.assert_allclose(restored.forward(X), before)
        assert metadata['epoch'] == 3
        assert metadata['history']['train_loss'] == [0.5]


def test_lr_schedules():
    s = StepDecay(0.1, drop=0.5, every=2)
    np.testing.assert_allclose([s(e) for e in range(5)], [0.1, 0.1, 0.05, 0.05, 0.025])
    assert abs(ExponentialDecay(0.1, 0.5)(3) - 0.0125) < 1e-12
    c = CosineDecay(0.1, total_epochs=10, min_lr=0.001)
    assert abs(c(0) - 0.1) < 1e-12 and abs(c(10) - 0.001) < 1e-12 and abs(c(50) - 0.001) < 1e-12
    assert c(0) > c(3) > c(7) > c(10)
    for bad in (lambda: StepDecay(-1), lambda: StepDecay(0.1, drop=1.5), lambda: CosineDecay(0.1, 0)):
        try: bad(); assert False, "should raise"
        except ValueError: pass


def test_training_with_schedule():
    model = TinyModel()
    X = np.array([[[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    y = np.array([0, 1], dtype=np.int64)
    opt = SGD(0.5)
    history = train(model, [(X, y)], epochs=4, optimizer=opt, lr_schedule=StepDecay(0.4, 0.5, 2), verbose=False)
    np.testing.assert_allclose(history['learning_rate'], [0.4, 0.4, 0.2, 0.2])
    assert opt.learning_rate == 0.2          # schedule overrides the optimizer's own rate
    try: train(model, [(X, y)], epochs=1, lr_schedule=lambda e: -1.0, verbose=False); assert False
    except ValueError: pass


if __name__ == '__main__':
    tests = [value for name, value in globals().copy().items() if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print(f'{len(tests)} tests passed')
