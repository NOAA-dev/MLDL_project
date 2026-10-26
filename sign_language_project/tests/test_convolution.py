"""Tests for Person 2 (Conv2D, NHWC layout). Run from inside the person2 folder:

    python test_convolution.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn.convolution import Conv2D


# ---------- helpers ----------

def naive_conv_forward(x, W, b, stride, padding):
    """Slow reference implementation with plain loops, used only to check Conv2D."""
    n, height, width, c = x.shape
    k, _, _, f = W.shape
    x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    out_h = (height + 2 * padding - k) // stride + 1
    out_w = (width + 2 * padding - k) // stride + 1
    out = np.zeros((n, out_h, out_w, f))
    for sample in range(n):
        for r in range(out_h):
            for col in range(out_w):
                patch = x_pad[sample, r * stride:r * stride + k, col * stride:col * stride + k, :]
                for filt in range(f):
                    out[sample, r, col, filt] = np.sum(patch * W[:, :, :, filt]) + b[filt]
    return out


def relative_error(a, b):
    return np.abs(a - b).max() / max(1e-12, np.abs(a).max() + np.abs(b).max())


def numerical_gradient(loss_fn, array, eps=1e-5):
    """Central difference (f(v+eps) - f(v-eps)) / (2 eps) for every entry of array."""
    grad = np.zeros_like(array)
    flat = array.reshape(-1)
    flat_grad = grad.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        loss_plus = loss_fn()
        flat[i] = original - eps
        loss_minus = loss_fn()
        flat[i] = original
        flat_grad[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def expect_error(error_type, function, *args):
    try:
        function(*args)
    except error_type:
        return
    raise AssertionError(f"{function} should raise {error_type.__name__}")


CONFIGS = [
    # (in_ch, out_ch, k, stride, padding, H, W)
    (1, 1, 3, 1, 0, 6, 6),
    (2, 3, 3, 1, 1, 5, 5),
    (2, 3, 3, 2, 0, 7, 7),
    (3, 2, 2, 2, 1, 6, 5),
    (1, 4, 5, 1, 2, 8, 8),
    (2, 2, 3, 2, 1, 8, 8),
    (2, 2, 3, 3, 0, 8, 8),   # kernel does not tile the input evenly
]


# ---------- tests ----------

def test_output_shape():
    for in_ch, out_ch, k, stride, padding, height, width in CONFIGS:
        layer = Conv2D(in_ch, out_ch, k, stride, padding)
        x = np.random.default_rng(0).random((2, height, width, in_ch))
        out = layer.forward(x)
        expected_h = (height + 2 * padding - k) // stride + 1
        expected_w = (width + 2 * padding - k) // stride + 1
        assert out.shape == (2, expected_h, expected_w, out_ch), (out.shape, expected_h, expected_w)


def test_hand_computed_example():
    x = np.arange(16, dtype=np.float64).reshape(1, 4, 4, 1)
    layer = Conv2D(1, 1, 2)
    layer.W[:] = 1.0
    out = layer.forward(x)
    expected = np.array([[10, 14, 18], [26, 30, 34], [42, 46, 50]], dtype=np.float64)
    assert np.array_equal(out[0, :, :, 0], expected)

    layer.b[:] = 5.0
    assert np.array_equal(layer.forward(x)[0, :, :, 0], expected + 5)

    strided = Conv2D(1, 1, 2, stride=2)
    strided.W[:] = 1.0
    out = strided.forward(x)
    assert np.array_equal(out[0, :, :, 0], np.array([[10, 18], [42, 50]], dtype=np.float64))


def test_padding_example():
    x = np.ones((1, 3, 3, 1))
    layer = Conv2D(1, 1, 3, padding=1)
    layer.W[:] = 1.0
    out = layer.forward(x)
    expected = np.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]], dtype=np.float64)
    assert np.array_equal(out[0, :, :, 0], expected)


def test_forward_matches_naive_loops():
    rng = np.random.default_rng(1)
    for in_ch, out_ch, k, stride, padding, height, width in CONFIGS:
        layer = Conv2D(in_ch, out_ch, k, stride, padding)
        layer.b[:] = rng.standard_normal(out_ch)
        x = rng.standard_normal((3, height, width, in_ch))
        fast = layer.forward(x)
        slow = naive_conv_forward(x, layer.W, layer.b, stride, padding)
        assert np.allclose(fast, slow, atol=1e-10), (in_ch, out_ch, k, stride, padding)


def test_multiple_filters_are_independent():
    rng = np.random.default_rng(2)
    layer = Conv2D(2, 4, 3, padding=1)
    layer.b[:] = rng.standard_normal(4)
    x = rng.standard_normal((2, 6, 6, 2))
    out = layer.forward(x)
    for f in range(4):
        single = Conv2D(2, 1, 3, padding=1)
        single.W[:] = layer.W[:, :, :, f:f + 1]
        single.b[:] = layer.b[f:f + 1]
        assert np.allclose(single.forward(x)[..., 0], out[..., f])


def test_gradient_check():
    rng = np.random.default_rng(3)
    worst = 0.0
    for in_ch, out_ch, k, stride, padding, height, width in CONFIGS:
        layer = Conv2D(in_ch, out_ch, k, stride, padding)
        layer.b[:] = rng.standard_normal(out_ch)
        x = rng.standard_normal((2, height, width, in_ch))
        weights = rng.standard_normal(layer.forward(x).shape)   # loss = sum(out * weights), so dL/dout = weights

        def loss():
            return float(np.sum(layer.forward(x) * weights))

        layer.forward(x)
        dx = layer.backward(weights)
        dW, db = layer.dW.copy(), layer.db.copy()

        dx_num = numerical_gradient(loss, x)
        dW_num = numerical_gradient(loss, layer.W)
        db_num = numerical_gradient(loss, layer.b)

        assert dx.shape == x.shape and dW.shape == layer.W.shape and db.shape == layer.b.shape
        for name, analytic, numeric in (("dx", dx, dx_num), ("dW", dW, dW_num), ("db", db, db_num)):
            error = relative_error(analytic, numeric)
            worst = max(worst, error)
            assert error < 1e-6, f"{name} relative error {error:.2e} for config {(in_ch, out_ch, k, stride, padding, height, width)}"
    print(f"      worst relative gradient error: {worst:.2e} (tolerance 1e-6)")


def test_gradients_are_overwritten_not_accumulated():
    rng = np.random.default_rng(4)
    layer = Conv2D(2, 3, 3, padding=1)
    x = rng.standard_normal((2, 5, 5, 2))
    grad_out = rng.standard_normal(layer.forward(x).shape)
    layer.backward(grad_out)
    first_dW, first_db = layer.dW.copy(), layer.db.copy()
    layer.backward(grad_out)
    assert np.array_equal(layer.dW, first_dW) and np.array_equal(layer.db, first_db)


def test_parameters_interface():
    layer = Conv2D(2, 3, 3, padding=1)
    pairs = layer.parameters()
    assert len(pairs) == 2
    assert pairs[0][0] is layer.W and pairs[0][1] is layer.dW
    assert pairs[1][0] is layer.b and pairs[1][1] is layer.db
    for param, grad in pairs:
        assert param.shape == grad.shape

    # references taken before backward() must see the new gradients
    x = np.random.default_rng(6).random((2, 5, 5, 2))
    out = layer.forward(x)
    layer.backward(np.ones_like(out))
    assert np.abs(pairs[0][1]).sum() > 0 and np.abs(pairs[1][1]).sum() > 0

    # an in-place update must change the layer
    before = layer.forward(x).copy()
    for param, grad in layer.parameters():
        param -= 0.1 * grad
    assert not np.allclose(layer.forward(x), before)


def test_mock_attribute_names():
    layer = Conv2D(2, 3, 3)
    assert layer.kernels is layer.W and layer.bias is layer.b
    assert layer.dkernels is layer.dW and layer.dbias is layer.db
    assert layer.W.shape == (3, 3, 2, 3)          # (k, k, C_in, C_out), same as the mock


def test_initialization():
    big = Conv2D(8, 64, 3, seed=0)
    expected_std = np.sqrt(2.0 / (8 * 3 * 3))
    assert abs(big.W.std() - expected_std) / expected_std < 0.05
    assert np.all(big.b == 0)
    assert big.W.dtype == np.float64 and big.dW.shape == big.W.shape and big.db.shape == big.b.shape

    assert np.array_equal(Conv2D(3, 16, 3, seed=7).W, Conv2D(3, 16, 3, seed=7).W)
    assert not np.array_equal(Conv2D(3, 16, 3, seed=7).W, Conv2D(3, 16, 3, seed=8).W)


def test_global_seed_initialization():
    np.random.seed(0)
    first = Conv2D(3, 4, 3)
    second = Conv2D(3, 4, 3)
    np.random.seed(0)
    first_again = Conv2D(3, 4, 3)
    assert np.array_equal(first.W, first_again.W)
    assert not np.array_equal(first.W, second.W), "two layers should not share the same weights"


def test_errors():
    layer = Conv2D(2, 3, 3)
    expect_error(ValueError, layer.forward, np.zeros((1, 8, 8, 1)))       # wrong channel count
    expect_error(ValueError, layer.forward, np.zeros((2, 8, 8)))          # not 4D
    expect_error(ValueError, layer.forward, np.zeros((1, 2, 2, 2)))       # kernel larger than input
    expect_error(RuntimeError, Conv2D(1, 1, 3).backward, np.zeros((1, 2, 2, 1)))
    layer.forward(np.zeros((1, 8, 8, 2)))
    expect_error(ValueError, layer.backward, np.zeros((1, 5, 5, 3)))      # wrong gradient shape
    expect_error(ValueError, Conv2D, 0, 1, 3)
    expect_error(ValueError, Conv2D, 1, 1, 3, 0)


def test_channels_first_input_rejected():
    layer = Conv2D(1, 8, 3, padding=1)
    expect_error(ValueError, layer.forward, np.zeros((4, 1, 32, 32)))     # NCHW from Person 1's loader


def test_team_input_format():
    x = np.random.default_rng(5).random((16, 32, 32, 1)).astype(np.float32)
    layer = Conv2D(1, 8, 3, padding=1)
    out = layer.forward(x)
    assert out.shape == (16, 32, 32, 8) and out.dtype == np.float64
    dx = layer.backward(np.ones_like(out))
    assert dx.shape == x.shape
    assert np.isfinite(out).all() and np.isfinite(dx).all()

    layer32 = Conv2D(1, 8, 3, padding=1, dtype=np.float32)
    assert layer32.forward(x).dtype == np.float32


# ---------- runner ----------

def main():
    tests = [
        test_output_shape,
        test_hand_computed_example,
        test_padding_example,
        test_forward_matches_naive_loops,
        test_multiple_filters_are_independent,
        test_gradient_check,
        test_gradients_are_overwritten_not_accumulated,
        test_parameters_interface,
        test_mock_attribute_names,
        test_initialization,
        test_global_seed_initialization,
        test_errors,
        test_channels_first_input_rejected,
        test_team_input_format,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as err:
            failed += 1
            print(f"FAIL  {test.__name__}  -> {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")

    x = np.random.default_rng(0).random((32, 32, 32, 1))
    layer = Conv2D(1, 8, 3, padding=1)
    start = time.time()
    out = layer.forward(x)
    layer.backward(np.ones_like(out))
    print(f"Speed: forward + backward, batch of 32 images (32x32x1), 8 filters 3x3: {time.time() - start:.3f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()