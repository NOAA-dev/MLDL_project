"""Person 4 tests. Person 3's real layers are used; only Conv2D is mocked unless
USE_REAL_CONV=1 (then Person 2's cnn/convolution.py is used). Run: python -m tests.test_cnn_model"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cnn.cnn_model import CNN, CNNConfig, compute_shapes
from tests import mock_layers
LIB = None if os.environ.get("USE_REAL_CONV") == "1" else mock_layers
H, W, C = 64, 64, 1

def test_shapes_formula():
    s = dict(compute_shapes(CNNConfig()))
    assert s["conv1"] == (64,64,16) and s["pool1"] == (32,32,16)
    assert s["pool2"] == (16,16,32) and s["pool3"] == (8,8,64)
    assert s["flatten"] == (4096,) and s["softmax"] == (26,)

def test_batch_and_single():
    m = CNN(CNNConfig(), LIB)
    X = np.random.RandomState(1).rand(4,H,W,C).astype(np.float32)
    P = m.forward(X, verbose=True)
    assert P.shape == (4,26) and np.allclose(P.sum(1), 1) and (P >= 0).all()
    assert np.allclose(m.forward(X[:1])[0], P[0])               # batch == single
    cid, p = m.predict_single(X[0])
    assert cid == int(P[0].argmax()) and p.shape == (26,)
    assert m.predict(X).shape == (4,) and m.predict(X[0]).shape == (1,)

def test_deterministic():
    X = np.random.RandomState(2).rand(2,H,W,C)
    assert np.array_equal(CNN(CNNConfig(), LIB).forward(X), CNN(CNNConfig(), LIB).forward(X))

def test_rgb_config():
    cfg = CNNConfig(input_shape=(32,32,3), num_classes=10, conv_channels=(4,8))
    assert CNN(cfg, LIB).forward(np.random.rand(2,32,32,3)).shape == (2,10)

def test_bad_input():
    m = CNN(CNNConfig(), LIB)
    for bad in (np.zeros((2,64,64)), np.zeros((2,1,64,64)), np.zeros((2,32,32,1))):  # last two are NCHW / wrong size
        try: m.forward(bad); assert False, "should raise"
        except ValueError: pass

def test_collapse_raises():
    try: compute_shapes(CNNConfig(input_shape=(2,2,1), conv_channels=(2,2,2))); assert False
    except ValueError: pass

def test_parameters_listing():
    m = CNN(CNNConfig(), LIB)
    ps = m.parameters()
    assert any(p.shape == (4096,26) for p, _ in ps) and any(p.shape == (26,) for p, _ in ps)
    for p, g in ps: assert p.shape == g.shape

def test_tail_backward_chain():
    """Person 3's layers composed in MY order: numerical gradient check of softmax-CE w.r.t. conv1
    output, through relu->pool->conv2(mock fwd only so excluded)... small config, one block."""
    cfg = CNNConfig(input_shape=(8,8,1), num_classes=3, conv_channels=(2,))
    m = CNN(cfg, LIB)
    tail = m.layers[1:]                                   # relu1, pool1, flatten, dense
    rs = np.random.RandomState(0)
    z = rs.randn(2,8,8,2); y = np.array([0,2])
    def loss(zz):
        out = zz
        for l in tail: out = l.forward(out)
        p = m.softmax.forward(out)
        return -np.mean(np.log(p[np.arange(2), y]))
    out = z
    for l in tail: out = l.forward(out)
    p = m.softmax.forward(out)
    g = p.copy(); g[np.arange(2), y] -= 1; g /= 2       # (p - onehot)/N, as Person 3 documents
    for l in reversed(tail): g = l.backward(g)
    num = np.zeros_like(z); eps = 1e-6
    for idx in np.ndindex(*z.shape):
        z[idx] += eps; a = loss(z); z[idx] -= 2*eps; b = loss(z); z[idx] += eps
        num[idx] = (a-b)/(2*eps)
    err = np.max(np.abs(num-g))
    assert err < 1e-6, err
    print("   max abs grad error", err)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
