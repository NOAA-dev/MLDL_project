"""Person 4 tests. Uses mocks unless USE_REAL_LAYERS=1. Run: python -m tests.test_cnn_model"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cnn.cnn_model import CNN, CNNConfig, compute_shapes
from tests import mock_layers
LIB = None if os.environ.get("USE_REAL_LAYERS") == "1" else mock_layers

def test_shapes_formula():
    s = dict(compute_shapes(CNNConfig()))
    assert s["conv1"] == (8,64,64) and s["pool1"] == (8,32,32)
    assert s["pool2"] == (16,16,16) and s["flatten"] == (4096,) and s["softmax"] == (26,)

def test_batch_and_single():
    m = CNN(CNNConfig(), LIB)
    X = np.random.RandomState(1).rand(4,1,64,64).astype(np.float32)
    P = m.forward(X, verbose=True)
    assert P.shape == (4,26) and np.allclose(P.sum(1), 1, atol=1e-5) and (P >= 0).all()
    assert np.allclose(m.forward(X[:1])[0], P[0], atol=1e-5)   # batch == single
    cid, p = m.predict_single(X[0])
    assert cid == int(P[0].argmax()) and p.shape == (26,)
    assert m.predict(X).shape == (4,)

def test_deterministic():
    X = np.random.RandomState(2).rand(2,1,64,64).astype(np.float32)
    assert np.array_equal(CNN(CNNConfig(), LIB).forward(X), CNN(CNNConfig(), LIB).forward(X))

def test_rgb_config():
    cfg = CNNConfig(input_shape=(3,32,32), num_classes=10, conv_channels=(4,8))
    assert CNN(cfg, LIB).forward(np.random.rand(2,3,32,32).astype(np.float32)).shape == (2,10)

def test_bad_input():
    m = CNN(CNNConfig(), LIB)
    for bad in (np.zeros((2,64,64)), np.zeros((2,3,64,64)), np.zeros((2,1,32,32))):
        try: m.forward(bad); assert False, "should raise"
        except ValueError: pass

def test_collapse_raises():
    try: compute_shapes(CNNConfig(input_shape=(1,2,2), conv_channels=(2,2,2))); assert False
    except ValueError: pass

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
