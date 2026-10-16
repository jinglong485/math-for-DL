"""Tests for A4 -- backpropagation.  Run: pytest tests/test_a4_backprop.py"""
import numpy as np
import pytest

from mc_assignments import a4_backprop as a4
from _helpers import forbidden_usages, require_implemented, rand, num_grad, assert_grad, make_spirals


def test_rules():
    require_implemented(a4)
    assert forbidden_usages(a4) == []


def _probe(layer, X, C):
    """Scalar L = sum(C * layer(X)); returns analytic dL/dX."""
    layer.forward(X)
    return layer.backward(C)


# ---------------------------------------------------------------- Linear
def test_linear_init_and_forward():
    lin = a4.Linear(3, 2, np.random.default_rng(0))
    W = np.random.default_rng(0).standard_normal((3, 2)) * np.sqrt(1 / 3)
    assert lin.W.shape == (3, 2) and lin.b.shape == (2,)
    assert np.allclose(lin.W, W), "initialise W with rng.standard_normal((d_in, d_out)) * sqrt(1/d_in)"
    assert np.all(lin.b == 0)
    X = rand(5, 3, seed=1)
    lin.b = rand(2, seed=2)
    assert np.allclose(lin.forward(X), X @ lin.W + lin.b)


def test_linear_backward():
    lin = a4.Linear(4, 3, np.random.default_rng(1))
    lin.b = rand(3, seed=3)
    X, C = rand(6, 4, seed=4), rand(6, 3, seed=5)
    dX = _probe(lin, X, C)
    assert_grad(dX, num_grad(lambda Z: float(np.sum(C * (Z @ lin.W + lin.b))), X), what="dL/dX")
    W0 = lin.W.copy(); b0 = lin.b.copy()
    assert_grad(lin.dW, num_grad(lambda W: float(np.sum(C * (X @ W + b0))), W0), what="dW")
    assert_grad(lin.db, num_grad(lambda b: float(np.sum(C * (X @ W0 + b))), b0), what="db")
    assert [p.shape for p in lin.params()] == [g.shape for g in lin.grads()] == [(4, 3), (3,)]
    assert lin.params()[0] is lin.W, "params() must return the parameter arrays themselves (not copies)"


def test_textbook_two_layer_example():
    l1 = a4.Linear(2, 2, np.random.default_rng(0)); l2 = a4.Linear(2, 1, np.random.default_rng(0))
    l1.W = np.array([[1.0, 2.0], [1.0, -1.0]]); l2.W = np.array([[1.0], [-1.0]])
    l1.b = np.zeros(2); l2.b = np.zeros(1)
    act, mse = a4.ReLU(), a4.MSELoss()
    x = np.array([[1.0, 1.0]])
    y = l2.forward(act.forward(l1.forward(x)))
    assert mse.forward(y, np.array([[3.0]])) == pytest.approx(2.0)
    dx = l1.backward(act.backward(l2.backward(mse.backward())))
    assert np.allclose(l2.dW, [[-4], [-2]]) and np.allclose(l1.dW, [[-2, 2], [-2, 2]]) and np.allclose(dx, [[2, -4]])


# ---------------------------------------------------------------- activations
@pytest.mark.parametrize("cls,fn", [(a4.Tanh, np.tanh), (a4.Sigmoid, lambda z: 1 / (1 + np.exp(-z))),
                                    (a4.ReLU, lambda z: np.maximum(z, 0))])
def test_activations(cls, fn):
    Z, C = rand(5, 4, seed=6) + 0.01, rand(5, 4, seed=7)      # +0.01: keep ReLU away from its kink
    layer = cls()
    assert np.allclose(layer.forward(Z), fn(Z))
    dZ = layer.backward(C)
    assert_grad(dZ, num_grad(lambda W: float(np.sum(C * fn(W))), Z), what=f"{cls.__name__} backward")
    assert layer.params() == [] and layer.grads() == []


def test_relu_at_zero_and_sigmoid_extremes():
    r = a4.ReLU(); r.forward(np.array([[-1.0, 0.0, 2.0]]))
    assert np.allclose(r.backward(np.ones((1, 3))), [[0, 0, 1]]), "use ReLU'(0) = 0"
    s = a4.Sigmoid()
    with np.errstate(over="raise"):
        H = s.forward(np.array([[-1e4, 0.0, 1e4]]))
    assert np.allclose(H, [[0, 0.5, 1]])


# ---------------------------------------------------------------- losses
def test_mse_loss():
    Y, T = rand(5, 2, seed=8), rand(5, 2, seed=9)
    m = a4.MSELoss()
    assert m.forward(Y, T) == pytest.approx(np.sum((Y - T) ** 2) / 10)
    assert_grad(m.backward(), num_grad(lambda Z: np.sum((Z - T) ** 2) / 10, Y))


def test_softmax_cross_entropy_textbook():
    ce = a4.SoftmaxCrossEntropy()
    assert ce.forward(np.array([[0.0, np.log(2.0), 0.0]]), np.array([2])) == pytest.approx(np.log(4))
    assert np.allclose(ce.backward(), [[0.25, 0.5, -0.75]])


def test_softmax_cross_entropy_batch_and_stable():
    A, y = rand(7, 4, seed=10), np.array([0, 3, 1, 1, 2, 0, 3])
    ce = a4.SoftmaxCrossEntropy()
    ref = lambda Z: float(-np.mean((Z - np.log(np.exp(Z).sum(1, keepdims=True)))[np.arange(7), y]))
    assert ce.forward(A, y) == pytest.approx(ref(A))
    assert_grad(ce.backward(), num_grad(ref, A), what="dL/dA")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        v = ce.forward(np.array([[1e4, 0.0], [0.0, 1e4]]), np.array([1, 1]))
    assert v == pytest.approx(5e3)


# ---------------------------------------------------------------- MLP
def test_mlp_structure():
    net = a4.MLP([2, 5, 4, 3], activation="relu", seed=0)
    shapes = [p.shape for p in net.params()]
    assert shapes == [(2, 5), (5,), (5, 4), (4,), (4, 3), (3,)]
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((2, 5)) * np.sqrt(1 / 2)
    assert np.allclose(net.params()[0], W1), "create the Linear layers in order with ONE rng"
    assert net.forward(rand(9, 2)).shape == (9, 3)


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_mlp_full_gradient_check(activation):
    X, y = make_spirals(8, 3, seed=1)
    net = a4.MLP([2, 6, 5, 3], activation=activation, seed=2)
    for p in net.params():                      # nonzero biases make the check stronger
        p += 0.1 * np.random.default_rng(3).standard_normal(p.shape)
    net.loss(X, y)
    dX = net.backward()
    grads = [g.copy() for g in net.grads()]
    for i, p in enumerate(net.params()):
        def f(v, p=p):
            old = p.copy(); p[...] = v
            out = net.loss(X, y); p[...] = old
            return out
        assert_grad(grads[i], num_grad(f, p.copy()), what=f"gradient of parameter {i}")
    assert_grad(dX, num_grad(lambda Z: net.loss(Z, y), X), what="dL/dX")


def test_train_updates_in_place_with_momentum():
    X, y = make_spirals(10, 3, seed=4)
    net = a4.MLP([2, 4, 3], seed=5)
    p_ids = [id(p) for p in net.params()]
    p0 = [p.copy() for p in net.params()]
    L0 = net.loss(X, y); net.backward(); g0 = [g.copy() for g in net.grads()]
    hist = a4.train(net, X, y, lr=0.1, epochs=1, momentum=0.9)
    assert len(hist) == 1 and hist[0] == pytest.approx(L0)
    assert [id(p) for p in net.params()] == p_ids, "update the parameters in place (p += ...)"
    for p, q, g in zip(net.params(), p0, g0):
        assert np.allclose(p, q - 0.1 * g)
    # second step uses the velocity: p2 = p1 + 0.9 * (-lr g0) - lr g1
    net2 = a4.MLP([2, 4, 3], seed=5)
    a4.train(net2, X, y, lr=0.1, epochs=1, momentum=0.9)
    p1 = [p.copy() for p in net2.params()]
    net2.loss(X, y); net2.backward(); g1 = [g.copy() for g in net2.grads()]
    net3 = a4.MLP([2, 4, 3], seed=5)
    a4.train(net3, X, y, lr=0.1, epochs=2, momentum=0.9)
    for p, q, ga, gb in zip(net3.params(), p1, g0, g1):
        assert np.allclose(p, q + 0.9 * (-0.1 * ga) - 0.1 * gb)


def test_mlp_learns_spirals():
    X, y = make_spirals(100, 3, seed=0)
    net = a4.MLP([2, 64, 64, 3], activation="tanh", seed=0)
    hist = np.asarray(a4.train(net, X, y, lr=0.2, epochs=1500, momentum=0.9))
    assert len(hist) == 1500 and hist[-1] < 0.3 * hist[0]
    acc = a4.accuracy(net, X, y)
    assert acc >= 0.95, f"training accuracy only {acc:.1%}"


def test_accuracy():
    class Fixed:
        def forward(self, X):
            return X
    assert a4.accuracy(Fixed(), np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]), np.array([0, 0, 0])) == pytest.approx(2 / 3)
