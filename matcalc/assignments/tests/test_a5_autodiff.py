"""Tests for A5 -- automatic differentiation from scratch.  Run: pytest tests/test_a5_autodiff.py"""
import numpy as np
import pytest

from mc_assignments import a5_autodiff as a5
from mc_assignments.a5_autodiff import Dual, Tensor
from _helpers import forbidden_usages, require_implemented, rand, num_grad, assert_grad


def test_rules():
    require_implemented(a5)
    assert forbidden_usages(a5) == []


# =============================================================== Part A
def test_dual_textbook():
    y = Dual(2.0, 1.0) ** 3 + 2 * Dual(2.0, 1.0)
    assert (y.val, y.der) == pytest.approx((12.0, 14.0))
    assert a5.derivative(lambda x: x ** 3 + 2 * x, 2.0) == pytest.approx(14.0)


def test_dual_arithmetic_both_sides():
    x = Dual(3.0, 1.0)
    cases = [(x + 1, 4, 1), (1 + x, 4, 1), (x - 1, 2, 1), (1 - x, -2, -1), (2 * x, 6, 2), (x * 2, 6, 2),
             (x * x, 9, 6), (x / 2, 1.5, 0.5), (6 / x, 2, -6 / 9), (x / x, 1, 0), (-x, -3, -1),
             (x ** 0.5, np.sqrt(3), 0.5 / np.sqrt(3)), (x ** -1, 1 / 3, -1 / 9)]
    for i, (d, v, dv) in enumerate(cases):
        assert isinstance(d, Dual), f"case {i} did not return a Dual"
        assert (d.val, d.der) == pytest.approx((v, dv)), f"case {i}"


def test_dual_elementary_functions():
    f = lambda x: a5.d_sin(x) * a5.d_exp(x) + a5.d_log(x) / a5.d_cos(x)
    df = lambda x: np.cos(x) * np.exp(x) + np.sin(x) * np.exp(x) \
        + (1 / x) / np.cos(x) + np.log(x) * np.sin(x) / np.cos(x) ** 2
    for x in [0.3, 1.1, 2.0]:
        assert a5.derivative(f, x) == pytest.approx(df(x), rel=1e-12)
    assert a5.d_sin(0.5) == pytest.approx(np.sin(0.5)), "the d_ functions must also accept plain floats"


def test_gradient_forward():
    f = lambda v: v[0] * v[1] + a5.d_sin(v[2]) * v[0] ** 2
    x = np.array([1.5, -0.5, 0.7])
    g = a5.gradient_forward(f, x)
    assert g.shape == (3,)
    assert np.allclose(g, [x[1] + 2 * x[0] * np.sin(x[2]), x[0], x[0] ** 2 * np.cos(x[2])])


# =============================================================== Part B
def test_unbroadcast():
    g = np.ones((4, 3))
    assert a5.unbroadcast(g, (4, 3)).shape == (4, 3)
    assert np.allclose(a5.unbroadcast(g, (3,)), [4, 4, 4])
    assert np.allclose(a5.unbroadcast(g, (1, 3)), [[4, 4, 4]])
    assert np.allclose(a5.unbroadcast(g, (4, 1)), 3 * np.ones((4, 1)))
    assert np.allclose(a5.unbroadcast(g, ()), 12)
    assert np.allclose(a5.unbroadcast(np.ones((2, 4, 3)), (4, 1)), 6 * np.ones((4, 1)))


def test_textbook_reverse_sweep():
    x1, x2 = Tensor(1.0), Tensor(2.0)
    f = (x1 * x2 + x1) ** 2
    f.backward()
    assert f.data == pytest.approx(9.0)
    assert x1.grad == pytest.approx(18.0) and x2.grad == pytest.approx(6.0)


def _check(build, ref, *arrays, tol=1e-6):
    """build(*Tensors) -> scalar Tensor; ref(*arrays) is the same function in plain NumPy.

    The value must match ref, and every input's .grad must match finite
    differences of ref (not of your own forward pass)."""
    ts = [Tensor(a) for a in arrays]
    out = build(*ts)
    assert out.data.size == 1
    assert float(out.data) == pytest.approx(ref(*arrays), rel=1e-10), "forward value is wrong"
    out.backward()
    for i, (t, a) in enumerate(zip(ts, arrays)):
        def f(z, i=i):
            args = list(arrays); args[i] = z
            return float(ref(*args))
        assert_grad(t.grad, num_grad(f, a), tol=tol, what=f"gradient of input {i}")


_sig = lambda z: 1 / (1 + np.exp(-z))
OPS = {   # name: (Tensor version, NumPy reference, inputs)
    "add_sub_neg":   (lambda a, b: (a + b - (-a) * 2).sum(),
                      lambda a, b: (a + b + a * 2).sum(), [rand(3, 2, seed=1), rand(3, 2, seed=2)]),
    "mul_div":       (lambda a, b: (a * b / (b * b + 1.0)).sum(),
                      lambda a, b: (a * b / (b * b + 1.0)).sum(), [rand(3, 2, seed=3), rand(3, 2, seed=4)]),
    "numbers":       (lambda a: (2.0 * a + 1.0 - a / 3.0 + 1.0 / (a * a + 2.0) - 3.0 + (1.0 - a) * a).sum(),
                      lambda a: (2.0 * a + 1.0 - a / 3.0 + 1.0 / (a * a + 2.0) - 3.0 + (1.0 - a) * a).sum(), [rand(4, seed=5)]),
    "pow":           (lambda a: ((a * a + 1.0) ** 1.5).sum(),
                      lambda a: ((a * a + 1.0) ** 1.5).sum(), [rand(4, seed=6)]),
    "matmul_T":      (lambda a, b: (a @ b.T).tanh().sum(),
                      lambda a, b: np.tanh(a @ b.T).sum(), [rand(3, 4, seed=7), rand(2, 4, seed=8)]),
    "sum_axis":      (lambda a: (a.sum(axis=0) * a.sum(axis=1).sum()).sum() + (a.sum(axis=1) ** 2).sum(),
                      lambda a: (a.sum(axis=0) * a.sum(axis=1).sum()).sum() + (a.sum(axis=1) ** 2).sum(), [rand(3, 4, seed=9)]),
    "mean":          (lambda a: (a.mean(axis=1) ** 2).sum() + (a.mean(axis=0) ** 3).sum() + a.mean(),
                      lambda a: (a.mean(axis=1) ** 2).sum() + (a.mean(axis=0) ** 3).sum() + a.mean(), [rand(3, 4, seed=10)]),
    "exp_log":       (lambda a: ((a * a + 1.0).log() * a.exp()).sum(),
                      lambda a: (np.log(a * a + 1.0) * np.exp(a)).sum(), [rand(5, seed=11)]),
    "tanh_sigmoid":  (lambda a: (a.tanh() * a.sigmoid()).sum(),
                      lambda a: (np.tanh(a) * _sig(a)).sum(), [rand(5, seed=12)]),
    "relu":          (lambda a: (a.relu() * a).sum(),
                      lambda a: (np.maximum(a, 0) * a).sum(), [rand(6, seed=13) + 0.01]),
    "broadcast":     (lambda X, b: ((X + b) * (X + b)).mean(),
                      lambda X, b: ((X + b) * (X + b)).mean(), [rand(5, 3, seed=14), rand(3, seed=15)]),
    "broadcast_col": (lambda X, c: (X * c).sum() + (c - X).exp().sum(),
                      lambda X, c: (X * c).sum() + np.exp(c - X).sum(), [rand(5, 3, seed=16), rand(5, 1, seed=17)]),
}


@pytest.mark.parametrize("name", list(OPS))
def test_tensor_operations(name):
    build, ref, arrays = OPS[name]
    _check(build, ref, *arrays)


def test_reused_nodes_and_diamond():
    x = Tensor(3.0)
    y = x * x                                   # x used twice by one op
    z = y + y * x                               # y used twice; diamond
    z.backward()
    assert x.grad == pytest.approx(2 * 3 + 3 * 9)  # d/dx (x^2 + x^3)


def test_backward_accumulates():
    x = Tensor(np.array([1.0, 2.0]))
    (x * x).sum().backward()
    (x * 3.0).sum().backward()
    assert np.allclose(x.grad, [2 + 3, 4 + 3]), "backward() must ADD to existing .grad"
    s = Tensor(3.0)
    s.backward(); s.backward()
    assert s.grad == pytest.approx(2.0), "backward() adds ones to the output's own .grad too"


def test_two_layer_network():
    X, W1, b1, W2 = rand(6, 3, seed=18), rand(3, 4, seed=19), rand(4, seed=20), rand(4, 2, seed=21)
    _check(lambda X, W1, b1, W2: ((X @ W1 + b1).tanh() @ W2).sigmoid().mean(),
           lambda X, W1, b1, W2: _sig(np.tanh(X @ W1 + b1) @ W2).mean(), X, W1, b1, W2)


def test_cross_entropy_primitive():
    A, y = rand(5, 3, seed=22), np.array([0, 2, 1, 2, 0])
    ref = lambda Z: float(-np.mean((Z - np.log(np.exp(Z).sum(1, keepdims=True)))[np.arange(5), y]))
    t = Tensor(A)
    out = a5.cross_entropy(t, y)
    assert float(out.data) == pytest.approx(ref(A))
    out.backward()
    assert_grad(t.grad, num_grad(ref, A))
    big = Tensor(np.array([[1e4, 0.0]]))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        v = a5.cross_entropy(big, np.array([1]))
        v.backward()
    assert float(v.data) == pytest.approx(1e4) and np.allclose(big.grad, [[1, -1]])


def test_cross_entropy_inside_a_graph():
    X, W, y = rand(8, 3, seed=23), rand(3, 4, seed=24), np.array([0, 1, 2, 3, 0, 1, 2, 3])
    def ref(X, W):
        Z = np.tanh(X @ W) * 2.0
        return float(-np.mean((Z - np.log(np.exp(Z).sum(1, keepdims=True)))[np.arange(8), y]))
    _check(lambda X, W: a5.cross_entropy((X @ W).tanh() * 2.0, y), ref, X, W)
