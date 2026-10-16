"""Tests for A6 -- attention, normalisation, flows, traces.  Run: pytest tests/test_a6_layers.py"""
import numpy as np
import pytest

from mc_assignments import a6_layers as a6
from _helpers import forbidden_usages, require_implemented, rand, spd, num_grad, num_jac, assert_grad


def test_rules():
    require_implemented(a6)
    assert forbidden_usages(a6) == []


# ---------------------------------------------------------------- softmax rows
def test_softmax_rows():
    S = np.array([[1000.0, 1001.0, 1002.0], [0.0, -np.inf, -np.inf]])
    A = a6.softmax_rows(S)
    assert np.allclose(A, [[0.09003057, 0.24472847, 0.66524096], [1, 0, 0]])


def test_softmax_rows_backward():
    S, C = rand(4, 5, seed=1), rand(4, 5, seed=2)
    A = a6.softmax_rows(S)
    dS = a6.softmax_rows_backward(A, C)
    assert_grad(dS, num_grad(lambda Z: float(np.sum(C * a6.softmax_rows(Z))), S))


# ---------------------------------------------------------------- attention
def test_attention_textbook_example():
    one = np.ones((1, 1))
    Y, cache = a6.attention_forward(np.array([[0.0], [1.0]]), one, one, one)
    assert Y[0, 0] == pytest.approx(0.5)
    dX, dWq, dWk, dWv = a6.attention_backward(np.array([[1.0], [0.0]]), cache)
    assert np.allclose(dX, [[0.75], [0.5]])


def _attention_setup(seed=3):
    X = rand(5, 4, seed=seed)
    Wq, Wk, Wv = rand(4, 3, seed=seed + 1), rand(4, 3, seed=seed + 2), rand(4, 2, seed=seed + 3)
    return X, Wq, Wk, Wv, rand(5, 2, seed=seed + 4)


def test_attention_forward_shape_and_scaling():
    X, Wq, Wk, Wv, _ = _attention_setup()
    Y, _ = a6.attention_forward(X, Wq, Wk, Wv)
    assert Y.shape == (5, 2)
    S = (X @ Wq) @ (X @ Wk).T / np.sqrt(3)
    A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
    assert np.allclose(Y, A @ X @ Wv), "scores must be divided by sqrt(d_k)"


@pytest.mark.parametrize("causal", [False, True])
def test_attention_backward(causal):
    X, Wq, Wk, Wv, C = _attention_setup()
    Y, cache = a6.attention_forward(X, Wq, Wk, Wv, causal=causal)
    grads = a6.attention_backward(C, cache)
    args = [X, Wq, Wk, Wv]
    for i, name in enumerate(["dX", "dWq", "dWk", "dWv"]):
        def f(Z, i=i):
            a = list(args); a[i] = Z
            return float(np.sum(C * a6.attention_forward(*a, causal=causal)[0]))
        assert_grad(grads[i], num_grad(f, args[i]), what=name)


def test_causal_mask():
    X, Wq, Wk, Wv, _ = _attention_setup()
    Y, _ = a6.attention_forward(X, Wq, Wk, Wv, causal=True)
    assert np.allclose(Y[0], X[0] @ Wv), "token 0 can only attend to itself"
    X2 = X.copy(); X2[3:] += 10.0                # changing later tokens must not change earlier outputs
    Y2, _ = a6.attention_forward(X2, Wq, Wk, Wv, causal=True)
    assert np.allclose(Y[:3], Y2[:3])


# ---------------------------------------------------------------- normalisation
def test_layernorm_forward():
    X = rand(4, 6, seed=8) * 3 + 2
    Y, _ = a6.layernorm_forward(X, np.ones(6), np.zeros(6), eps=0.0)
    assert np.allclose(Y.mean(axis=1), 0) and np.allclose(Y.var(axis=1), 1)
    g, b = rand(6, seed=9), rand(6, seed=10)
    assert np.allclose(a6.layernorm_forward(X, g, b, eps=0.0)[0], g * Y + b)


def test_layernorm_textbook_jacobian():
    x = np.array([[1.0, 2.0, 3.0]])
    _, cache = a6.layernorm_forward(x, np.ones(3), np.zeros(3), eps=0.0)
    J = np.array([a6.layernorm_backward(e[None, :], cache)[0][0] for e in np.eye(3)])
    expected = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 6 / np.sqrt(2 / 3)
    assert np.allclose(J, expected)


@pytest.mark.parametrize("which", ["layernorm", "batchnorm"])
def test_norm_backward(which):
    fwd, bwd = getattr(a6, which + "_forward"), getattr(a6, which + "_backward")
    X, C = rand(6, 5, seed=11) * 2 + 1, rand(6, 5, seed=12)
    g, b = rand(5, seed=13), rand(5, seed=14)
    _, cache = fwd(X, g, b)
    dX, dg, db = bwd(C, cache)
    assert_grad(dX, num_grad(lambda Z: float(np.sum(C * fwd(Z, g, b)[0])), X), what="dX")
    assert_grad(dg, num_grad(lambda Z: float(np.sum(C * fwd(X, Z, b)[0])), g), what="dgamma")
    assert_grad(db, num_grad(lambda Z: float(np.sum(C * fwd(X, g, Z)[0])), b), what="dbeta")


def test_batchnorm_is_columnwise():
    X = rand(8, 3, seed=15) * 4 - 1
    Y, _ = a6.batchnorm_forward(X, np.ones(3), np.zeros(3), eps=0.0)
    assert np.allclose(Y.mean(axis=0), 0) and np.allclose(Y.var(axis=0), 1)


# ---------------------------------------------------------------- flows
def _coupling_params(d=2, seed=16):
    return rand(d, d, seed=seed), rand(d, seed=seed + 1), rand(d, d, seed=seed + 2), rand(d, seed=seed + 3)


def test_coupling_inverse_and_logdet():
    P = _coupling_params()
    x = rand(5, 4, seed=20)
    z, logdet = a6.coupling_forward(x, *P)
    assert z.shape == (5, 4) and logdet.shape == (5,)
    Ws, bs, Wt, bt = P
    sc = np.tanh(x[:, :2] @ Ws + bs)
    assert np.allclose(z, np.hstack([x[:, :2], x[:, 2:] * np.exp(sc) + x[:, :2] @ Wt + bt]))
    assert np.allclose(a6.coupling_inverse(z, *P), x)
    for n in range(5):
        J = num_jac(lambda v: a6.coupling_forward(v[None, :], *P)[0][0], x[n])
        assert logdet[n] == pytest.approx(np.log(abs(np.linalg.det(J))), abs=1e-6)
        assert np.allclose(np.triu(J[:2, 2:]), 0, atol=1e-8)


def test_linear_flow_textbook_value():
    v, _, _ = a6.linear_flow_nll(np.array([[2.0, 3.0]]), np.array([[2.0, 0.0], [1.0, 1.0]]), np.zeros(2))
    assert v == pytest.approx(np.log(2 * np.pi) + 2.5 + np.log(2))


def test_linear_flow_gradients():
    X = rand(20, 3, seed=21) @ rand(3, 3, seed=22) + 1.0
    A, b = rand(3, 3, seed=23) + 2 * np.eye(3), rand(3, seed=24)
    v, dA, db = a6.linear_flow_nll(X, A, b)
    assert_grad(dA, num_grad(lambda M: a6.linear_flow_nll(X, M, b)[0], A), what="dA")
    assert_grad(db, num_grad(lambda c: a6.linear_flow_nll(X, A, c)[0], b), what="db")
    Z = np.linalg.solve(A, (X - b).T).T
    assert v == pytest.approx(1.5 * np.log(2 * np.pi) + 0.5 * np.mean(np.sum(Z ** 2, 1)) + np.log(abs(np.linalg.det(A))))


# ---------------------------------------------------------------- traces and eigenvectors
def test_hutchinson():
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    rng = np.random.default_rng(0)
    est = a6.hutchinson_trace(lambda v: M @ v, 2, 10, np.random.default_rng(0))
    ref = np.mean([e @ M @ e for e in (rng.choice([-1.0, 1.0], size=2) for _ in range(10))])
    assert est == pytest.approx(ref), "draw each vector with rng.choice([-1.0, 1.0], size=n)"
    D = np.diag(rand(6, seed=25))
    assert a6.hutchinson_trace(lambda v: D @ v, 6, 3, np.random.default_rng(1)) == pytest.approx(np.trace(D))
    B = rand(30, 30, seed=26)
    est = a6.hutchinson_trace(lambda v: B @ v, 30, 4000, np.random.default_rng(2))
    assert abs(est - np.trace(B)) < 3.0          # standard error is about 0.6 here


def test_rayleigh_quotient():
    S = spd(4, seed=27); x = rand(4, seed=28)
    R, g = a6.rayleigh_quotient(S, x)
    assert R == pytest.approx(x @ S @ x / (x @ x))
    assert_grad(g, num_grad(lambda v: a6.rayleigh_quotient(S, v)[0], x))
    w, Q = np.linalg.eigh(S)
    assert np.allclose(a6.rayleigh_quotient(S, Q[:, 2])[1], 0, atol=1e-10)


def test_top_eigenvector():
    lam, u = a6.top_eigenvector(np.array([[2.0, 1.0], [1.0, 2.0]]))
    assert lam == pytest.approx(3.0) and abs(u @ np.array([1, 1]) / np.sqrt(2)) == pytest.approx(1.0)
    S = spd(6, seed=29) * 50
    w, Q = np.linalg.eigh(S)
    lam, u = a6.top_eigenvector(S, n_steps=2000)
    assert lam == pytest.approx(w[-1], rel=1e-8) and np.linalg.norm(u) == pytest.approx(1.0)
    assert abs(u @ Q[:, -1]) == pytest.approx(1.0, abs=1e-6)
