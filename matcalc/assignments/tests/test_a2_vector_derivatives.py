"""Tests for A2 -- vector derivatives and optimisation.  Run: pytest tests/test_a2_vector_derivatives.py"""
import numpy as np
import pytest

from mc_assignments import a2_vector_derivatives as a2
from _helpers import forbidden_usages, require_implemented, rand, num_grad, num_jac, assert_grad


def test_rules():
    require_implemented(a2)
    assert forbidden_usages(a2) == []


def test_quadratic_textbook():
    A = np.array([[2.0, 1.0], [1.0, 3.0]]); b = np.array([1.0, -1.0])
    f, g, H = a2.quadratic(np.array([1.0, 2.0]), A, b)
    assert f == pytest.approx(17.0) and np.allclose(g, [9, 13]) and np.allclose(H, 2 * A)


def test_quadratic_nonsymmetric():
    A = rand(5, 5, seed=1); b = rand(5, seed=2); x = rand(5, seed=3)
    f, g, H = a2.quadratic(x, A, b)
    assert_grad(g, num_grad(lambda v: a2.quadratic(v, A, b)[0], x), what="gradient (non-symmetric A)")
    assert np.allclose(H, A + A.T)


@pytest.mark.parametrize("lam", [0.0, 0.5])
def test_least_squares(lam):
    Phi, t, w = rand(10, 3, seed=4), rand(10, seed=5), rand(3, seed=6)
    E, g, H = a2.least_squares(w, Phi, t, lam)
    assert E == pytest.approx(0.5 * np.sum((Phi @ w - t) ** 2) + 0.5 * lam * w @ w)
    assert_grad(g, num_grad(lambda v: a2.least_squares(v, Phi, t, lam)[0], w))
    assert np.allclose(H, num_jac(lambda v: a2.least_squares(v, Phi, t, lam)[1], w), atol=1e-6)


def test_ridge_solution():
    Phi = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]); t = np.array([6.0, 0.0, 0.0])
    assert np.allclose(a2.ridge_solution(Phi, t, 0.0), [5, -3])
    Phi = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])          # dependent columns
    w = a2.ridge_solution(Phi, np.array([1.0, 2.0, 4.0]), 0.1)
    _, g, _ = a2.least_squares(w, Phi, np.array([1.0, 2.0, 4.0]), 0.1)
    assert np.allclose(g, 0, atol=1e-10)


def test_logsumexp_and_softmax_stable():
    a = np.array([1000.0, 1001.0, 1002.0])
    assert np.allclose(a2.softmax(a), [0.09003057, 0.24472847, 0.66524096])
    assert a2.logsumexp(a) == pytest.approx(1002.40760596)
    assert np.isfinite(a2.logsumexp(np.array([-1e4, -1e4])))
    A = rand(4, 6, seed=7) * 50
    assert np.allclose(a2.softmax(A, axis=1).sum(axis=1), 1)
    assert np.allclose(a2.logsumexp(A, axis=1), np.log(np.sum(np.exp(A), axis=1)))


def test_logsumexp_gradient_is_softmax():
    a = rand(5, seed=8)
    assert_grad(a2.softmax(a), num_grad(a2.logsumexp, a))


def test_softmax_jacobian():
    a = np.array([0.0, np.log(2.0), 0.0])
    assert np.allclose(16 * a2.softmax_jacobian(a), [[3, -2, -1], [-2, 4, -2], [-1, -2, 3]])
    a = rand(6, seed=9)
    assert np.allclose(a2.softmax_jacobian(a), num_jac(a2.softmax, a), atol=1e-7)


def test_sigmoid_stable():
    with np.errstate(over="raise"):
        s = a2.sigmoid(np.array([-1e4, -30.0, 0.0, 30.0, 1e4]))
    assert np.allclose(s, [0, 1 / (1 + np.exp(30)), 0.5, 1 / (1 + np.exp(-30)), 1])


def test_logistic_loss():
    Phi = rand(20, 3, seed=10); t = (rand(20, seed=11) > 0).astype(float); w = rand(3, seed=12)
    E, g, H = a2.logistic_loss(w, Phi, t)
    y = 1 / (1 + np.exp(-Phi @ w))
    assert E == pytest.approx(-np.sum(t * np.log(y) + (1 - t) * np.log(1 - y)))
    assert_grad(g, num_grad(lambda v: a2.logistic_loss(v, Phi, t)[0], w))
    assert np.allclose(H, num_jac(lambda v: a2.logistic_loss(v, Phi, t)[1], w), atol=1e-6)
    assert np.all(np.linalg.eigvalsh(H) >= -1e-12)


def test_logistic_loss_extreme_logits():
    Phi = np.array([[1000.0], [-1000.0]]); t = np.array([0.0, 1.0])
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        E, g, H = a2.logistic_loss(np.array([1.0]), Phi, t)
    assert E == pytest.approx(2000.0)
    assert np.allclose(g, [2000.0])


def test_normalize_jacobian():
    assert np.allclose(a2.normalize_jacobian(np.array([3.0, 4.0])), np.array([[16, -12], [-12, 9]]) / 125)
    x = rand(4, seed=13)
    assert np.allclose(a2.normalize_jacobian(x), num_jac(lambda v: v / np.linalg.norm(v), x), atol=1e-7)


def test_gradient_descent_theory():
    H = np.diag([1.0, 10.0])
    assert a2.gd_convergence_factor(H, 0.1) == pytest.approx(0.9)
    assert a2.gd_convergence_factor(H, 0.21) == pytest.approx(1.1)
    assert a2.optimal_learning_rate(H) == pytest.approx(2 / 11)
    assert a2.gd_convergence_factor(H, 2 / 11) == pytest.approx(9 / 11)
    Q = np.linalg.qr(rand(3, 3, seed=14))[0]
    H3 = (Q * np.array([2.0, 5.0, 50.0])) @ Q.T
    assert a2.optimal_learning_rate(H3) == pytest.approx(2 / 52)


def test_newton_logistic():
    x = np.array([1.0, 2.0, -1.0, -2.0]); t = np.array([1.0, 0.0, 0.0, 1.0])
    assert a2.newton_logistic(x[:, None], t, 1)[0] == pytest.approx(-0.4)
    w = a2.newton_logistic(x[:, None], t, 10)
    assert a2.logistic_loss(w, x[:, None], t)[1][0] == pytest.approx(0.0, abs=1e-10)
    Phi = np.column_stack([np.ones(50), rand(50, 2, seed=15)])
    tt = (rand(50, seed=16) > 0).astype(float)
    w = a2.newton_logistic(Phi, tt, 15)
    assert np.allclose(a2.logistic_loss(w, Phi, tt)[1], 0, atol=1e-8)
