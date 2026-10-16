"""Tests for A3 -- derivatives with respect to matrices; Gaussians.  Run: pytest tests/test_a3_matrix_derivatives.py"""
import numpy as np
import pytest

from mc_assignments import a3_matrix_derivatives as a3
from _helpers import forbidden_usages, require_implemented, rand, spd, num_grad, assert_grad


def test_rules():
    require_implemented(a3)
    assert forbidden_usages(a3) == []


# --------------------------------------------- the derivative catalogue
A4 = rand(4, 4, seed=1)
a4, b4 = rand(4, seed=2), rand(4, seed=3)
X4 = rand(4, 4, seed=4) + 4 * np.eye(4)          # comfortably invertible, not symmetric
CASES = {
    "trace_AX":      (lambda X: a3.trace_AX(X, A4),            X4),
    "trace_XtAX":    (lambda X: a3.trace_XtAX(X, rand(4, 4, seed=5)), rand(4, 3, seed=6)),
    "bilinear":      (lambda X: a3.bilinear(X, rand(3, seed=7), b4), rand(3, 4, seed=8)),
    "logabsdet":     (a3.logabsdet,                             X4),
    "det":           (a3.det,                                   X4 / 3),
    "trace_inv":     (lambda X: a3.trace_inv(X, A4),            X4),
    "inv_bilinear":  (lambda X: a3.inv_bilinear(X, a4, b4),     X4),
    "frobenius_lsq": (lambda X: a3.frobenius_lsq(X, rand(5, 4, seed=9), rand(5, 3, seed=10)), rand(4, 3, seed=11)),
    "tanh_weighted": (lambda X: a3.tanh_weighted(X, rand(3, 5, seed=12)), rand(3, 5, seed=13)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_catalogue_gradients(name):
    f, X = CASES[name]
    value, G = f(X.copy())
    assert np.ndim(value) == 0, f"{name}: the value must be a scalar"
    assert_grad(G, num_grad(lambda Z: f(Z)[0], X), what=f"{name} gradient")


def test_catalogue_values():
    X = X4
    assert a3.trace_AX(X, A4)[0] == pytest.approx(np.trace(A4 @ X))
    assert a3.logabsdet(X)[0] == pytest.approx(np.log(abs(np.linalg.det(X))))
    assert a3.det(X)[0] == pytest.approx(np.linalg.det(X))
    assert a3.trace_inv(X, A4)[0] == pytest.approx(np.trace(np.linalg.inv(X) @ A4))
    assert a3.inv_bilinear(X, a4, b4)[0] == pytest.approx(a4 @ np.linalg.inv(X) @ b4)


def test_logdet_textbook_example():
    value, G = a3.logabsdet(np.array([[2.0, 1.0], [1.0, 1.0]]))
    assert value == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(G, [[1, -1], [-1, 2]])


def test_trace_quadratic_textbook_example():
    value, G = a3.trace_XtAX(np.array([[1.0, 0.0], [1.0, 1.0]]), np.array([[1.0, 2.0], [0.0, 1.0]]))
    assert value == pytest.approx(5.0) and np.allclose(G, [[4, 2], [4, 2]])


def test_negative_determinant():
    X = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]) + 0.1 * rand(3, 3, seed=14)
    assert np.linalg.det(X) < 0
    v, G = a3.logabsdet(X)
    assert v == pytest.approx(np.log(-np.linalg.det(X)))
    assert_grad(G, num_grad(lambda Z: a3.logabsdet(Z)[0], X))
    assert_grad(a3.det(X)[1], num_grad(lambda Z: a3.det(Z)[0], X))


# --------------------------------------------- Gaussians
def _nll_direct(X, mu, S):
    N, D = X.shape
    d = X - mu
    return 0.5 * N * D * np.log(2 * np.pi) + 0.5 * N * np.log(np.linalg.det(S)) \
        + 0.5 * np.sum(d @ np.linalg.inv(S) * d)


def test_gaussian_nll_value():
    X, mu, S = rand(12, 3, seed=15), rand(3, seed=16), spd(3, seed=17)
    assert a3.gaussian_nll(X, mu, S)[0] == pytest.approx(_nll_direct(X, mu, S))
    assert a3.gaussian_nll(np.zeros((1, 1)), np.zeros(1), np.eye(1))[0] == pytest.approx(0.5 * np.log(2 * np.pi))


def test_gaussian_nll_gradients():
    X, mu, S = rand(12, 3, seed=18), rand(3, seed=19), spd(3, seed=20)
    _, dmu, dS = a3.gaussian_nll(X, mu, S)
    assert_grad(dmu, num_grad(lambda m: a3.gaussian_nll(X, m, S)[0], mu), what="d/dmu")
    assert_grad(dS, num_grad(lambda T: a3.gaussian_nll(X, mu, T)[0], S), what="d/dSigma")


def test_gaussian_mle_textbook_and_stationary():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 0.0]])
    mu, S = a3.gaussian_mle(X)
    assert np.allclose(mu, [2, 2]) and np.allclose(S, np.array([[2, 2], [2, 8]]) / 3)
    X = rand(40, 4, seed=21)
    mu, S = a3.gaussian_mle(X)
    _, dmu, dS = a3.gaussian_nll(X, mu, S)
    assert np.allclose(dmu, 0, atol=1e-9) and np.allclose(dS, 0, atol=1e-9)


def test_gaussian_conditional():
    mu_c, S_c = a3.gaussian_conditional(np.zeros(2), np.array([[4.0, 2.0], [2.0, 3.0]]), [1], [3.0])
    assert np.allclose(mu_c, [2.0]) and np.allclose(S_c, [[8 / 3]])
    mu, S = rand(5, seed=22), spd(5, seed=23)
    idx_b, xb = [0, 3], np.array([0.5, -1.0])
    mu_c, S_c = a3.gaussian_conditional(mu, S, idx_b, xb)
    a = [1, 2, 4]
    L = np.linalg.inv(S)                                  # the precision-matrix form
    Laa, Lab = L[np.ix_(a, a)], L[np.ix_(a, idx_b)]
    assert np.allclose(S_c, np.linalg.inv(Laa))
    assert np.allclose(mu_c, mu[a] - np.linalg.solve(Laa, Lab @ (xb - mu[idx_b])))


def test_kl_gaussians():
    m, S = rand(3, seed=24), spd(3, seed=25)
    assert a3.kl_gaussians(m, S, m, S) == pytest.approx(0.0, abs=1e-12)
    # 1-D closed form: KL(N(0,1) || N(1,4)) = (1/4 + 1/4 - 1 + ln 4) / 2
    assert a3.kl_gaussians(np.zeros(1), np.eye(1), np.ones(1), 4 * np.eye(1)) == pytest.approx(0.5 * (-0.5 + np.log(4)))
    for seed in range(5):
        assert a3.kl_gaussians(rand(3, seed=seed), spd(3, seed=seed + 10), rand(3, seed=seed + 20), spd(3, seed=seed + 30)) > 0


def test_kl_diag_standard():
    v, dmu, dlv = a3.kl_diag_standard(np.array([1.0, 0.0]), np.log(np.array([1.0, 4.0])))
    assert v == pytest.approx(0.5 * (4 - np.log(4))) and np.allclose(dmu, [1, 0]) and np.allclose(dlv, [0, 1.5])
    mu, lv = rand(6, seed=26), rand(6, seed=27)
    v, dmu, dlv = a3.kl_diag_standard(mu, lv)
    assert v == pytest.approx(a3.kl_gaussians(mu, np.diag(np.exp(lv)), np.zeros(6), np.eye(6)))
    assert_grad(dmu, num_grad(lambda m: a3.kl_diag_standard(m, lv)[0], mu), what="d/dmu")
    assert_grad(dlv, num_grad(lambda l: a3.kl_diag_standard(mu, l)[0], lv), what="d/dlogvar")
