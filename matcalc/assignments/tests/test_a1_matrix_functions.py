"""Tests for A1 -- gradient checking, matrix functions, identities.  Run: pytest tests/test_a1_matrix_functions.py"""
import numpy as np
import pytest

from mc_assignments import a1_matrix_functions as a1
from _helpers import forbidden_usages, require_implemented, rand, spd, rel_err


def test_rules():
    require_implemented(a1)
    assert forbidden_usages(a1) == []


# ---------------- numerical derivatives ----------------
def test_numerical_gradient_textbook():
    g = a1.numerical_gradient(lambda x: x[0] ** 2 * x[1] + 3 * x[1], np.array([1.0, 2.0]))
    assert np.allclose(g, [4, 4], atol=1e-7)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 2)])
def test_numerical_gradient_shapes(shape):
    X = rand(*shape, seed=1)
    W = rand(*shape, seed=2)
    g = a1.numerical_gradient(lambda Z: float(np.sum(W * Z ** 2)), X)
    assert g.shape == X.shape
    assert np.allclose(g, 2 * W * X, atol=1e-7)


def test_numerical_gradient_does_not_modify_input():
    X = rand(3, 3); X0 = X.copy()
    a1.numerical_gradient(lambda Z: float(np.sum(Z ** 3)), X)
    assert np.array_equal(X, X0), "numerical_gradient changed its input array"


def test_numerical_jacobian():
    g = lambda x: np.array([x[0] ** 2 * x[1], 5 * x[0] + np.sin(x[1])])
    J = a1.numerical_jacobian(g, np.array([1.0, 0.0]))
    assert J.shape == (2, 2) and np.allclose(J, [[0, 1], [5, 1]], atol=1e-7)
    A = rand(4, 3)
    J2 = a1.numerical_jacobian(lambda x: A @ x, rand(3, seed=5))
    assert J2.shape == (4, 3) and np.allclose(J2, A, atol=1e-7)


def test_relative_error():
    assert a1.relative_error([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert a1.relative_error([1.0, 0.0], [0.0, 1.0]) == pytest.approx(np.sqrt(2))
    assert a1.relative_error([3.0, 4.0], [3.0, 4.5]) == pytest.approx(0.5 / np.hypot(3, 4.5))
    assert a1.relative_error(np.zeros(3), np.zeros(3)) == 0.0
    assert a1.relative_error([1e-8, 0], [1e-8, 1e-16]) < 1e-7


# ---------------- matrix functions ----------------
def test_logdet_spd_textbook_and_overflow():
    assert a1.logdet_spd(np.array([[4.0, 2.0], [2.0, 3.0]])) == pytest.approx(np.log(8))
    assert a1.logdet_spd(0.1 * np.eye(400)) == pytest.approx(400 * np.log(0.1))
    assert a1.logdet_spd(1e3 * np.eye(200)) == pytest.approx(200 * np.log(1e3))
    S = spd(6, seed=3)
    assert a1.logdet_spd(S) == pytest.approx(np.linalg.slogdet(S)[1])


def test_logdet_spd_rejects_indefinite():
    with pytest.raises(ValueError):
        a1.logdet_spd(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_sqrtm_and_inverse_sqrt():
    R = a1.sqrtm_psd(np.array([[5.0, 4.0], [4.0, 5.0]]))
    assert np.allclose(R, [[2, 1], [1, 2]])
    S = spd(5, seed=4)
    R = a1.sqrtm_psd(S)
    assert np.allclose(R, R.T) and np.allclose(R @ R, S)
    assert np.all(np.linalg.eigvalsh(R) >= -1e-12)
    W = a1.inv_sqrtm_pd(S)
    assert np.allclose(W @ S @ W, np.eye(5))
    v = np.array([1.0, 2.0])
    assert np.allclose(a1.sqrtm_psd(np.outer(v, v)) @ a1.sqrtm_psd(np.outer(v, v)), np.outer(v, v))


def test_sym_matrix_function():
    S = spd(4, seed=6)
    assert np.allclose(a1.sym_matrix_function(S, lambda w: w ** 2), S @ S)
    assert np.allclose(a1.sym_matrix_function(S, lambda w: 1 / w), np.linalg.inv(S))
    assert np.trace(a1.sym_matrix_function(S, np.log)) == pytest.approx(np.linalg.slogdet(S)[1])


def test_expm():
    E = a1.expm(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert np.allclose(E, [[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]])
    th = 0.7
    assert np.allclose(a1.expm(np.array([[0, -th], [th, 0]])), [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert np.allclose(a1.expm(np.array([[0.0, 2.0], [0.0, 0.0]])), [[1, 2], [0, 1]])
    M = rand(4, 4, seed=7)
    assert np.linalg.det(a1.expm(M)) == pytest.approx(np.exp(np.trace(M)), rel=1e-9)


def test_expm_large_norm_needs_scaling():
    th = 20.0                                  # ||A|| = 20: plain Taylor with 20 terms is far off
    R = a1.expm(np.array([[0, -th], [th, 0]]))
    assert rel_err(R, [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) < 1e-9
    S = spd(3, seed=8) * 3
    w, Q = np.linalg.eigh(S)
    assert rel_err(a1.expm(S), (Q * np.exp(w)) @ Q.T) < 1e-9


# ---------------- identities ----------------
def test_sherman_morrison():
    A = np.diag([2.0, 4.0]); u = v = np.array([1.0, 1.0])
    assert np.allclose(a1.sherman_morrison(np.linalg.inv(A), u, v), np.array([[5, -1], [-1, 3]]) / 14)
    A = spd(6, seed=9); u, v = rand(6, seed=1), rand(6, seed=2)
    assert np.allclose(a1.sherman_morrison(np.linalg.inv(A), u, v), np.linalg.inv(A + np.outer(u, v)))


def test_woodbury():
    n, k = 30, 3
    d = np.random.default_rng(0).uniform(1, 2, n)
    U, V, C = rand(n, k, seed=1), rand(k, n, seed=2), spd(k, seed=3)
    W = a1.woodbury(np.diag(1 / d), U, C, V)
    assert np.allclose(W, np.linalg.inv(np.diag(d) + U @ C @ V))


def test_block_inverse_and_logdet():
    M = np.array([[2.0, 1, 0], [1, 2, 1], [0, 1, 2]])
    Minv = a1.block_inverse(M[:1, :1], M[:1, 1:], M[1:, :1], M[1:, 1:])
    assert Minv.shape == (3, 3) and np.allclose(Minv, np.array([[3, -2, 1], [-2, 4, -2], [1, -2, 3]]) / 4)
    B = rand(7, 7, seed=4) + 3 * np.eye(7)
    blocks = (B[:3, :3], B[:3, 3:], B[3:, :3], B[3:, 3:])
    assert np.allclose(a1.block_inverse(*blocks), np.linalg.inv(B))
    assert a1.block_logabsdet(*blocks) == pytest.approx(np.linalg.slogdet(B)[1])
    assert a1.block_logabsdet(M[:1, :1], M[:1, 1:], M[1:, :1], M[1:, 1:]) == pytest.approx(np.log(4))
