"""Tests for Assignment 3 -- eigenvalues of symmetric matrices.   Run:  pytest tests/test_a3_eigen.py"""
import numpy as np
import pytest

from svd_assignments import a3_eigen as a3
from _helpers import (forbidden_usages, require_implemented, random_matrix, max_abs, ref_eigvalsh,
                      assert_orthonormal_columns, assert_parallel)


def test_no_forbidden_imports():
    require_implemented(a3)
    assert forbidden_usages(a3) == [], "forbidden imports/usages found"


def _sym(n, seed):
    B = random_matrix(n, n, seed=seed)
    return (B + B.T) / 2


# ---------------- Rayleigh quotient ----------------
@pytest.mark.parametrize("x, expected", [([1, 0], 2.0), ([2, 1], 14 / 5), ([5, 4], 122 / 41)])
def test_rayleigh_quotient_textbook(x, expected):
    assert a3.rayleigh_quotient([[2, 1], [1, 2]], x) == pytest.approx(expected)


def test_rayleigh_quotient_zero_vector():
    with pytest.raises(ValueError):
        a3.rayleigh_quotient(np.eye(2), [0, 0])


# ---------------- power iteration ----------------
def test_power_iteration_textbook():
    lam, v = a3.power_iteration(np.array([[2.0, 1.0], [1.0, 2.0]]), x0=[1.0, 0.0])
    assert lam == pytest.approx(3.0, abs=1e-9)
    assert abs(np.sum(v * v) - 1) < 1e-12, "v must be a unit vector"
    assert_parallel(v, [1, 1], tol=1e-6)


def test_power_iteration_random_spd():
    Q = _q(6)
    lams = np.array([10.0, 4.0, 3.0, 2.0, 1.0, 0.5])
    S = (Q * lams) @ Q.T
    lam, v = a3.power_iteration(S)
    assert lam == pytest.approx(10.0, rel=1e-9)
    assert_parallel(v, Q[:, 0], tol=1e-6)


def _q(n):
    from _helpers import ref_qr
    return ref_qr(np.random.default_rng(n).standard_normal((n, n)))[0]


# ---------------- Jacobi eigenvalue algorithm ----------------
def test_jacobi_textbook_repeated_eigenvalue():
    S = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)
    lam, V = a3.jacobi_eigh(S)
    assert np.allclose(lam, [4, 1, 1])
    assert_orthonormal_columns(V, 1e-12, "V")
    assert max_abs(S @ V - V * lam) < 1e-12


@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_jacobi_random_symmetric(n):
    S = _sym(n, seed=n)
    lam, V = a3.jacobi_eigh(S)
    assert lam.shape == (n,) and V.shape == (n, n)
    assert np.all(np.diff(lam) <= 1e-12), "eigenvalues must be sorted in decreasing order"
    assert np.allclose(lam, np.sort(ref_eigvalsh(S))[::-1], atol=1e-12)
    assert_orthonormal_columns(V, 1e-12, "V")
    assert max_abs(S @ V - V * lam) < 1e-11


def test_jacobi_diagonal_and_negative():
    lam, V = a3.jacobi_eigh(np.diag([-3.0, 5.0, 0.0]))
    assert np.allclose(lam, [5, 0, -3])
    lam, _ = a3.jacobi_eigh([[0, 1], [1, 0]])
    assert np.allclose(lam, [1, -1])


def test_jacobi_rejects_bad_input():
    with pytest.raises(ValueError):
        a3.jacobi_eigh([[1, 2], [3, 4]])          # not symmetric
    with pytest.raises(ValueError):
        a3.jacobi_eigh(np.ones((2, 3)))           # not square


def test_jacobi_small_eigenvalues_are_accurate():
    # eigenvalues 1, 1e-4, 1e-8: absolute accuracy ~1e-15 is required
    Q = _q(3)
    S = (Q * np.array([1.0, 1e-4, 1e-8])) @ Q.T
    lam, _ = a3.jacobi_eigh(S)
    assert np.allclose(lam, [1.0, 1e-4, 1e-8], rtol=0, atol=1e-14)


# ---------------- spectral reconstruction ----------------
def test_spectral_reconstruct_roundtrip():
    S = _sym(7, seed=11)
    lam, V = a3.jacobi_eigh(S)
    assert max_abs(a3.spectral_reconstruct(lam, V) - S) < 1e-12
    assert np.allclose(a3.spectral_reconstruct([3, 1], np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
                       [[2, 1], [1, 2]])


# ---------------- definiteness ----------------
@pytest.mark.parametrize("S, label", [
    ([[2, 1], [1, 2]], "positive definite"),
    ([[1, 2], [2, 4]], "positive semidefinite"),
    ([[1, 3], [3, 1]], "indefinite"),
    ([[-2, 1], [1, -2]], "negative definite"),
    ([[-1, 0], [0, 0]], "negative semidefinite"),
    (np.zeros((3, 3)), "positive semidefinite"),
    ([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], "positive definite"),
])
def test_classify_definiteness(S, label):
    assert a3.classify_definiteness(S) == label


def test_classify_gram_matrix_is_psd():
    A = random_matrix(3, 5, seed=2)                # A^T A is 5x5 with rank 3
    assert a3.classify_definiteness(A.T @ A) == "positive semidefinite"
    assert a3.classify_definiteness(A @ A.T) == "positive definite"
