"""Tests for Assignment 2 -- orthogonality and QR.   Run:  pytest tests/test_a2_orthogonal.py"""
import numpy as np
import pytest

from svd_assignments import a2_orthogonal as a2
from _helpers import (forbidden_usages, require_implemented, random_matrix, max_abs, ref_solve,
                      assert_orthonormal_columns)


def test_no_forbidden_imports():
    require_implemented(a2)
    assert forbidden_usages(a2) == [], "forbidden imports/usages found"


# ---------------- Gram-Schmidt ----------------
def test_gram_schmidt_textbook_example():
    A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
    Q, R = a2.gram_schmidt(A)
    s2, s3, s6 = np.sqrt(2), np.sqrt(3), np.sqrt(6)
    Q_expected = np.array([[1 / s2, 1 / s6, -1 / s3], [1 / s2, -1 / s6, 1 / s3], [0, 2 / s6, 1 / s3]])
    R_expected = np.array([[s2, 1 / s2, 1 / s2], [0, np.sqrt(1.5), 1 / s6], [0, 0, 2 / s3]])
    assert np.allclose(Q, Q_expected) and np.allclose(R, R_expected)


@pytest.mark.parametrize("m, n", [(1, 1), (4, 1), (5, 3), (6, 6), (20, 8)])
def test_gram_schmidt_properties(m, n):
    A = random_matrix(m, n, seed=m + 10 * n)
    Q, R = a2.gram_schmidt(A)
    assert Q.shape == (m, n) and R.shape == (n, n)
    assert_orthonormal_columns(Q, 1e-12, "Q")
    assert max_abs(np.tril(R, -1)) == 0.0, "R must be upper triangular"
    assert np.all(np.diag(R) > 0), "R must have a positive diagonal"
    assert max_abs(Q @ R - A) < 1e-12


def test_gram_schmidt_dependent_columns():
    A = np.array([[1, 2, 3], [2, 4, 1], [3, 6, 2]], dtype=float)   # column 2 = 2 * column 1
    with pytest.raises(ValueError):
        a2.gram_schmidt(A)


def test_gram_schmidt_is_the_modified_version():
    # kappa = 1e7: modified GS keeps ~1e-10 orthogonality, classical GS only ~1e-4
    A = random_matrix(40, 20, kappa=1e7, seed=0)
    Q, _ = a2.gram_schmidt(A)
    err = max_abs(Q.T @ Q - np.eye(20))
    assert err < 1e-7, f"orthogonality lost (max|Q^T Q - I| = {err:.1e}); use MODIFIED Gram-Schmidt"


# ---------------- Householder QR ----------------
@pytest.mark.parametrize("m, n", [(1, 1), (5, 3), (3, 5), (4, 4), (6, 1), (1, 4)])
def test_qr_householder_properties(m, n):
    A = random_matrix(m, n, seed=3 * m + n)
    Q, R = a2.qr_householder(A)
    assert Q.shape == (m, m) and R.shape == (m, n)
    assert_orthonormal_columns(Q, 1e-12, "Q")
    assert max_abs(np.tril(R, -1)) <= 1e-14 * max(1.0, max_abs(A)), "R must be upper triangular"
    assert max_abs(Q @ R - A) < 1e-12


def test_qr_householder_rank_deficient_and_zero():
    for A in (np.zeros((3, 2)), np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])):
        Q, R = a2.qr_householder(A)
        assert_orthonormal_columns(Q, 1e-12, "Q")
        assert max_abs(Q @ R - A) < 1e-12


def test_qr_householder_stays_orthogonal_when_ill_conditioned():
    A = random_matrix(40, 20, kappa=1e14, seed=1)
    Q, R = a2.qr_householder(A)
    assert_orthonormal_columns(Q, 1e-12, "Q")
    assert max_abs(Q @ R - A) < 1e-12


# ---------------- back substitution ----------------
@pytest.mark.parametrize("n", [1, 3, 10])
def test_back_substitution(n):
    rng = np.random.default_rng(n)
    U = np.triu(rng.standard_normal((n, n))) + 3 * np.eye(n)
    b = rng.standard_normal(n)
    x = a2.back_substitution(U, b)
    assert np.allclose(x, ref_solve(U, b))


def test_back_substitution_singular():
    with pytest.raises(ValueError):
        a2.back_substitution([[1.0, 2.0], [0.0, 0.0]], [1.0, 1.0])


# ---------------- is_orthonormal ----------------
def test_is_orthonormal():
    c, s = np.cos(0.3), np.sin(0.3)
    assert a2.is_orthonormal(np.array([[c, -s], [s, c]]))
    assert a2.is_orthonormal(np.array([[1.0], [0.0], [0.0]]))
    assert not a2.is_orthonormal(np.array([[1.0, 1.0], [0.0, 1.0]]))
    assert not a2.is_orthonormal(2 * np.eye(3))


# ---------------- projection matrix ----------------
def test_projection_matrix_textbook():
    A = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
    P = a2.projection_matrix(A)
    assert np.allclose(6 * P, [[5, 2, -1], [2, 2, 2], [-1, 2, 5]])
    assert np.allclose(P @ np.array([6.0, 0, 0]), [5, 2, -1])


@pytest.mark.parametrize("m, n", [(4, 1), (6, 3), (5, 5)])
def test_projection_matrix_properties(m, n):
    A = random_matrix(m, n, seed=m * 7 + n)
    P = a2.projection_matrix(A)
    assert P.shape == (m, m)
    assert max_abs(P @ P - P) < 1e-12 and max_abs(P - P.T) < 1e-12
    assert max_abs(P @ A - A) < 1e-12
    assert np.trace(P) == pytest.approx(n)


# ---------------- complete_basis ----------------
def test_complete_basis_from_one_vector():
    q = np.ones((3, 1)) / np.sqrt(3)
    Q = a2.complete_basis(q)
    assert Q.shape == (3, 3)
    assert_orthonormal_columns(Q, 1e-12, "the completed basis")
    assert np.allclose(Q[:, :1], q)


@pytest.mark.parametrize("m, k", [(5, 0), (5, 2), (4, 4), (1, 1)])
def test_complete_basis_shapes(m, k):
    Q0 = _orthonormal(m, k)
    Q = a2.complete_basis(Q0)
    assert Q.shape == (m, m)
    assert_orthonormal_columns(Q, 1e-12, "the completed basis")
    assert np.allclose(Q[:, :k], Q0)


def _orthonormal(m, k):
    from _helpers import ref_qr
    return ref_qr(np.random.default_rng(m + k).standard_normal((m, m)))[0][:, :k]
