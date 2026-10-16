"""Tests for Assignment 5 -- pseudo-inverse and least squares.   Run:  pytest tests/test_a5_pinv.py"""
import numpy as np
import pytest

from svd_assignments import a5_pinv as a5
from _helpers import forbidden_usages, require_implemented, random_matrix, max_abs, ref_pinv, ref_lstsq


def test_no_forbidden_imports():
    require_implemented(a5)
    assert forbidden_usages(a5) == [], "forbidden imports/usages found"


# ---------------- pinv: textbook examples ----------------
@pytest.mark.parametrize("A, expected", [
    ([[3, 0], [4, 5]], np.array([[5, 0], [-4, 3]]) / 15),                    # Worked Ex. 9.1
    ([[3, 2, 2], [2, 3, -2]], np.array([[7, 2], [2, 7], [10, -10]]) / 45),   # Worked Ex. 9.2
    ([[1, 0], [0, 1], [1, 1]], np.array([[2, -1, 1], [-1, 2, 1]]) / 3),      # Worked Ex. 9.3
    ([[1, 2], [2, 4], [3, 6]], np.array([[1, 2, 3], [2, 4, 6]]) / 70),       # Worked Ex. 9.4
    ([[1, 1], [1, 1]], np.array([[1, 1], [1, 1]]) / 4),                      # Worked Ex. 9.5
    ([[2, 0, 0], [0, 0, 0], [0, 0, -5]], np.diag([0.5, 0, -0.2])),           # Exercise 9.1
    ([[1], [2], [2]], np.array([[1, 2, 2]]) / 9),                            # Exercise 9.2
])
def test_pinv_textbook(A, expected):
    assert np.allclose(a5.pinv(np.array(A, dtype=float)), expected, atol=1e-12)


def test_pinv_zero_and_scalar():
    Z = a5.pinv(np.zeros((2, 3)))
    assert Z.shape == (3, 2) and np.all(Z == 0)
    assert np.allclose(a5.pinv([[4.0]]), [[0.25]])


@pytest.mark.parametrize("m, n, rank", [(5, 3, None), (3, 5, None), (4, 4, None), (6, 5, 2), (5, 7, 3)])
def test_pinv_random(m, n, rank):
    A = random_matrix(m, n, rank=rank, seed=m * 10 + n)
    X = a5.pinv(A)
    assert X.shape == (n, m)
    assert np.allclose(X, ref_pinv(A), atol=1e-10)
    assert max(a5.penrose_residuals(A, X)) < 1e-10


def test_pinv_rcond():
    D = np.diag([1.0, 1e-10])
    assert np.allclose(np.diag(a5.pinv(D)), [1.0, 1e10])
    assert np.allclose(np.diag(a5.pinv(D, rcond=1e-8)), [1.0, 0.0])


def test_pinv_of_pinv_and_transpose():
    A = random_matrix(5, 3, rank=2, seed=8)
    assert np.allclose(a5.pinv(a5.pinv(A)), A, atol=1e-10)
    assert np.allclose(a5.pinv(A.T), a5.pinv(A).T, atol=1e-10)


# ---------------- Penrose residuals ----------------
def test_penrose_residuals_detect_wrong_answer():
    A = np.array([[1, 2], [2, 4], [3, 6]], dtype=float)
    good = a5.penrose_residuals(A, np.array([[1, 2, 3], [2, 4, 6]]) / 70)
    assert len(good) == 4 and all(isinstance(r, float) for r in good)
    assert max(good) < 1e-12
    bad = a5.penrose_residuals(A, A.T)
    assert bad[0] > 1.0, "AXA = A must fail for X = A^T"


# ---------------- full-rank formulas ----------------
@pytest.mark.parametrize("m, n", [(6, 3), (3, 6), (4, 4)])
def test_pinv_full_rank(m, n):
    A = random_matrix(m, n, seed=m + n)
    assert np.allclose(a5.pinv_full_rank(A), ref_pinv(A), atol=1e-10)


def test_pinv_full_rank_rejects_rank_deficient():
    with pytest.raises(ValueError):
        a5.pinv_full_rank([[1, 2], [2, 4], [3, 6]])


# ---------------- least squares ----------------
LINE_A = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
LINE_B = np.array([6.0, 0.0, 0.0])


@pytest.mark.parametrize("solver", ["lstsq_normal", "lstsq_qr", "lstsq_min_norm"])
def test_line_fit_textbook(solver):
    assert np.allclose(getattr(a5, solver)(LINE_A, LINE_B), [5, -3])


@pytest.mark.parametrize("solver", ["lstsq_normal", "lstsq_qr", "lstsq_min_norm"])
def test_least_squares_random_tall(solver):
    A = random_matrix(20, 5, seed=3)
    b = np.random.default_rng(3).standard_normal(20)
    x = getattr(a5, solver)(A, b)
    assert np.allclose(x, ref_lstsq(A, b, rcond=None)[0], atol=1e-10)
    assert max_abs(A.T @ (b - A @ x)) < 1e-10, "residual must be orthogonal to the columns"


def test_cholesky():
    S = np.array([[4.0, 2.0], [2.0, 3.0]])
    L = a5.cholesky(S)
    assert np.allclose(np.triu(L, 1), 0) and np.allclose(L @ L.T, S)
    with pytest.raises(ValueError):
        a5.cholesky([[1.0, 2.0], [2.0, 1.0]])


def test_lstsq_normal_rejects_dependent_columns():
    with pytest.raises(ValueError):
        a5.lstsq_normal([[1, 2], [2, 4], [3, 6]], [1, 2, 4])


def test_lstsq_qr_beats_normal_equations_when_ill_conditioned():
    t = np.linspace(0, 1, 50)
    A = np.column_stack([t ** j for j in range(11)])      # degree-10 Vandermonde, kappa ~ 1e7
    c = np.ones(11)
    x = a5.lstsq_qr(A, A @ c)
    assert max_abs(x - c) < 1e-6


def test_min_norm_underdetermined_and_rank_deficient():
    assert np.allclose(a5.lstsq_min_norm([[1, 2]], [5]), [1, 2])                       # Worked Ex. 8.4
    assert np.allclose(a5.lstsq_min_norm([[1, 1, 1]], [3]), [1, 1, 1])                 # Exercise 8.2
    x = a5.lstsq_min_norm([[1, 2], [2, 4], [3, 6]], [1, 2, 4])                         # Worked Ex. 9.4
    assert np.allclose(x, [17 / 70, 17 / 35])
    A = random_matrix(6, 8, rank=3, seed=12)
    b = np.random.default_rng(12).standard_normal(6)
    assert np.allclose(a5.lstsq_min_norm(A, b), ref_lstsq(A, b, rcond=None)[0], atol=1e-10)
