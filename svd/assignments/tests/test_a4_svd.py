"""Tests for Assignment 4 -- the SVD.   Run:  pytest tests/test_a4_svd.py"""
import numpy as np
import pytest

from svd_assignments import a4_svd as a4
from _helpers import (forbidden_usages, require_implemented, random_matrix, max_abs, ref_svd, ref_matrix_rank,
                      assert_orthonormal_columns)

METHODS = ["svd_jacobi", "svd_via_eig"]
# shapes: (m, n, rank or None)
SHAPES = [(1, 1, None), (3, 2, None), (2, 3, None), (5, 5, None), (8, 3, None), (3, 8, None),
          (6, 5, 2), (4, 6, 1), (3, 4, 0)]


def test_no_forbidden_imports():
    require_implemented(a4)
    assert forbidden_usages(a4) == [], "forbidden imports/usages found"


def _matrix(m, n, rank, seed=0):
    if rank == 0:
        return np.zeros((m, n))
    return random_matrix(m, n, rank=rank, seed=seed + 31 * m + n)


def _check_svd(U, s, Vt, A, full, ortho_tol, rec_tol):
    m, n = A.shape
    k = min(m, n)
    s = np.asarray(s)
    assert s.shape == (k,), f"s must have shape ({k},), got {s.shape}"
    if full:
        assert U.shape == (m, m) and Vt.shape == (n, n), f"full shapes wrong: {U.shape}, {Vt.shape}"
    else:
        assert U.shape == (m, k) and Vt.shape == (k, n), f"reduced shapes wrong: {U.shape}, {Vt.shape}"
    assert np.all(s >= 0), "singular values must be non-negative"
    assert np.all(np.diff(s) <= 1e-12 * max(1.0, s[0] if k else 0)), "singular values must be decreasing"
    assert_orthonormal_columns(U, ortho_tol, "U")
    assert_orthonormal_columns(Vt.T, ortho_tol, "V")
    rec = (U[:, :k] * s) @ Vt[:k, :]
    assert max_abs(rec - A) <= rec_tol * max(1.0, max_abs(A)), "U Sigma V^T does not reproduce A"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("A, expected", [
    ([[3, 0], [4, 5]], [3 * np.sqrt(5), np.sqrt(5)]),
    ([[3, 2, 2], [2, 3, -2]], [5, 3]),
    ([[1, 1], [1, 1], [1, 1]], [np.sqrt(6), 0]),
    ([[1, 1], [0, 1], [1, 0]], [np.sqrt(3), 1]),
    ([[-3, 0], [0, 2]], [3, 2]),
])
def test_textbook_singular_values(method, A, expected):
    U, s, Vt = getattr(a4, method)(np.array(A, dtype=float))
    assert np.allclose(s, expected, atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("m, n, rank", SHAPES)
def test_svd_properties(method, full, m, n, rank):
    A = _matrix(m, n, rank)
    U, s, Vt = getattr(a4, method)(A, full_matrices=full)
    tol = 1e-12 if method == "svd_jacobi" else 1e-8
    _check_svd(U, s, Vt, A, full, ortho_tol=tol, rec_tol=tol)
    assert np.allclose(s, ref_svd(A, compute_uv=False), atol=1e-7 * max(1.0, max_abs(A)))


@pytest.mark.parametrize("method", METHODS)
def test_zero_singular_values_are_exactly_zero(method):
    A = _matrix(6, 5, 2)
    s = getattr(a4, method)(A)[1]
    assert np.all(s[2:] == 0.0), f"expected exact zeros after the rank, got {s}"


def test_wide_example_full_vectors():
    A = np.array([[3, 2, 2], [2, 3, -2]], dtype=float)
    U, s, Vt = a4.svd_jacobi(A, full_matrices=True)
    v3 = Vt[2]
    assert abs(abs(v3 @ np.array([2, -2, -1]) / 3) - 1) < 1e-12, "third right singular vector should span N(A)"
    assert max_abs(A @ v3) < 1e-12


def test_svd_jacobi_lauchli_accuracy():
    eps = 1e-8
    L = np.array([[1, 1], [eps, 0], [0, eps]])
    s = a4.svd_jacobi(L)[1]
    assert s[0] == pytest.approx(np.sqrt(2 + eps ** 2), rel=1e-12)
    assert s[1] == pytest.approx(eps, rel=1e-6), "one-sided Jacobi must resolve sigma_2 = 1e-8"


def test_svd_jacobi_graded_matrix():
    A = random_matrix(12, 6, kappa=1e10, seed=5)
    s = a4.svd_jacobi(A)[1]
    expected = np.logspace(0, -10, 6)
    assert np.allclose(s / expected, 1.0, atol=1e-4), "small singular values lost accuracy"


@pytest.mark.parametrize("method", METHODS)
def test_repeated_singular_values(method):
    Q = _orth(4)
    A = 2.0 * Q
    U, s, Vt = getattr(a4, method)(A)
    assert np.allclose(s, 2.0)
    _check_svd(U, s, Vt, A, False, 1e-10, 1e-10)


def _orth(n):
    from _helpers import ref_qr
    return ref_qr(np.random.default_rng(n).standard_normal((n, n)))[0]


# ---------------- helpers ----------------
@pytest.mark.parametrize("m, n, rank", SHAPES)
def test_singular_values(m, n, rank):
    A = _matrix(m, n, rank)
    s = a4.singular_values(A)
    assert np.allclose(s, ref_svd(A, compute_uv=False), atol=1e-12)


@pytest.mark.parametrize("m, n", [(4, 2), (2, 4), (3, 3)])
def test_svd_reconstruct_full_and_reduced(m, n):
    A = random_matrix(m, n, seed=4)
    for full in (False, True):
        U, s, Vt = ref_svd(A, full_matrices=full)
        assert max_abs(a4.svd_reconstruct(U, s, Vt) - A) < 1e-12


@pytest.mark.parametrize("A, r", [
    ([[1, 2], [2, 4], [3, 6]], 1),
    (np.eye(5), 5),
    (np.zeros((3, 4)), 0),
    ([[3, 2, 2], [2, 3, -2]], 2),
])
def test_matrix_rank_examples(A, r):
    assert a4.matrix_rank(A) == r


def test_matrix_rank_random_and_tol():
    A = _matrix(7, 6, 4)
    assert a4.matrix_rank(A) == ref_matrix_rank(A) == 4
    D = np.diag([1.0, 1e-6, 1e-12])
    assert a4.matrix_rank(D) == 3
    assert a4.matrix_rank(D, tol=1e-3) == 1


@pytest.mark.parametrize("m, n, rank", [(3, 3, 2), (4, 6, 1), (6, 3, None), (3, 6, None), (3, 4, 0)])
def test_four_subspaces(m, n, rank):
    A = _matrix(m, n, rank)
    r = ref_matrix_rank(A)
    sub = a4.four_subspaces(A)
    assert set(sub) == {"row", "null", "column", "left_null"}
    assert sub["row"].shape == (n, r) and sub["null"].shape == (n, n - r)
    assert sub["column"].shape == (m, r) and sub["left_null"].shape == (m, m - r)
    for key, M in sub.items():
        assert_orthonormal_columns(M, 1e-10, key)
    assert max_abs(A @ sub["null"]) < 1e-10, "A must map the null space to 0"
    assert max_abs(A.T @ sub["left_null"]) < 1e-10, "A^T must map the left null space to 0"
    assert max_abs(sub["row"].T @ sub["null"]) < 1e-10, "row space must be orthogonal to null space"
    C = sub["column"]
    assert max_abs(C @ (C.T @ A) - A) < 1e-10, "the column basis must span C(A)"
    R = sub["row"]
    assert max_abs((A @ R) @ R.T - A) < 1e-10, "the row basis must span C(A^T)"
