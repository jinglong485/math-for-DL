"""Tests for Assignment 6 -- applications.   Run:  pytest tests/test_a6_applications.py"""
import numpy as np
import pytest

from svd_assignments import a6_applications as a6
from _helpers import (forbidden_usages, require_implemented, random_matrix, max_abs, ref_svd, ref_norm, ref_cond,
                      ref_pinv, ref_solve, ref_eigvalsh, assert_orthonormal_columns, assert_parallel)


def test_no_forbidden_imports():
    require_implemented(a6)
    assert forbidden_usages(a6) == [], "forbidden imports/usages found"


# ---------------- norms and condition number ----------------
def test_norms_textbook():
    A = [[3, 0], [4, 5]]
    assert a6.spectral_norm(A) == pytest.approx(3 * np.sqrt(5))
    assert a6.frobenius_norm(A) == pytest.approx(np.sqrt(50))
    assert a6.nuclear_norm(A) == pytest.approx(4 * np.sqrt(5))


@pytest.mark.parametrize("m, n", [(4, 3), (3, 6), (5, 5)])
def test_norms_random(m, n):
    A = random_matrix(m, n, seed=m * n)
    assert a6.spectral_norm(A) == pytest.approx(ref_norm(A, 2), rel=1e-12)
    assert a6.frobenius_norm(A) == pytest.approx(ref_norm(A, "fro"), rel=1e-12)
    assert a6.nuclear_norm(A) == pytest.approx(ref_norm(A, "nuc"), rel=1e-12)


def test_condition_number():
    assert a6.condition_number([[3, 0], [4, 5]]) == pytest.approx(3.0)
    assert a6.condition_number([[1, 1], [1, 1.0001]]) == pytest.approx(40002.0, rel=1e-6)
    A = random_matrix(5, 4, seed=1)
    assert a6.condition_number(A) == pytest.approx(ref_cond(A), rel=1e-10)
    assert a6.condition_number([[1, 2], [2, 4], [3, 6]]) == pytest.approx(1.0)   # rank 1: sigma_1 / sigma_1
    assert a6.condition_number(np.zeros((2, 2))) == np.inf


# ---------------- low-rank approximation ----------------
def test_low_rank_textbook():
    A = np.array([[3, 2, 2], [2, 3, -2]], dtype=float)
    assert np.allclose(a6.low_rank_approx(A, 1), [[2.5, 2.5, 0], [2.5, 2.5, 0]])
    assert np.allclose(a6.low_rank_approx([[3, 0], [4, 5]], 1), 1.5 * np.array([[1, 1], [3, 3]]))


@pytest.mark.parametrize("k", [0, 1, 3, 5, 9])
def test_low_rank_eckart_young(k):
    A = random_matrix(8, 5, seed=21)
    Ak = a6.low_rank_approx(A, k)
    s = ref_svd(A, compute_uv=False)
    assert Ak.shape == A.shape
    expected_2 = s[k] if k < 5 else 0.0
    expected_f = np.sqrt(np.sum(s[k:] ** 2))
    assert ref_norm(A - Ak, 2) == pytest.approx(expected_2, abs=1e-10)
    assert ref_norm(A - Ak, "fro") == pytest.approx(expected_f, abs=1e-10)


def test_low_rank_negative_k():
    with pytest.raises(ValueError):
        a6.low_rank_approx(np.eye(3), -1)


def test_eckart_young_error():
    s = [10, 6, 3, 1]
    assert a6.eckart_young_error(s, 2, "2") == pytest.approx(3.0)
    assert a6.eckart_young_error(s, 2, "fro") == pytest.approx(np.sqrt(10))
    assert a6.eckart_young_error(s, 0, "fro") == pytest.approx(np.sqrt(146))
    assert a6.eckart_young_error(s, 4, "2") == 0.0 and a6.eckart_young_error(s, 7, "fro") == 0.0
    with pytest.raises(ValueError):
        a6.eckart_young_error(s, 1, "nuc")


def test_compression_ratio():
    assert a6.compression_ratio(800, 600, 40) == pytest.approx(56040 / 480000)
    assert a6.compression_ratio(1000, 1000, 50) == pytest.approx(0.10005)


# ---------------- regularisation ----------------
def test_truncated_svd_solve_textbook():
    A = np.diag([1.0, 1e-3])
    b = np.array([1.01, 0.011])
    assert np.allclose(a6.truncated_svd_solve(A, b, 1), [1.01, 0.0])
    assert np.allclose(a6.truncated_svd_solve(A, b, 2), [1.01, 11.0])


def test_truncated_svd_solve_full_rank_equals_pinv():
    A = random_matrix(7, 4, seed=5)
    b = np.random.default_rng(5).standard_normal(7)
    assert np.allclose(a6.truncated_svd_solve(A, b, 4), ref_pinv(A) @ b)
    for bad_k in (0, 5):
        with pytest.raises(ValueError):
            a6.truncated_svd_solve(A, b, bad_k)


def test_ridge_textbook():
    assert np.allclose(a6.ridge_solve([[1, 2]], [5], 5.0), [0.5, 1.0])
    assert np.allclose(a6.ridge_solve(np.diag([1.0, 0.1]), [1.0, 1.0], 0.01), [1 / 1.01, 5.0])


@pytest.mark.parametrize("lam", [1e-3, 0.1, 10.0])
def test_ridge_matches_regularised_normal_equations(lam):
    A = random_matrix(9, 4, seed=7)
    b = np.random.default_rng(7).standard_normal(9)
    expected = ref_solve(A.T @ A + lam * np.eye(4), A.T @ b)
    assert np.allclose(a6.ridge_solve(A, b, lam), expected, atol=1e-10)


def test_ridge_limits():
    A = random_matrix(5, 6, rank=3, seed=2)
    b = np.random.default_rng(2).standard_normal(5)
    assert np.allclose(a6.ridge_solve(A, b, 0.0), ref_pinv(A) @ b, atol=1e-10)
    norms = [np.sqrt(np.sum(a6.ridge_solve(A, b, l) ** 2)) for l in (0.0, 0.1, 1.0, 10.0)]
    assert all(x > y for x, y in zip(norms, norms[1:])), "||x_lambda|| must decrease as lambda grows"
    with pytest.raises(ValueError):
        a6.ridge_solve(A, b, -1.0)


# ---------------- PCA ----------------
def test_pca_textbook():
    X = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1], [0, 0]], dtype=float)
    comps, var, scores, mean = a6.pca(X, 2)
    assert comps.shape == (2, 2) and var.shape == (2,) and scores.shape == (5, 2)
    assert np.allclose(mean, 0)
    assert np.allclose(var, [4.5, 0.5])
    assert_parallel(comps[0], [1, 1])
    assert np.allclose(np.abs(scores[:, 0]), [2.1213203, 2.1213203, 2.1213203, 2.1213203, 0], atol=1e-6)


def test_pca_random_cloud():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)) * [3.0, 2.0, 1.0, 0.5] + [1, 2, 3, 4]
    comps, var, scores, mean = a6.pca(X, 3)
    C = np.cov(X, rowvar=False)
    assert np.allclose(mean, X.mean(axis=0))
    assert np.allclose(var, np.sort(ref_eigvalsh(C))[::-1][:3], rtol=1e-10)
    assert_orthonormal_columns(comps.T, 1e-12, "components")
    assert np.allclose(scores, (X - mean) @ comps.T)
