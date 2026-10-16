"""Tests for Assignment 1 -- vectors and matrices.   Run:  pytest tests/test_a1_vectors.py"""
import numpy as np
import pytest

from svd_assignments import a1_vectors as a1
from _helpers import forbidden_usages, require_implemented, builtin_product_usages, random_matrix, ref_matrix_rank


def test_no_forbidden_imports():
    require_implemented(a1)
    assert forbidden_usages(a1) == [], "forbidden imports/usages found"


@pytest.mark.parametrize("func", ["dot", "matmul"])
def test_no_builtin_products(func):
    require_implemented(getattr(a1, func))
    problems = builtin_product_usages(getattr(a1, func))
    assert problems == [], f"{func} must be written with loops/element-wise products, found: {problems}"


# ---------------- dot ----------------
def test_dot_textbook_example():
    result = a1.dot([1, 2, 2], [2, 0, 1])
    assert isinstance(result, float)
    assert result == pytest.approx(4.0)


@pytest.mark.parametrize("n", [1, 3, 50])
def test_dot_random(n):
    rng = np.random.default_rng(n)
    u, v = rng.standard_normal(n), rng.standard_normal(n)
    assert a1.dot(u, v) == pytest.approx(float(np.sum(u * v)), rel=1e-12, abs=1e-12)


def test_dot_length_mismatch():
    with pytest.raises(ValueError):
        a1.dot([1, 2, 3], [1, 2])


# ---------------- norm ----------------
@pytest.mark.parametrize("x, expected", [([3, -4, 0], (7, 5, 4)), ([3, -4, 12], (19, 13, 12))])
def test_norm_textbook(x, expected):
    got = (a1.norm(x, 1), a1.norm(x, 2), a1.norm(x, np.inf))
    assert got == pytest.approx(expected)


def test_norm_default_is_2():
    assert a1.norm([1, 2, 2]) == pytest.approx(3.0)


def test_norm_no_overflow_or_underflow():
    assert a1.norm([3e200, 4e200]) == pytest.approx(5e200, rel=1e-12)
    assert a1.norm([3e-200, 4e-200]) == pytest.approx(5e-200, rel=1e-12)


def test_norm_zero_vector_and_bad_p():
    assert a1.norm(np.zeros(4)) == 0.0
    with pytest.raises(ValueError):
        a1.norm([1, 2], p=3)


# ---------------- normalize, angle, projection ----------------
def test_normalize():
    v = a1.normalize([1, 2, 2])
    assert np.allclose(v, [1 / 3, 2 / 3, 2 / 3])
    with pytest.raises(ValueError):
        a1.normalize([0, 0, 0])


def test_angle_values():
    assert np.degrees(a1.angle([1, 2, 2], [2, 0, 1])) == pytest.approx(53.3957, abs=1e-4)
    assert a1.angle([1, 0], [0, 3]) == pytest.approx(np.pi / 2)
    assert a1.angle([1, 2], [-2, -4]) == pytest.approx(np.pi)


def test_angle_parallel_is_not_nan():
    x = np.array([0.1, 0.2, 0.3])
    for k in (1.0, 3.0, 7.0, 1e5):
        a = a1.angle(x, k * x)
        assert not np.isnan(a) and a == pytest.approx(0.0, abs=1e-7)


def test_angle_zero_vector():
    with pytest.raises(ValueError):
        a1.angle([0, 0], [1, 1])


def test_project_onto_vector():
    p = a1.project_onto_vector([1, 1, 1], [1, 2, 2])
    assert np.allclose(p, np.array([1, 2, 2]) * 5 / 9)
    e = np.array([1, 1, 1]) - p
    assert abs(e @ np.array([1, 2, 2])) < 1e-12
    with pytest.raises(ValueError):
        a1.project_onto_vector([1, 2], [0, 0])


def test_outer():
    u, v = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])
    O = a1.outer(u, v)
    assert O.shape == (3, 2)
    assert np.allclose(O, np.outer(u, v))


# ---------------- matmul ----------------
@pytest.mark.parametrize("m, n, p", [(1, 1, 1), (2, 3, 4), (5, 1, 5), (3, 3, 3), (1, 6, 1)])
def test_matmul_random(m, n, p):
    A, B = random_matrix(m, n, seed=1), random_matrix(n, p, seed=2)
    C = a1.matmul(A, B)
    assert C.shape == (m, p)
    assert np.allclose(C, np.matmul(A, B), atol=1e-12)


def test_matmul_textbook():
    assert np.allclose(a1.matmul([[1, 2], [0, 1]], [[3, 1], [1, 0]]), [[5, 1], [1, 0]])


def test_matmul_shape_mismatch():
    with pytest.raises(ValueError):
        a1.matmul(np.ones((2, 3)), np.ones((2, 3)))


# ---------------- rank by elimination ----------------
@pytest.mark.parametrize("A, r", [
    ([[1, 2, 3], [2, 4, 7], [3, 6, 10]], 2),           # textbook Worked Example 2.3
    ([[1, 2, 0, 1], [2, 4, 1, 4], [3, 6, 1, 5]], 2),   # textbook Exercise 2.3
    (np.eye(4), 4),
    (np.zeros((3, 5)), 0),
    ([[0, 0], [0, 3]], 1),                            # needs a row swap
    ([[1e-14, 1], [1, 1]], 2),                        # needs pivoting to be accurate
])
def test_rank_gaussian_examples(A, r):
    assert a1.rank_gaussian(A) == r


@pytest.mark.parametrize("m, n, k", [(6, 5, 3), (4, 7, 2), (8, 8, 8)])
def test_rank_gaussian_random(m, n, k):
    A = random_matrix(m, n, rank=k, seed=m * n)
    assert a1.rank_gaussian(A) == ref_matrix_rank(A) == k
