#!/usr/bin/env python3
"""
Test suite for Part 1 -- Vectors, Matrices, and Gaussian Elimination.

Run:  python test_part1.py
"""
import sys
import numpy as np
import traceback

# ---------------------------------------------------------------------------
# Import the student's code
# ---------------------------------------------------------------------------
try:
    from part1_vectors_matrices import (
        dot, norm, normalize, angle, project_onto_vector,
        matmul, outer, rref, rank_by_elimination, null_space_by_elimination,
    )
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Make sure part1_vectors_matrices.py is in the same directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
passed, failed, errors = [], [], []


def run_test(name, fn):
    """Run a single test function. Record pass/fail/error."""
    try:
        fn()
        passed.append(name)
        print(f"  ✓ PASS  {name}")
    except NotImplementedError:
        errors.append(name)
        print(f"  ✗ SKIP  {name}  (NotImplementedError — not yet implemented)")
    except AssertionError as e:
        failed.append(name)
        print(f"  ✗ FAIL  {name}:  {e}")
    except Exception as e:
        failed.append(name)
        print(f"  ✗ FAIL  {name}:  unexpected error → {e}")
        traceback.print_exc(limit=2)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
u3 = np.array([1.0, 2.0, 3.0])
v3 = np.array([4.0, -5.0, 6.0])

A33 = np.array([[2, 1, 1],
                [4, 3, 3],
                [8, 7, 9]], dtype=float)

A23 = np.array([[1, 2, 3],
                [4, 5, 6]], dtype=float)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dot():
    result = dot(u3, v3)
    expected = float(np.dot(u3, v3))  # 1*4 + 2*(-5) + 3*6 = 12
    assert abs(result - expected) < 1e-12, \
        f"dot({list(u3)}, {list(v3)}) = {result}, expected {expected}"
    # Also test zero vectors
    z = np.zeros(3)
    assert abs(dot(z, u3)) < 1e-12, "dot with zero vector should be 0"


def test_norm_2():
    result = norm(u3, 2)
    expected = np.linalg.norm(u3, 2)
    assert abs(result - expected) < 1e-12, \
        f"norm({list(u3)}, 2) = {result}, expected {expected}"


def test_norm_1():
    result = norm(v3, 1)
    expected = np.linalg.norm(v3, 1)  # 4 + 5 + 6 = 15
    assert abs(result - expected) < 1e-12, \
        f"norm({list(v3)}, 1) = {result}, expected {expected}"


def test_norm_inf():
    result = norm(v3, np.inf)
    expected = np.linalg.norm(v3, np.inf)  # 6
    assert abs(result - expected) < 1e-12, \
        f"norm({list(v3)}, inf) = {result}, expected {expected}"


def test_norm_overflow():
    """The scaling trick should prevent overflow for huge entries."""
    big = np.array([1e200, 1e200])
    result = norm(big, 2)
    expected = 1e200 * np.sqrt(2)
    assert abs(result - expected) / expected < 1e-10, \
        f"norm with huge entries: {result}, expected ~{expected}"


def test_normalize():
    u_hat = normalize(u3)
    assert abs(np.linalg.norm(u_hat) - 1.0) < 1e-12, \
        f"normalize should produce a unit vector, got norm = {np.linalg.norm(u_hat)}"
    # Check direction
    assert abs(dot(u_hat, u3) - norm(u3)) < 1e-12, \
        "normalize should preserve direction"
    # Check zero vector raises
    try:
        normalize(np.zeros(3))
        assert False, "normalize(zero) should raise ValueError"
    except ValueError:
        pass


def test_angle():
    # Angle between [1,0,0] and [0,1,0] should be pi/2
    result = angle([1, 0, 0], [0, 1, 0])
    assert abs(result - np.pi / 2) < 1e-12, \
        f"angle(e1, e2) = {result}, expected pi/2 = {np.pi/2}"
    # Parallel vectors → angle 0
    result2 = angle([1, 2, 3], [2, 4, 6])
    assert abs(result2) < 1e-10, f"angle of parallel vectors = {result2}, expected 0"


def test_project_onto_vector():
    proj = project_onto_vector(v3, u3)
    # Projection should satisfy: (v - proj) · u = 0
    residual = dot(v3 - proj, u3)
    assert abs(residual) < 1e-10, \
        f"Residual (v - proj) . u = {residual}, expected 0"
    # Projection should be a scalar multiple of u
    ratio = proj / u3
    assert np.allclose(ratio, ratio[0]), \
        f"Projection should be parallel to a, got ratios {ratio}"


def test_matmul():
    C = matmul(A33, A33)
    expected = A33 @ A33
    assert np.allclose(C, expected, atol=1e-10), \
        f"matmul(A,A) doesn't match A@A.\nGot:\n{C}\nExpected:\n{expected}"
    # Non-square
    C2 = matmul(A23, A33)
    expected2 = A23 @ A33
    assert np.allclose(C2, expected2, atol=1e-10), \
        f"matmul(2x3, 3x3) doesn't match"


def test_outer():
    result = outer(u3, v3)
    expected = np.outer(u3, v3)
    assert np.allclose(result, expected, atol=1e-12), \
        f"outer product doesn't match np.outer"
    assert result.shape == (3, 3), f"Shape should be (3,3), got {result.shape}"


def test_rref():
    R, pivots = rref(A33)
    # A33 is invertible (3x3), so should have 3 pivot columns
    assert len(pivots) == 3, \
        f"Expected 3 pivot columns for invertible matrix, got {len(pivots)}: {pivots}"
    # RREF of an invertible matrix should be I
    assert np.allclose(R, np.eye(3), atol=1e-10), \
        f"RREF of invertible 3x3 should be I, got:\n{R}"
    # Rank-deficient matrix
    B = np.array([[1, 2, 3], [2, 4, 6]], dtype=float)
    R2, piv2 = rref(B)
    assert len(piv2) == 1, f"Rank-1 matrix should have 1 pivot, got {len(piv2)}"


def test_rank_by_elimination():
    r1 = rank_by_elimination(A33)
    assert r1 == 3, f"rank of 3x3 invertible matrix = {r1}, expected 3"
    r2 = rank_by_elimination([[1, 2, 3], [2, 4, 6]])
    assert r2 == 1, f"rank of rank-1 matrix = {r2}, expected 1"
    r3 = rank_by_elimination(np.zeros((2, 3)))
    assert r3 == 0, f"rank of zero matrix = {r3}, expected 0"


def test_null_space():
    # N(A) for A = [[1,2,3],[2,4,6]] should be 2-dimensional
    B = np.array([[1, 2, 3], [2, 4, 6]], dtype=float)
    NS = null_space_by_elimination(B)
    assert NS.shape[0] == 3, f"Null space vectors should have length 3"
    assert NS.shape[1] == 2, f"Null space should be 2-dimensional, got {NS.shape[1]}"
    # Check that B @ NS ≈ 0
    resid = B @ NS
    assert np.allclose(resid, 0, atol=1e-10), \
        f"A @ null_space should be zero, got:\n{resid}"
    # Full-rank matrix should have trivial null space
    NS2 = null_space_by_elimination(A33)
    assert NS2.shape[1] == 0, \
        f"Full-rank matrix should have 0-dimensional null space, got {NS2.shape[1]}"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Part 1: Vectors, Matrices, and Gaussian Elimination")
    print("=" * 60)

    tests = [
        ("dot", test_dot),
        ("norm (2-norm)", test_norm_2),
        ("norm (1-norm)", test_norm_1),
        ("norm (inf-norm)", test_norm_inf),
        ("norm (overflow protection)", test_norm_overflow),
        ("normalize", test_normalize),
        ("angle", test_angle),
        ("project_onto_vector", test_project_onto_vector),
        ("matmul", test_matmul),
        ("outer", test_outer),
        ("rref", test_rref),
        ("rank_by_elimination", test_rank_by_elimination),
        ("null_space_by_elimination", test_null_space),
    ]

    for name, fn in tests:
        run_test(name, fn)

    total = len(tests)
    n_pass = len(passed)
    n_fail = len(failed)
    n_skip = len(errors)

    print()
    print("-" * 60)
    print(f"Results: {n_pass}/{total} passed", end="")
    if n_fail:
        print(f", {n_fail} failed", end="")
    if n_skip:
        print(f", {n_skip} not yet implemented", end="")
    print()

    if n_pass == total:
        print("\n🎉 All tests passed! You're ready for Part 2.")
    elif n_fail == 0 and n_skip > 0:
        print(f"\n📝 {n_skip} function(s) still need implementing. Keep going!")
    else:
        print(f"\n🔧 {n_fail} test(s) failed. Check the messages above for hints.")
    print("=" * 60)
