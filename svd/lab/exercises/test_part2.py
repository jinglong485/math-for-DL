#!/usr/bin/env python3
"""
Test suite for Part 2 -- Orthogonalisation and QR Factorisation.

Run:  python test_part2.py
"""
import sys
import numpy as np
import traceback

# ---------------------------------------------------------------------------
# Import the student's code
# ---------------------------------------------------------------------------
try:
    from part2_orthogonalisation_qr import (
        gram_schmidt_classical, gram_schmidt_modified,
        householder_qr, back_substitution, forward_substitution,
        solve, projection_matrix, complete_orthonormal_basis,
    )
    from part1_vectors_matrices import norm, _as_float_matrix
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Make sure part1 and part2 files are in the same directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
passed, failed, errors = [], [], []


def run_test(name, fn):
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


def is_orthonormal(Q, tol=1e-10):
    """Check that Q^T Q ≈ I."""
    QtQ = Q.T @ Q
    return np.allclose(QtQ, np.eye(QtQ.shape[0]), atol=tol)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
A33 = np.array([[2, 1, 1],
                [4, 3, 3],
                [8, 7, 9]], dtype=float)

A43 = np.array([[1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]], dtype=float)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gram_schmidt_classical():
    Q, R = gram_schmidt_classical(A43)
    assert Q.shape == (4, 3), f"Q shape should be (4,3), got {Q.shape}"
    assert R.shape == (3, 3), f"R shape should be (3,3), got {R.shape}"
    assert is_orthonormal(Q), \
        f"Q should have orthonormal columns. Q^T Q =\n{Q.T @ Q}"
    # Check factorisation
    assert np.allclose(Q @ R, A43, atol=1e-10), \
        f"Q @ R should equal A.\nQ@R:\n{Q @ R}\nA:\n{A43}"
    # R should be upper triangular
    assert np.allclose(np.tril(R, -1), 0, atol=1e-10), \
        "R should be upper triangular"


def test_gram_schmidt_modified():
    Q, R = gram_schmidt_modified(A43)
    assert Q.shape == (4, 3), f"Q shape should be (4,3), got {Q.shape}"
    assert is_orthonormal(Q), \
        f"Q should have orthonormal columns. Q^T Q =\n{Q.T @ Q}"
    assert np.allclose(Q @ R, A43, atol=1e-10), \
        f"Q @ R should equal A"
    assert np.allclose(np.tril(R, -1), 0, atol=1e-10), \
        "R should be upper triangular"


def test_householder_qr():
    Q, R = householder_qr(A33)
    assert Q.shape == (3, 3), f"Q shape should be (3,3), got {Q.shape}"
    assert R.shape == (3, 3), f"R shape should be (3,3), got {R.shape}"
    # Q should be orthogonal: Q^T Q = I
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10), \
        f"Q should be orthogonal. Q^T Q =\n{Q.T @ Q}"
    assert np.allclose(Q @ Q.T, np.eye(3), atol=1e-10), \
        f"Q should be orthogonal. Q Q^T =\n{Q @ Q.T}"
    # Check factorisation
    assert np.allclose(Q @ R, A33, atol=1e-10), \
        f"Q @ R should equal A.\nQ@R:\n{Q @ R}\nA:\n{A33}"
    # R upper triangular
    assert np.allclose(np.tril(R, -1), 0, atol=1e-10), \
        "R should be upper triangular"


def test_householder_qr_tall():
    """Householder on a tall matrix (4x3)."""
    Q, R = householder_qr(A43)
    assert Q.shape == (4, 4), f"Q shape should be (4,4) for full QR, got {Q.shape}"
    assert R.shape == (4, 3), f"R shape should be (4,3), got {R.shape}"
    assert np.allclose(Q @ R, A43, atol=1e-10), "Q @ R should equal A"
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-10), "Q should be orthogonal"


def test_back_substitution():
    U = np.array([[2, 1, 3],
                  [0, 4, 5],
                  [0, 0, 6]], dtype=float)
    b = np.array([14, 29, 18], dtype=float)
    x = back_substitution(U, b)
    assert np.allclose(U @ x, b, atol=1e-10), \
        f"U @ x should equal b.\nU@x = {U @ x}\nb = {b}"


def test_forward_substitution():
    L = np.array([[2, 0, 0],
                  [1, 3, 0],
                  [4, 5, 6]], dtype=float)
    b = np.array([4, 7, 38], dtype=float)
    x = forward_substitution(L, b)
    assert np.allclose(L @ x, b, atol=1e-10), \
        f"L @ x should equal b.\nL@x = {L @ x}\nb = {b}"


def test_solve():
    b = np.array([1, 2, 3], dtype=float)
    x = solve(A33, b)
    expected = np.linalg.solve(A33, b)
    assert np.allclose(x, expected, atol=1e-10), \
        f"solve(A, b) = {x}, expected {expected}"
    # Verify residual
    assert np.allclose(A33 @ x, b, atol=1e-10), \
        f"A @ x should equal b. Residual = {A33 @ x - b}"


def test_projection_matrix():
    # Project onto column space of a 3x2 matrix
    V = np.array([[1, 0],
                  [0, 1],
                  [0, 0]], dtype=float)  # span of e1, e2
    P = projection_matrix(V)
    assert P.shape == (3, 3), f"P shape should be (3,3), got {P.shape}"
    # P should be idempotent: P^2 = P
    assert np.allclose(P @ P, P, atol=1e-10), \
        f"P should be idempotent (P^2 = P)"
    # P should be symmetric
    assert np.allclose(P, P.T, atol=1e-10), \
        f"P should be symmetric"
    # Projecting e3 onto span(e1, e2) should give zero
    e3 = np.array([0, 0, 1.0])
    assert np.allclose(P @ e3, 0, atol=1e-10), \
        f"P @ e3 should be zero for V = [e1, e2]"
    # Projecting e1 should give e1
    e1 = np.array([1, 0, 0.0])
    assert np.allclose(P @ e1, e1, atol=1e-10), \
        f"P @ e1 should be e1 for V = [e1, e2]"


def test_complete_orthonormal_basis():
    # Start with 2 orthonormal columns in R^4
    Q_start = np.array([[1, 0],
                        [0, 1],
                        [0, 0],
                        [0, 0]], dtype=float)
    Q_full = complete_orthonormal_basis(Q_start)
    assert Q_full.shape == (4, 4), \
        f"Should extend to 4x4, got {Q_full.shape}"
    assert np.allclose(Q_full.T @ Q_full, np.eye(4), atol=1e-10), \
        f"Result should be orthogonal. Q^T Q =\n{Q_full.T @ Q_full}"
    # First two columns should match the input
    assert np.allclose(Q_full[:, :2], Q_start, atol=1e-10), \
        "First columns should match the input"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Part 2: Orthogonalisation and QR Factorisation")
    print("=" * 60)

    tests = [
        ("gram_schmidt_classical", test_gram_schmidt_classical),
        ("gram_schmidt_modified", test_gram_schmidt_modified),
        ("householder_qr (square)", test_householder_qr),
        ("householder_qr (tall)", test_householder_qr_tall),
        ("back_substitution", test_back_substitution),
        ("forward_substitution", test_forward_substitution),
        ("solve", test_solve),
        ("projection_matrix", test_projection_matrix),
        ("complete_orthonormal_basis", test_complete_orthonormal_basis),
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
        print("\n🎉 All tests passed! You're ready for Part 3.")
    elif n_fail == 0 and n_skip > 0:
        print(f"\n📝 {n_skip} function(s) still need implementing. Keep going!")
    else:
        print(f"\n🔧 {n_fail} test(s) failed. Check the messages above for hints.")
    print("=" * 60)
