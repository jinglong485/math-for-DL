#!/usr/bin/env python3
"""
Test suite for Part 3 -- Eigenvalues of Symmetric Matrices.

Run:  python test_part3.py
"""
import sys
import numpy as np
import traceback

# ---------------------------------------------------------------------------
# Import the student's code
# ---------------------------------------------------------------------------
try:
    from part3_eigenvalues import (
        rayleigh_quotient, power_iteration, eigs_by_deflation, jacobi_eigh,
    )
    from part1_vectors_matrices import norm, outer
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Make sure part1, part2, and part3 files are in the same directory.")
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


# ---------------------------------------------------------------------------
# Test data: symmetric matrix with known eigenvalues
# ---------------------------------------------------------------------------
# S = V diag(6, 3, 1) V^T  where V is an orthogonal matrix
# This gives us known eigenvalues 6, 3, 1 to check against
np.random.seed(42)
_V, _ = np.linalg.qr(np.random.randn(3, 3))
S_known = _V @ np.diag([6.0, 3.0, 1.0]) @ _V.T
S_known = (S_known + S_known.T) / 2  # ensure perfect symmetry

# A simpler 2x2 case
S_2x2 = np.array([[5, 2],
                   [2, 2]], dtype=float)
# eigenvalues: (7 ± sqrt(9+16))/2 ... let numpy tell us
eig_2x2 = np.sort(np.linalg.eigvalsh(S_2x2))[::-1]  # sorted descending

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rayleigh_quotient():
    # For an eigenvector, R(v) should equal the eigenvalue
    evals, evecs = np.linalg.eigh(S_known)
    for i in range(3):
        rq = rayleigh_quotient(S_known, evecs[:, i])
        assert abs(rq - evals[i]) < 1e-10, \
            f"R(v_{i}) = {rq}, expected eigenvalue {evals[i]}"
    # For a general vector, R should be in [lambda_min, lambda_max]
    x = np.array([1.0, 1.0, 1.0])
    rq = rayleigh_quotient(S_known, x)
    assert 1.0 - 1e-8 <= rq <= 6.0 + 1e-8, \
        f"R(x) = {rq} should be in [1, 6]"


def test_power_iteration_basic():
    lam, v, history = power_iteration(S_known)
    # Should find the largest eigenvalue (6.0)
    assert abs(lam - 6.0) < 1e-8, \
        f"Dominant eigenvalue = {lam}, expected 6.0"
    # v should be a unit vector
    assert abs(norm(v) - 1.0) < 1e-10, \
        f"Eigenvector should be unit, got norm = {norm(v)}"
    # S v ≈ lam v
    residual = norm(S_known @ v - lam * v)
    assert residual < 1e-8, \
        f"||Sv - lam v|| = {residual}, should be ~0"


def test_power_iteration_2x2():
    lam, v, _ = power_iteration(S_2x2)
    assert abs(lam - eig_2x2[0]) < 1e-8, \
        f"Dominant eigenvalue of 2x2 = {lam}, expected {eig_2x2[0]:.6f}"


def test_power_iteration_convergence():
    """History should show convergence."""
    _, _, history = power_iteration(S_known)
    assert len(history) > 0, "History should not be empty"
    # Last entry should be close to 6.0
    assert abs(history[-1] - 6.0) < 1e-8, \
        f"Final Rayleigh quotient = {history[-1]}, expected ~6.0"


def test_eigs_by_deflation():
    lams, vecs = eigs_by_deflation(S_known, 3)
    expected = np.array([6.0, 3.0, 1.0])
    # Sort both for comparison
    lams_sorted = np.sort(lams)[::-1]
    assert np.allclose(lams_sorted, expected, atol=1e-6), \
        f"Eigenvalues = {lams_sorted}, expected {expected}"
    # Each column should be an eigenvector of the ORIGINAL S_known
    for i in range(3):
        v = vecs[:, i]
        lam = lams[i]
        residual = norm(S_known @ v - lam * v)
        assert residual < 1e-5, \
            f"||Sv_{i} - lam_{i} v_{i}|| = {residual}, expected ~0"


def test_eigs_by_deflation_2x2():
    lams, vecs = eigs_by_deflation(S_2x2, 2)
    lams_sorted = np.sort(lams)[::-1]
    assert np.allclose(lams_sorted, eig_2x2, atol=1e-6), \
        f"2x2 eigenvalues = {lams_sorted}, expected {eig_2x2}"


def test_jacobi_eigh_basic():
    evals, V = jacobi_eigh(S_known)
    expected = np.array([6.0, 3.0, 1.0])
    assert np.allclose(evals, expected, atol=1e-10), \
        f"Jacobi eigenvalues = {evals}, expected {expected}"
    # V should be orthogonal
    assert np.allclose(V.T @ V, np.eye(3), atol=1e-10), \
        f"V should be orthogonal. V^T V =\n{V.T @ V}"
    # S = V diag(evals) V^T
    reconstructed = V @ np.diag(evals) @ V.T
    assert np.allclose(reconstructed, S_known, atol=1e-10), \
        f"V diag(evals) V^T should reconstruct S"


def test_jacobi_eigh_2x2():
    evals, V = jacobi_eigh(S_2x2)
    assert np.allclose(evals, eig_2x2, atol=1e-10), \
        f"2x2 Jacobi eigenvalues = {evals}, expected {eig_2x2}"
    assert np.allclose(V.T @ V, np.eye(2), atol=1e-10), \
        "V should be orthogonal"


def test_jacobi_eigh_identity():
    """Identity matrix should have all eigenvalues = 1."""
    evals, V = jacobi_eigh(np.eye(3))
    assert np.allclose(evals, [1, 1, 1], atol=1e-10), \
        f"eigenvalues of I should be [1,1,1], got {evals}"


def test_jacobi_eigh_history():
    evals, V, history = jacobi_eigh(S_known, return_history=True)
    assert isinstance(history, list), "history should be a list"
    assert len(history) > 0, "history should be non-empty"
    # off-diagonal norm should decrease
    if len(history) >= 2:
        assert history[-1] <= history[0] + 1e-12, \
            f"off-diagonal norm should decrease: first={history[0]}, last={history[-1]}"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Part 3: Eigenvalues of Symmetric Matrices")
    print("=" * 60)

    tests = [
        ("rayleigh_quotient", test_rayleigh_quotient),
        ("power_iteration (3x3)", test_power_iteration_basic),
        ("power_iteration (2x2)", test_power_iteration_2x2),
        ("power_iteration (convergence)", test_power_iteration_convergence),
        ("eigs_by_deflation (3x3)", test_eigs_by_deflation),
        ("eigs_by_deflation (2x2)", test_eigs_by_deflation_2x2),
        ("jacobi_eigh (3x3)", test_jacobi_eigh_basic),
        ("jacobi_eigh (2x2)", test_jacobi_eigh_2x2),
        ("jacobi_eigh (identity)", test_jacobi_eigh_identity),
        ("jacobi_eigh (history)", test_jacobi_eigh_history),
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
        print("\n🎉 All tests passed! You've mastered eigenvalue algorithms!")
    elif n_fail == 0 and n_skip > 0:
        print(f"\n📝 {n_skip} function(s) still need implementing. Keep going!")
    else:
        print(f"\n🔧 {n_fail} test(s) failed. Check the messages above for hints.")
    print("=" * 60)
