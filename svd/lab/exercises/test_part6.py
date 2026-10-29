"""
test_part6.py -- Tests for Part 6: Least Squares and Regularisation.

Run with:  python test_part6.py
"""
import numpy as np
import sys

# ── import student code ──────────────────────────────────────────────
try:
    from part6_least_squares import (cholesky, lstsq_normal, lstsq_qr,
                                      lstsq_svd, tsvd_solve, ridge_solve,
                                      vandermonde)
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure part6_least_squares.py is in the same directory and all dependencies are met.")
    sys.exit(1)


passed = 0
failed = 0
total = 0


def run_test(name, fn):
    global passed, failed, total
    total += 1
    try:
        errors = fn()
        if errors:
            print(f"  FAIL  {name}")
            for e in errors:
                print(f"        {e}")
            failed += 1
        else:
            print(f"  PASS  {name}")
            passed += 1
    except NotImplementedError:
        print(f"  SKIP  {name}  (not yet implemented)")
        failed += 1
    except Exception as e:
        print(f"  FAIL  {name}  (exception: {e})")
        failed += 1


# ── Test data ────────────────────────────────────────────────────────
# A well-conditioned overdetermined system for least squares
A_ls = np.array([[1.0, 1.0],
                  [1.0, 2.0],
                  [1.0, 3.0],
                  [1.0, 4.0]])
b_ls = np.array([2.0, 3.1, 3.9, 5.1])
x_ref = np.linalg.lstsq(A_ls, b_ls, rcond=None)[0]

# Symmetric positive definite for Cholesky
S_spd = np.array([[4.0, 2.0, 1.0],
                   [2.0, 5.0, 3.0],
                   [1.0, 3.0, 6.0]])


# ── Tests for cholesky ──────────────────────────────────────────────
print("\n=== Part 6: Least Squares ===\n")
print("--- cholesky ---")


def test_cholesky_basic():
    errors = []
    L = cholesky(S_spd)
    # L should be lower triangular
    if not np.allclose(L, np.tril(L)):
        errors.append("L is not lower triangular")
    # L L^T should reconstruct S
    err = np.max(np.abs(L @ L.T - S_spd))
    if err > 1e-10:
        errors.append(f"L L^T != S, error = {err:.2e}")
    return errors


def test_cholesky_matches_numpy():
    errors = []
    L = cholesky(S_spd)
    L_ref = np.linalg.cholesky(S_spd)
    err = np.max(np.abs(L - L_ref))
    if err > 1e-10:
        errors.append(f"max diff from numpy = {err:.2e}")
    return errors


def test_cholesky_not_pd():
    errors = []
    S_bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # not positive definite
    try:
        cholesky(S_bad)
        errors.append("should have raised ValueError for non-PD matrix")
    except ValueError:
        pass
    return errors


run_test("cholesky: reconstruction L L^T = S", test_cholesky_basic)
run_test("cholesky: matches numpy.linalg.cholesky", test_cholesky_matches_numpy)
run_test("cholesky: raises ValueError for non-PD", test_cholesky_not_pd)


# ── Tests for lstsq_normal ──────────────────────────────────────────
print("\n--- lstsq_normal ---")


def test_lstsq_normal():
    errors = []
    x = lstsq_normal(A_ls, b_ls)
    err = np.max(np.abs(x.ravel() - x_ref))
    if err > 1e-8:
        errors.append(f"max error = {err:.2e}")
        errors.append(f"  yours:  {x.ravel()}")
        errors.append(f"  numpy:  {x_ref}")
    return errors


run_test("lstsq_normal: overdetermined system", test_lstsq_normal)


# ── Tests for lstsq_qr ──────────────────────────────────────────────
print("\n--- lstsq_qr ---")


def test_lstsq_qr():
    errors = []
    x = lstsq_qr(A_ls, b_ls)
    err = np.max(np.abs(x.ravel() - x_ref))
    if err > 1e-8:
        errors.append(f"max error = {err:.2e}")
        errors.append(f"  yours:  {x.ravel()}")
        errors.append(f"  numpy:  {x_ref}")
    return errors


run_test("lstsq_qr: overdetermined system", test_lstsq_qr)


# ── Tests for lstsq_svd ─────────────────────────────────────────────
print("\n--- lstsq_svd ---")


def test_lstsq_svd():
    errors = []
    x = lstsq_svd(A_ls, b_ls)
    err = np.max(np.abs(x.ravel() - x_ref))
    if err > 1e-8:
        errors.append(f"max error = {err:.2e}")
    return errors


run_test("lstsq_svd: overdetermined system", test_lstsq_svd)


# ── Tests for tsvd_solve ─────────────────────────────────────────────
print("\n--- tsvd_solve ---")


def test_tsvd_full_rank():
    errors = []
    x = tsvd_solve(A_ls, b_ls, k=2)
    err = np.max(np.abs(x.ravel() - x_ref))
    if err > 1e-8:
        errors.append(f"full-rank TSVD error = {err:.2e} (should match lstsq)")
    return errors


def test_tsvd_truncated():
    errors = []
    # With k=1, we keep only the largest singular value: should give a different result
    x_k1 = tsvd_solve(A_ls, b_ls, k=1)
    x_k2 = tsvd_solve(A_ls, b_ls, k=2)
    if np.allclose(x_k1.ravel(), x_k2.ravel(), atol=1e-6):
        errors.append("k=1 and k=2 gave the same answer — truncation is not working")
    # k=1 result should still be a reasonable vector
    if x_k1.ravel().shape[0] != 2:
        errors.append(f"expected 2-element solution, got shape {x_k1.shape}")
    return errors


run_test("tsvd_solve: k = full rank matches lstsq", test_tsvd_full_rank)
run_test("tsvd_solve: k=1 differs from k=2 (truncation works)", test_tsvd_truncated)


# ── Tests for ridge_solve ────────────────────────────────────────────
print("\n--- ridge_solve ---")


def test_ridge_zero_lambda():
    errors = []
    x = ridge_solve(A_ls, b_ls, lam=0.0)
    err = np.max(np.abs(x.ravel() - x_ref))
    if err > 1e-6:
        errors.append(f"ridge with lambda=0 should match lstsq, error = {err:.2e}")
    return errors


def test_ridge_large_lambda():
    errors = []
    x_small = ridge_solve(A_ls, b_ls, lam=0.01)
    x_large = ridge_solve(A_ls, b_ls, lam=100.0)
    # Larger lambda should shrink the solution toward zero
    if np.linalg.norm(x_large) >= np.linalg.norm(x_small):
        errors.append("larger lambda should produce a smaller solution norm")
    return errors


run_test("ridge_solve: lambda=0 matches lstsq", test_ridge_zero_lambda)
run_test("ridge_solve: larger lambda shrinks solution", test_ridge_large_lambda)


# ── Tests for vandermonde ────────────────────────────────────────────
print("\n--- vandermonde ---")


def test_vandermonde():
    errors = []
    t = np.array([1.0, 2.0, 3.0])
    V = vandermonde(t, 2)
    V_expected = np.array([[1.0, 1.0, 1.0],
                            [1.0, 2.0, 4.0],
                            [1.0, 3.0, 9.0]])
    err = np.max(np.abs(V - V_expected))
    if err > 1e-15:
        errors.append(f"vandermonde(t, 2) error = {err:.2e}")
        errors.append(f"  expected:\n{V_expected}")
        errors.append(f"  got:\n{V}")
    if V.shape != (3, 3):
        errors.append(f"expected shape (3, 3), got {V.shape}")
    return errors


def test_vandermonde_degree0():
    errors = []
    t = np.array([5.0, 10.0])
    V = vandermonde(t, 0)
    if V.shape != (2, 1):
        errors.append(f"degree 0 should give shape (2, 1), got {V.shape}")
    if not np.allclose(V, np.ones((2, 1))):
        errors.append(f"degree 0 should be all ones")
    return errors


run_test("vandermonde: degree 2 with [1, 2, 3]", test_vandermonde)
run_test("vandermonde: degree 0", test_vandermonde_degree0)


# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Part 6 Results: Passed {passed}/{total} tests")
print(f"{'='*50}")
if passed == total:
    print("🎉  All Part 6 tests passed!  Least squares mastered!")
    print("    Next up: Part 7 (applications — the grand finale).")
else:
    print(f"💪  Keep going — {total - passed} test(s) remaining.")
print()
