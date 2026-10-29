"""
test_part4.py -- Tests for Part 4: Singular Value Decomposition.

Run with:  python test_part4.py
"""
import numpy as np
import sys

# ── import student code ──────────────────────────────────────────────
try:
    from part4_svd import svd_via_eig, svd_jacobi, svd
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure part4_svd.py is in the same directory and all dependencies are met.")
    sys.exit(1)


def _check_svd(name, U, s, Vt, A, tol=1e-8):
    """Verify U, s, Vt form a valid SVD of A."""
    errors = []
    m, n = A.shape
    k = min(m, n)

    # 1. Reconstruction: U @ diag(s) @ Vt ≈ A
    recon = (U[:, :k] * s[:k]) @ Vt[:k]
    err = np.max(np.abs(recon - A))
    if err > tol:
        errors.append(f"  reconstruction error = {err:.2e} (expected < {tol:.0e})")

    # 2. U has orthonormal columns: U^T U ≈ I
    UtU = U.T @ U
    err = np.max(np.abs(UtU - np.eye(UtU.shape[0])))
    if err > tol:
        errors.append(f"  U^T U - I error = {err:.2e}")

    # 3. Vt has orthonormal rows: Vt Vt^T ≈ I
    VVt = Vt @ Vt.T
    err = np.max(np.abs(VVt - np.eye(VVt.shape[0])))
    if err > tol:
        errors.append(f"  V^T V - I error = {err:.2e}")

    # 4. Singular values match numpy
    s_ref = np.linalg.svd(A, compute_uv=False)
    s_cmp = np.sort(s[:k])[::-1]
    s_ref_cmp = np.sort(s_ref)[::-1]
    min_len = min(len(s_cmp), len(s_ref_cmp))
    err = np.max(np.abs(s_cmp[:min_len] - s_ref_cmp[:min_len]))
    if err > tol:
        errors.append(f"  singular value mismatch = {err:.2e}")
        errors.append(f"    yours:  {s_cmp[:min_len]}")
        errors.append(f"    numpy:  {s_ref_cmp[:min_len]}")

    # 5. Singular values are non-negative and sorted
    if np.any(s < -tol):
        errors.append(f"  negative singular values found: {s[s < 0]}")
    if not np.all(np.diff(s[:k]) <= tol):
        # Check if sorted descending
        if np.any(np.diff(s[:k]) > tol):
            errors.append(f"  singular values not sorted descending")

    return errors


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


# ── Test matrices ────────────────────────────────────────────────────
A_tall = np.array([[3.0, 2.0, 2.0],
                    [2.0, 3.0, -2.0]])

A_square = np.array([[1.0, 2.0],
                      [3.0, 4.0]])

A_wide = np.array([[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0]])

A_rank1 = np.array([[1.0, 2.0],
                     [2.0, 4.0],
                     [3.0, 6.0]])

A_3x3 = np.array([[2.0, -1.0, 0.0],
                   [-1.0, 2.0, -1.0],
                   [0.0, -1.0, 2.0]])


# ── Tests for svd_via_eig ───────────────────────────────────────────
print("\n=== Part 4: SVD ===\n")
print("--- svd_via_eig ---")

def test_eig_tall():
    U, s, Vt = svd_via_eig(A_tall.T)  # 3x2, tall
    return _check_svd("tall", U, s, Vt, A_tall.T)

def test_eig_square():
    U, s, Vt = svd_via_eig(A_square)
    return _check_svd("square", U, s, Vt, A_square)

def test_eig_wide():
    U, s, Vt = svd_via_eig(A_wide)
    return _check_svd("wide", U, s, Vt, A_wide)

def test_eig_rank_deficient():
    U, s, Vt = svd_via_eig(A_rank1)
    errors = _check_svd("rank1", U, s, Vt, A_rank1)
    # Check that only 1 singular value is non-trivially nonzero
    s_ref = np.linalg.svd(A_rank1, compute_uv=False)
    if np.sum(s > 1e-8) != np.sum(s_ref > 1e-8):
        errors.append(f"  rank mismatch: yours has {np.sum(s > 1e-8)} nonzero sv, expected {np.sum(s_ref > 1e-8)}")
    return errors

run_test("svd_via_eig: tall matrix (3x2)", test_eig_tall)
run_test("svd_via_eig: square matrix (2x2)", test_eig_square)
run_test("svd_via_eig: wide matrix (2x4)", test_eig_wide)
run_test("svd_via_eig: rank-deficient matrix (3x2, rank 1)", test_eig_rank_deficient)


# ── Tests for svd_jacobi ────────────────────────────────────────────
print("\n--- svd_jacobi ---")

def test_jacobi_tall():
    U, s, Vt = svd_jacobi(A_tall.T)
    return _check_svd("tall", U, s, Vt, A_tall.T)

def test_jacobi_square():
    U, s, Vt = svd_jacobi(A_3x3)
    return _check_svd("square 3x3", U, s, Vt, A_3x3)

def test_jacobi_wide():
    U, s, Vt = svd_jacobi(A_wide)
    return _check_svd("wide", U, s, Vt, A_wide)

def test_jacobi_identity():
    I3 = np.eye(3)
    U, s, Vt = svd_jacobi(I3)
    errors = _check_svd("identity", U, s, Vt, I3)
    if not np.allclose(s, [1.0, 1.0, 1.0], atol=1e-10):
        errors.append(f"  expected all singular values = 1, got {s}")
    return errors

run_test("svd_jacobi: tall matrix (3x2)", test_jacobi_tall)
run_test("svd_jacobi: symmetric 3x3", test_jacobi_square)
run_test("svd_jacobi: wide matrix (2x4)", test_jacobi_wide)
run_test("svd_jacobi: identity matrix", test_jacobi_identity)


# ── Tests for svd wrapper ───────────────────────────────────────────
print("\n--- svd (wrapper) ---")

def test_svd_wrapper():
    U, s, Vt = svd(A_tall.T)
    return _check_svd("wrapper", U, s, Vt, A_tall.T)

def test_svd_full_matrices():
    U, s, Vt = svd(A_tall.T, full_matrices=True)
    errors = _check_svd("full_matrices", U, s, Vt, A_tall.T)
    m, n = A_tall.T.shape
    if U.shape != (m, m):
        errors.append(f"  U shape should be ({m},{m}), got {U.shape}")
    if Vt.shape != (n, n):
        errors.append(f"  Vt shape should be ({n},{n}), got {Vt.shape}")
    return errors

run_test("svd: basic call delegates to svd_jacobi", test_svd_wrapper)
run_test("svd: full_matrices=True shapes", test_svd_full_matrices)


# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Part 4 Results: Passed {passed}/{total} tests")
print(f"{'='*50}")
if passed == total:
    print("🎉  All Part 4 tests passed!  You've built the SVD from scratch!")
    print("    Next up: Part 5 (pseudo-inverse).")
else:
    print(f"💪  Keep going — {total - passed} test(s) remaining.")
print()
