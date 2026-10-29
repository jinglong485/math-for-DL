"""
test_part7.py -- Tests for Part 7: Applications (norms, low-rank, PCA).

Run with:  python test_part7.py
"""
import numpy as np
import sys

# ── import student code ──────────────────────────────────────────────
try:
    from part7_applications import (singular_values, spectral_norm,
                                     frobenius_norm, nuclear_norm,
                                     condition_number, matrix_rank,
                                     low_rank_approx, pca)
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure part7_applications.py is in the same directory and all dependencies are met.")
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


# ── Test matrices ────────────────────────────────────────────────────
A = np.array([[3.0, 2.0, 2.0],
              [2.0, 3.0, -2.0]])

A_rank1 = np.array([[1.0, 2.0],
                     [2.0, 4.0],
                     [3.0, 6.0]])

A_3x3 = np.array([[2.0, -1.0, 0.0],
                   [-1.0, 2.0, -1.0],
                   [0.0, -1.0, 2.0]])

np.random.seed(42)
X_data = np.random.randn(50, 4)  # 50 samples, 4 features for PCA


# ── Tests for singular_values ───────────────────────────────────────
print("\n=== Part 7: Applications ===\n")
print("--- singular_values ---")


def test_singular_values():
    errors = []
    s = singular_values(A)
    s_ref = np.linalg.svd(A, compute_uv=False)
    err = np.max(np.abs(np.sort(s)[::-1] - np.sort(s_ref)[::-1]))
    if err > 1e-8:
        errors.append(f"singular value mismatch = {err:.2e}")
        errors.append(f"  yours:  {s}")
        errors.append(f"  numpy:  {s_ref}")
    return errors


run_test("singular_values: 2x3 matrix", test_singular_values)


# ── Tests for spectral_norm ──────────────────────────────────────────
print("\n--- spectral_norm ---")


def test_spectral_norm():
    errors = []
    sn = spectral_norm(A)
    sn_ref = np.linalg.norm(A, ord=2)
    if abs(sn - sn_ref) > 1e-8:
        errors.append(f"spectral norm = {sn}, expected {sn_ref}")
    return errors


def test_spectral_norm_identity():
    errors = []
    sn = spectral_norm(np.eye(5))
    if abs(sn - 1.0) > 1e-10:
        errors.append(f"spectral norm of I should be 1.0, got {sn}")
    return errors


run_test("spectral_norm: 2x3 matrix", test_spectral_norm)
run_test("spectral_norm: identity = 1", test_spectral_norm_identity)


# ── Tests for frobenius_norm ─────────────────────────────────────────
print("\n--- frobenius_norm ---")


def test_frobenius_norm():
    errors = []
    fn = frobenius_norm(A)
    fn_ref = np.linalg.norm(A, 'fro')
    if abs(fn - fn_ref) > 1e-10:
        errors.append(f"frobenius norm = {fn}, expected {fn_ref}")
    return errors


run_test("frobenius_norm: 2x3 matrix", test_frobenius_norm)


# ── Tests for nuclear_norm ───────────────────────────────────────────
print("\n--- nuclear_norm ---")


def test_nuclear_norm():
    errors = []
    nn = nuclear_norm(A)
    nn_ref = np.linalg.norm(A, 'nuc')
    if abs(nn - nn_ref) > 1e-6:
        errors.append(f"nuclear norm = {nn}, expected {nn_ref}")
    return errors


run_test("nuclear_norm: 2x3 matrix", test_nuclear_norm)


# ── Tests for condition_number ───────────────────────────────────────
print("\n--- condition_number ---")


def test_condition_number():
    errors = []
    kappa = condition_number(A_3x3)
    kappa_ref = np.linalg.cond(A_3x3)
    if abs(kappa - kappa_ref) / max(kappa_ref, 1.0) > 1e-6:
        errors.append(f"condition number = {kappa}, expected {kappa_ref}")
    return errors


def test_condition_number_identity():
    errors = []
    kappa = condition_number(np.eye(3))
    if abs(kappa - 1.0) > 1e-8:
        errors.append(f"condition number of I should be 1.0, got {kappa}")
    return errors


run_test("condition_number: 3x3 tridiagonal", test_condition_number)
run_test("condition_number: identity = 1", test_condition_number_identity)


# ── Tests for matrix_rank ────────────────────────────────────────────
print("\n--- matrix_rank ---")


def test_matrix_rank():
    errors = []
    for label, M, expected in [("3x3 full", A_3x3, 3),
                                 ("rank-1", A_rank1, 1),
                                 ("2x3", A, 2),
                                 ("zero", np.zeros((3, 3)), 0)]:
        r = matrix_rank(M)
        if r != expected:
            errors.append(f"{label}: rank = {r}, expected {expected}")
    return errors


run_test("matrix_rank: various matrices", test_matrix_rank)


# ── Tests for low_rank_approx ───────────────────────────────────────
print("\n--- low_rank_approx ---")


def test_low_rank_approx_full():
    errors = []
    Ak = low_rank_approx(A, k=2)
    err = np.max(np.abs(Ak - A))
    if err > 1e-8:
        errors.append(f"rank-2 approx of rank-2 matrix should be exact, error = {err:.2e}")
    return errors


def test_low_rank_approx_truncated():
    errors = []
    Ak = low_rank_approx(A_3x3, k=1)
    # Check rank
    r = np.linalg.matrix_rank(Ak)
    if r > 1:
        errors.append(f"rank-1 approximation has rank {r}")
    # Check it's the best rank-1 (Eckart-Young): error = sqrt(sum of sigma_2^2 + sigma_3^2)
    s_ref = np.linalg.svd(A_3x3, compute_uv=False)
    expected_error = np.sqrt(np.sum(s_ref[1:]**2))
    actual_error = np.linalg.norm(A_3x3 - Ak, 'fro')
    if abs(actual_error - expected_error) > 1e-6:
        errors.append(f"Frobenius error = {actual_error:.6f}, expected {expected_error:.6f}")
    return errors


run_test("low_rank_approx: full rank = exact reconstruction", test_low_rank_approx_full)
run_test("low_rank_approx: rank-1 of 3x3 (Eckart-Young)", test_low_rank_approx_truncated)


# ── Tests for pca ────────────────────────────────────────────────────
print("\n--- pca ---")


def test_pca_shapes():
    errors = []
    components, var, scores, mean = pca(X_data, k=2)
    N, d = X_data.shape
    if components.shape != (2, d):
        errors.append(f"components shape = {components.shape}, expected (2, {d})")
    if var.shape != (2,):
        errors.append(f"explained_variance shape = {var.shape}, expected (2,)")
    if scores.shape != (N, 2):
        errors.append(f"scores shape = {scores.shape}, expected ({N}, 2)")
    if mean.shape != (d,):
        errors.append(f"mean shape = {mean.shape}, expected ({d},)")
    return errors


def test_pca_centering():
    errors = []
    _, _, _, mean = pca(X_data, k=2)
    mean_ref = X_data.mean(axis=0)
    err = np.max(np.abs(mean - mean_ref))
    if err > 1e-10:
        errors.append(f"mean mismatch = {err:.2e}")
    return errors


def test_pca_variance():
    errors = []
    _, var, _, _ = pca(X_data, k=4)
    # Compare with numpy SVD
    Xc = X_data - X_data.mean(axis=0)
    s_ref = np.linalg.svd(Xc, compute_uv=False)
    var_ref = s_ref[:4]**2 / (X_data.shape[0] - 1)
    err = np.max(np.abs(np.sort(var)[::-1] - np.sort(var_ref)[::-1]))
    if err > 1e-6:
        errors.append(f"explained variance mismatch = {err:.2e}")
        errors.append(f"  yours:  {var}")
        errors.append(f"  numpy:  {var_ref}")
    return errors


def test_pca_reconstruction():
    errors = []
    components, _, scores, mean = pca(X_data, k=4)  # full rank
    X_recon = scores @ components + mean
    err = np.max(np.abs(X_recon - X_data))
    if err > 1e-6:
        errors.append(f"full-rank PCA reconstruction error = {err:.2e}")
    return errors


run_test("pca: output shapes", test_pca_shapes)
run_test("pca: mean matches data mean", test_pca_centering)
run_test("pca: explained variance matches numpy", test_pca_variance)
run_test("pca: full-rank reconstruction", test_pca_reconstruction)


# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Part 7 Results: Passed {passed}/{total} tests")
print(f"{'='*50}")
if passed == total:
    print()
    print("🎉🎉🎉  Congratulations!  🎉🎉🎉")
    print()
    print("You've built the entire SVD pipeline from scratch!")
    print()
    print("  Part 1: Vectors, matrices, Gaussian elimination")
    print("  Part 2: Gram-Schmidt, Householder QR, projections")
    print("  Part 3: Eigenvalues (Jacobi, power iteration)")
    print("  Part 4: The SVD (via eigendecomposition & one-sided Jacobi)")
    print("  Part 5: The Moore-Penrose pseudo-inverse")
    print("  Part 6: Least squares & regularisation")
    print("  Part 7: Norms, low-rank approximation, PCA")
    print()
    print("Every function used only basic NumPy arithmetic — no numpy.linalg!")
    print("You now have a deep understanding of how these algorithms work.")
    print()
else:
    print(f"💪  Keep going — {total - passed} test(s) remaining.")
print()
