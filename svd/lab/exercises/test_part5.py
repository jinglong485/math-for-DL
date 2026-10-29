"""
test_part5.py -- Tests for Part 5: The Pseudo-Inverse.

Run with:  python test_part5.py
"""
import numpy as np
import sys

# ── import student code ──────────────────────────────────────────────
try:
    from part5_pseudoinverse import pinv, pinv_full_rank, penrose_residuals
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure part5_pseudoinverse.py is in the same directory and all dependencies are met.")
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
A_tall = np.array([[3.0, 2.0, 2.0],
                    [2.0, 3.0, -2.0]]).T   # 3x2, full column rank

A_wide = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])       # 2x3, full row rank

A_square = np.array([[1.0, 2.0],
                      [3.0, 4.0]])

A_rank_deficient = np.array([[1.0, 2.0],
                              [2.0, 4.0],
                              [3.0, 6.0]])   # 3x2, rank 1

A_zero = np.zeros((2, 3))


# ── Tests for pinv ──────────────────────────────────────────────────
print("\n=== Part 5: Pseudo-Inverse ===\n")
print("--- pinv ---")


def test_pinv_square():
    errors = []
    Ap = pinv(A_square)
    Ap_ref = np.linalg.pinv(A_square)
    err = np.max(np.abs(Ap - Ap_ref))
    if err > 1e-8:
        errors.append(f"max error = {err:.2e} (expected < 1e-8)")
        errors.append(f"  yours:  {Ap}")
        errors.append(f"  numpy:  {Ap_ref}")
    return errors


def test_pinv_tall():
    errors = []
    Ap = pinv(A_tall)
    Ap_ref = np.linalg.pinv(A_tall)
    err = np.max(np.abs(Ap - Ap_ref))
    if err > 1e-8:
        errors.append(f"max error = {err:.2e}")
    # Also check: A+ A should be close to I for full-column-rank A
    ApA = Ap @ A_tall
    err2 = np.max(np.abs(ApA - np.eye(A_tall.shape[1])))
    if err2 > 1e-8:
        errors.append(f"A+ A != I, error = {err2:.2e}")
    return errors


def test_pinv_wide():
    errors = []
    Ap = pinv(A_wide)
    Ap_ref = np.linalg.pinv(A_wide)
    err = np.max(np.abs(Ap - Ap_ref))
    if err > 1e-8:
        errors.append(f"max error = {err:.2e}")
    # For full-row-rank: A A+ should be close to I
    AAp = A_wide @ Ap
    err2 = np.max(np.abs(AAp - np.eye(A_wide.shape[0])))
    if err2 > 1e-8:
        errors.append(f"A A+ != I, error = {err2:.2e}")
    return errors


def test_pinv_rank_deficient():
    errors = []
    Ap = pinv(A_rank_deficient)
    Ap_ref = np.linalg.pinv(A_rank_deficient)
    err = np.max(np.abs(Ap - Ap_ref))
    if err > 1e-6:
        errors.append(f"max error = {err:.2e} (expected < 1e-6)")
    return errors


def test_pinv_zero():
    errors = []
    Ap = pinv(A_zero)
    if not np.allclose(Ap, np.zeros((3, 2)), atol=1e-15):
        errors.append(f"pinv of zero matrix should be zero, got max = {np.max(np.abs(Ap)):.2e}")
    return errors


run_test("pinv: square invertible (2x2)", test_pinv_square)
run_test("pinv: tall full-column-rank (3x2)", test_pinv_tall)
run_test("pinv: wide full-row-rank (2x3)", test_pinv_wide)
run_test("pinv: rank-deficient (3x2, rank 1)", test_pinv_rank_deficient)
run_test("pinv: zero matrix", test_pinv_zero)


# ── Tests for pinv_full_rank ────────────────────────────────────────
print("\n--- pinv_full_rank ---")


def test_pinv_fr_tall():
    errors = []
    Ap = pinv_full_rank(A_tall)
    Ap_ref = np.linalg.pinv(A_tall)
    err = np.max(np.abs(Ap - Ap_ref))
    if err > 1e-8:
        errors.append(f"max error = {err:.2e}")
    return errors


def test_pinv_fr_wide():
    errors = []
    Ap = pinv_full_rank(A_wide)
    Ap_ref = np.linalg.pinv(A_wide)
    err = np.max(np.abs(Ap - Ap_ref))
    if err > 1e-8:
        errors.append(f"max error = {err:.2e}")
    return errors


def test_pinv_fr_rank_deficient_raises():
    errors = []
    try:
        pinv_full_rank(A_rank_deficient)
        errors.append("should have raised ValueError for rank-deficient matrix")
    except ValueError:
        pass  # expected
    return errors


run_test("pinv_full_rank: tall, full col rank (left inverse)", test_pinv_fr_tall)
run_test("pinv_full_rank: wide, full row rank (right inverse)", test_pinv_fr_wide)
run_test("pinv_full_rank: rank-deficient raises ValueError", test_pinv_fr_rank_deficient_raises)


# ── Tests for penrose_residuals ─────────────────────────────────────
print("\n--- penrose_residuals ---")


def test_penrose_true_pinv():
    errors = []
    for label, M in [("square", A_square), ("tall", A_tall), ("rank-def", A_rank_deficient)]:
        Ap = np.linalg.pinv(M)
        res = penrose_residuals(M, Ap)
        for i, r in enumerate(res):
            if r > 1e-8:
                errors.append(f"{label}: condition {i+1} residual = {r:.2e} (should be ~0)")
    return errors


def test_penrose_wrong_matrix():
    errors = []
    X_wrong = np.ones((2, 3))
    res = penrose_residuals(A_tall, X_wrong)
    # At least some residuals should be large
    if max(res) < 1e-6:
        errors.append("residuals should be large for a random (wrong) X, but all were tiny")
    return errors


run_test("penrose_residuals: all ~0 for true pseudo-inverse", test_penrose_true_pinv)
run_test("penrose_residuals: nonzero for wrong X", test_penrose_wrong_matrix)


# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Part 5 Results: Passed {passed}/{total} tests")
print(f"{'='*50}")
if passed == total:
    print("🎉  All Part 5 tests passed!  The pseudo-inverse is yours!")
    print("    Next up: Part 6 (least squares).")
else:
    print(f"💪  Keep going — {total - passed} test(s) remaining.")
print()
