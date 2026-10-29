"""
part5_pseudoinverse.py -- Exercise: the Moore-Penrose pseudo-inverse.

Implement the three functions below.  You'll use the SVD from Part 4.

Prerequisite: Part 4 (SVD).
"""
from part4_svd import *
from part1_vectors_matrices import _as_float_matrix


# =====================================================================
# Part 5 -- the pseudo-inverse
# =====================================================================

def pinv(A, rcond=None):
    """Moore-Penrose pseudo-inverse  A^+ = V Sigma^+ U^T.

    Singular values <= rcond * sigma_1 are treated as zero.
    Default rcond = max(m, n) * eps  (the same threshold as matrix rank).
    """
    A = _as_float_matrix(A)
    m, n = A.shape

    # === YOUR CODE HERE ===
    # Hint: Compute the SVD of A.  Build Σ⁺ by inverting only the singular
    # values that exceed rcond * σ₁ (set the rest to 0).  Then A⁺ = V Σ⁺ Uᵀ,
    # which can be written as (Vᵀ.T * s_inv) @ U.T using broadcasting.
    raise NotImplementedError("Implement me!")


def pinv_full_rank(A):
    """Pseudo-inverse from the full-rank formulas (no SVD needed).

    independent columns:  A^+ = (A^T A)^{-1} A^T      (left inverse)
    independent rows:     A^+ = A^T (A A^T)^{-1}      (right inverse)
    """
    A = _as_float_matrix(A)
    m, n = A.shape

    # === YOUR CODE HERE ===
    # Hint: Use rank_by_elimination(A) to find r.
    #   If r == n (full column rank): left inverse  (AᵀA)⁻¹Aᵀ  via solve().
    #   If r == m (full row rank):    right inverse Aᵀ(AAᵀ)⁻¹  via solve().
    #   Otherwise: raise ValueError (rank-deficient).
    raise NotImplementedError("Implement me!")


def penrose_residuals(A, X):
    """Frobenius norms of the four Penrose-condition residuals (all ~0 for X = A^+).

    The four conditions are:
        (1) A X A = A
        (2) X A X = X
        (3) (A X)^T = A X
        (4) (X A)^T = X A
    """
    A = _as_float_matrix(A)
    X = _as_float_matrix(X)

    # === YOUR CODE HERE ===
    # Hint: Form AX and XA once.  Compute the Frobenius norm (sqrt of sum of
    # squared entries) of each residual: AXA - A, XAX - X, AX - AXᵀ, XA - XAᵀ.
    # Return a 4-tuple of floats.
    raise NotImplementedError("Implement me!")
