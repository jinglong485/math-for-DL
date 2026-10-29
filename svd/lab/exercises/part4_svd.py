"""
part4_svd.py -- Exercise: the singular value decomposition.

Implement the three functions below.  The signatures, docstrings, and helper
imports are all in place; your job is to fill in the bodies.

Prerequisite: Part 3 (eigenvalues).
"""
from part3_eigenvalues import *
from part1_vectors_matrices import _as_float_matrix


# =====================================================================
# Part 4 -- the singular value decomposition
# =====================================================================

def svd_via_eig(A, full_matrices=False):
    """SVD from the eigen-decomposition of A^T A (the textbook hand method).

    1. eigenpairs of the smaller Gram matrix (A^T A, or A A^T for wide A)
    2. sigma_i = sqrt(lambda_i)  (clip tiny negative rounding to 0)
    3. u_i = A v_i / sigma_i     (never compute U independently!)
    4. complete U (and V) with orthonormal bases of the null spaces.

    Returns U, s, Vt like numpy.linalg.svd.  Simple, but it squares the
    condition number: singular values below ~ sqrt(eps) * sigma_1 are lost
    (they are returned as 0).
    """
    A = _as_float_matrix(A)
    m, n = A.shape

    # === YOUR CODE HERE ===
    # Hint: If m < n (wide matrix), recurse on A^T and swap U <-> V.
    # For tall/square: form A^T A, call jacobi_eigh to get eigenvalues/vectors,
    # then σᵢ = sqrt(clip(λᵢ, 0)), uᵢ = A @ vᵢ / σᵢ for each non-tiny σᵢ.
    # Use complete_orthonormal_basis to fill out U to m×m.
    raise NotImplementedError("Implement me!")


def svd_jacobi(A, full_matrices=False, tol=1e-15, max_sweeps=60):
    """SVD by one-sided Jacobi (Hestenes' method).

    Rotate PAIRS OF COLUMNS of a working copy W = A V until all columns of W
    are mutually orthogonal.  Then W = U Sigma: the column norms are the
    singular values and the normalised columns are the u_i.  A^T A is never
    formed, so small singular values are computed to high relative accuracy.

    For columns w_p, w_q let  alpha = |w_p|^2, beta = |w_q|^2, gamma = w_p.w_q.
    The rotation that makes them orthogonal is the Jacobi rotation that
    diagonalises the 2 x 2 matrix [[alpha, gamma], [gamma, beta]].
    """
    A = _as_float_matrix(A)
    m, n = A.shape

    # === YOUR CODE HERE ===
    # Hint: Start with W = A.copy(), V = I(n).  Sweep over all pairs (p, q):
    #   - compute α = wₚᵀwₚ, β = wqᵀwq, γ = wₚᵀwq
    #   - skip if |γ| is tiny relative to sqrt(αβ)
    #   - solve the 2×2 sub-problem [[α,γ],[γ,β]] for rotation (c, s)
    #     using ζ = (β - α)/(2γ), t = sign(ζ)/(|ζ| + sqrt(1 + ζ²))
    #   - rotate columns p, q of both W and V
    # After convergence: σⱼ = ‖Wⱼ‖, sort descending, uⱼ = Wⱼ/σⱼ.
    # Complete U with complete_orthonormal_basis.
    raise NotImplementedError("Implement me!")


def svd(A, full_matrices=False):
    """Our default SVD: one-sided Jacobi."""
    # === YOUR CODE HERE ===
    # Hint: Just delegate to svd_jacobi.
    raise NotImplementedError("Implement me!")
