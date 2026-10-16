"""
Assignment 4 -- The singular value decomposition.

Textbook: Chapters 6-7.   Lab Manual: Part 4.

Conventions (the same as numpy.linalg.svd):
  * functions return (U, s, Vt) where s is a 1-D array of the
    min(m, n) singular values in DECREASING order and Vt = V^T;
  * full_matrices=False -> U is (m x k), Vt is (k x n), k = min(m, n);
  * full_matrices=True  -> U is (m x m), Vt is (n x n), both orthogonal.
Singular vectors are only determined up to sign (and up to rotation for
repeated singular values), so the tests check properties, not entries.

Rules: nothing from numpy.linalg (or scipy).  You may import your own
functions from a1-a3 (e.g. jacobi_eigh, complete_basis).
"""
import numpy as np

from .a1_vectors import norm
from .a2_orthogonal import complete_basis
from .a3_eigen import jacobi_eigh

EPS = np.finfo(float).eps


def _as_float_matrix(A):
    """Helper (given): return A as a 2-D float array copy (a 1-D input becomes a column)."""
    A = np.array(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A


def svd_via_eig(A, full_matrices=False):
    """SVD by the textbook recipe: eigen-decompose A^T A (or A A^T if A is wide).

    sigma_i = sqrt(lambda_i), then u_i = A v_i / sigma_i for the non-zero
    sigma_i, then complete U (e.g. with complete_basis from A2).
    Eigenvalues lambda_i <= max(m, n) * EPS * lambda_1 count as zero (this
    method cannot resolve anything smaller); return their sigma_i as exactly 0.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def svd_jacobi(A, full_matrices=False, tol=1e-15, max_sweeps=60):
    """SVD by one-sided Jacobi rotations on the columns of A.

    Must NOT form A^T A.  The tests require the smallest singular value of the
    Lauchli matrix [[1, 1], [1e-8, 0], [0, 1e-8]] to be correct to 1e-6
    relative accuracy -- the eigenvalue method cannot achieve that.
    Singular values <= max(m, n) * EPS * sigma_1 must be returned as exactly 0.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def singular_values(A):
    """Return the min(m, n) singular values of A in decreasing order."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def svd_reconstruct(U, s, Vt):
    """Return U @ Sigma @ Vt, building the (rectangular) Sigma from s.

    Must work for both reduced and full factors, e.g. U (m x m), s (k,),
    Vt (n x n) with k = min(m, n).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def matrix_rank(A, tol=None):
    """Numerical rank: the number of singular values > tol.

    Default tol = max(m, n) * EPS * sigma_1.  The zero matrix has rank 0.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def four_subspaces(A, tol=None):
    """Orthonormal bases of the four fundamental subspaces, from the SVD.

    Returns a dict with keys
        "row"       (n x r)      basis of C(A^T)
        "null"      (n x (n-r))  basis of N(A)
        "column"    (m x r)      basis of C(A)
        "left_null" (m x (m-r))  basis of N(A^T)
    where r = matrix_rank(A, tol).  Each value is a 2-D array whose columns
    are orthonormal (an empty basis has shape (n, 0) or (m, 0)).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError
