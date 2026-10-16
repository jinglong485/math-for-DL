"""
Assignment 5 -- The pseudo-inverse and least squares.

Textbook: Chapters 8-9.   Lab Manual: Parts 5-6.

Rules: nothing from numpy.linalg (or scipy).  You may import your own
functions from a1-a4.
"""
import numpy as np

from .a2_orthogonal import qr_householder, back_substitution
from .a4_svd import svd_jacobi, matrix_rank

EPS = np.finfo(float).eps


def _as_float_matrix(A):
    """Helper (given): return A as a 2-D float array copy (a 1-D input becomes a column)."""
    A = np.array(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A


def pinv(A, rcond=None):
    """Moore-Penrose pseudo-inverse V Sigma^+ U^T of an (m x n) matrix.

    Singular values <= rcond * sigma_1 are treated as zero;
    default rcond = max(m, n) * EPS.  The result has shape (n x m);
    the pseudo-inverse of a zero matrix is the zero matrix.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def _solve(M, B):
    """Solve the square system M X = B with Householder QR."""
    Q, R = qr_householder(M)
    return back_substitution(R, Q.T @ B)


def pinv_full_rank(A):
    """Pseudo-inverse from the full-rank formulas (no SVD of A).

    independent columns:  (A^T A)^{-1} A^T
    independent rows:     A^T (A A^T)^{-1}
    Raises:
        ValueError: if A has neither independent columns nor independent rows.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def penrose_residuals(A, X):
    """Return the Frobenius norms of the four Penrose residuals, as a tuple

        (||AXA - A||, ||XAX - X||, ||(AX)^T - AX||, ||(XA)^T - XA||).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def cholesky(S):
    """Lower-triangular L with L L^T = S.

    Raises:
        ValueError: if S is not (numerically) positive definite.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def lstsq_normal(A, b):
    """Least-squares solution from the normal equations A^T A x = A^T b,
    solved with your Cholesky factorisation.

    Raises:
        ValueError: if A^T A is not positive definite (dependent columns).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def lstsq_qr(A, b):
    """Least-squares solution for A with independent columns, via Householder QR."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def lstsq_min_norm(A, b, rcond=None):
    """Minimum-norm least-squares solution A^+ b (works for any A)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError
