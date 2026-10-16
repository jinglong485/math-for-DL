"""
Assignment 2 -- Orthogonality: Gram-Schmidt, Householder QR, projections.

Textbook: Chapter 3.   Lab Manual: Part 2.

Rules: NumPy arrays, element-wise arithmetic and @ are allowed.
Nothing from numpy.linalg (or scipy).  You may import your own
functions from a1_vectors.
"""
import numpy as np

from .a1_vectors import norm


def _as_float_matrix(A):
    """Helper (given): return A as a 2-D float array copy (a 1-D input becomes a column)."""
    A = np.array(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A


def gram_schmidt(A):
    """Reduced QR factorisation by MODIFIED Gram-Schmidt.

    Args:
        A: (m x n) array with linearly independent columns (m >= n).
    Returns:
        (Q, R): Q is (m x n) with orthonormal columns, R is (n x n) upper
        triangular with POSITIVE diagonal, and Q @ R == A.
    Raises:
        ValueError: if a column is (numerically) dependent on the previous
        ones, i.e. its remaining part has norm <= 1e-12 * (norm of the column).

    The tests check that orthogonality survives for a matrix with condition
    number 1e7 -- classical Gram-Schmidt fails that test, modified passes.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def qr_householder(A):
    """Full QR factorisation by Householder reflections.

    Args:
        A: any (m x n) array.
    Returns:
        (Q, R): Q is (m x m) orthogonal, R is (m x n) upper triangular
        (exact zeros below the diagonal), and Q @ R == A.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def back_substitution(U, b):
    """Solve U x = b for an upper-triangular square U.

    Args:
        U: (n x n) upper triangular.  b: length-n vector (or n x k matrix).
    Raises:
        ValueError: if some diagonal entry of U is exactly 0.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def is_orthonormal(Q, tol=1e-10):
    """Return True if the columns of Q are orthonormal: max|Q^T Q - I| <= tol."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def projection_matrix(A):
    """Orthogonal projection matrix onto the column space of A.

    You may assume the columns of A are linearly independent.
    Returns the (m x m) matrix P with P = P^T = P @ P and P @ A == A.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def complete_basis(Q, tol=1e-10):
    """Extend orthonormal columns to a square orthogonal matrix.

    Args:
        Q: (m x k) array with orthonormal columns, 0 <= k <= m
           (k = 0 is allowed: Q has shape (m, 0)).
    Returns:
        (m x m) orthogonal matrix whose first k columns are exactly Q.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError
