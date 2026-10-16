"""
Assignment 3 -- Eigenvalues of symmetric matrices.

Textbook: Chapters 4-5.   Lab Manual: Part 3.

Rules: NumPy arrays, element-wise arithmetic and @ are allowed.
Nothing from numpy.linalg (or scipy).
"""
import numpy as np

from .a1_vectors import norm


def _as_float_matrix(A):
    """Helper (given): return A as a 2-D float array copy (a 1-D input becomes a column)."""
    A = np.array(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A


def rayleigh_quotient(S, x):
    """Return x^T S x / x^T x as a float.

    Raises:
        ValueError: if x is the zero vector.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def power_iteration(S, x0=None, tol=1e-12, max_iter=10_000):
    """Dominant eigenpair of a symmetric matrix by power iteration.

    Args:
        S: (n x n) symmetric matrix with |lambda_1| > |lambda_2|.
        x0: optional starting vector (default: a fixed random vector).
        tol: stop when successive Rayleigh quotients differ by <= tol * max(1, |lambda|).
        max_iter: maximum number of iterations.
    Returns:
        (lam, v): eigenvalue estimate (float) and unit eigenvector (1-D array).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def jacobi_eigh(S, tol=1e-14, max_sweeps=60):
    """All eigenvalues and eigenvectors of a real symmetric matrix (cyclic Jacobi).

    Returns:
        (eigvals, V): eigvals is a 1-D array sorted in DECREASING order;
        V is orthogonal with V[:, i] the eigenvector for eigvals[i],
        so that S @ V == V * eigvals.
    Raises:
        ValueError: if S is not square or not symmetric
                    (|S - S^T| > 1e-10 * max(1, max|S|)).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def spectral_reconstruct(eigvals, V):
    """Return V diag(eigvals) V^T = sum_i lambda_i v_i v_i^T."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def classify_definiteness(S, tol=1e-10):
    """Classify a symmetric matrix by the signs of its eigenvalues.

    An eigenvalue counts as zero if |lambda| <= tol * max(1, max|lambda|).
    Returns exactly one of the strings
        "positive definite", "positive semidefinite",
        "negative definite", "negative semidefinite", "indefinite".
    The zero matrix counts as "positive semidefinite".
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError
