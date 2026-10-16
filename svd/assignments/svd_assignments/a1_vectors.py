"""
Assignment 1 -- Vectors and matrices from scratch.

Textbook: Chapters 1-2.   Lab Manual: Part 1.

Rules: use NumPy arrays and element-wise arithmetic only.  Do NOT use
anything from numpy.linalg (or scipy).  In this assignment you may not use
the @ operator, np.dot, np.matmul, np.inner, np.einsum or np.tensordot
either -- that is what you are implementing.
"""
import numpy as np


def _as_float_matrix(A):
    """Helper (given): return A as a 2-D float array copy (a 1-D input becomes a column)."""
    A = np.array(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A


def dot(u, v):
    """Return the dot product u . v as a Python float.

    Args:
        u, v: 1-D array-likes of the same length.
    Raises:
        ValueError: if the lengths differ.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def norm(x, p=2):
    """Return the p-norm of the vector x for p in {1, 2, np.inf}.

    The 2-norm must not overflow for entries as large as 1e200 and must not
    underflow to 0 for entries as small as 1e-200 (hint: scale by max|x_i|).
    The norm of an empty or zero vector is 0.0.

    Raises:
        ValueError: for any other p.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def normalize(x):
    """Return the unit vector x / ||x||_2 as a float array.

    Raises:
        ValueError: if x is the zero vector.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def angle(u, v):
    """Return the angle between u and v in radians, in [0, pi].

    Must never return nan, even when rounding makes the cosine slightly
    larger than 1 in absolute value.

    Raises:
        ValueError: if u or v is the zero vector.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def project_onto_vector(b, a):
    """Return the orthogonal projection of b onto the line through a.

    Raises:
        ValueError: if a is the zero vector.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def outer(u, v):
    """Return the outer product u v^T, a (len(u) x len(v)) float array."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def matmul(A, B):
    """Return the matrix product A B using loops (no @, np.dot, np.matmul ...).

    Args:
        A: (m x n) array-like.  B: (n x p) array-like.
    Returns:
        (m x p) float array.
    Raises:
        ValueError: if the inner dimensions do not match.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def rank_gaussian(A, tol=1e-10):
    """Return the rank of A by Gaussian elimination with partial pivoting.

    An entry counts as zero if its absolute value is <= tol * max(1, max|A|).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError
