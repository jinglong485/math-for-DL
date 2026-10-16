"""
Assignment 1 -- Gradient checking, functions of matrices, matrix identities.

Textbook: Chapters 1-3 and 9.1-9.2.   Lab Manual: Part 1.

Rules: NumPy only.  numpy.linalg (solve, inv, det, slogdet, cholesky, eigh)
is allowed.  No JAX, PyTorch, TensorFlow, autograd, SciPy.
"""
import numpy as np


def numerical_gradient(f, X, h=1e-6):
    """Central-difference gradient of a scalar function f at X.

    Args:
        f: function taking an array shaped like X and returning a float.
        X: array of any shape (the point).  It must NOT be modified.
        h: step size.
    Returns:
        Float array with the same shape as X whose entry idx is
        (f(X + h E_idx) - f(X - h E_idx)) / (2h).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def numerical_jacobian(f, x, h=1e-6):
    """Central-difference Jacobian of f: R^n -> R^m.

    Inputs and outputs of any shape are flattened (row-major), so the
    result J has shape (m, n) with J[i, j] = d f_i / d x_j.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def relative_error(a, b):
    """||a - b|| / max(||a||, ||b||, 1e-12) with Frobenius (2-) norms, as a float."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def logdet_spd(S):
    """ln det S for a symmetric positive definite S, computed from its Cholesky factor.

    Must work when det S under- or overflows (e.g. 0.1 * I of size 400).
    Raises:
        ValueError: if S is not positive definite.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def sym_matrix_function(S, fn):
    """f(S) = Q f(Lambda) Q^T for a symmetric matrix S = Q Lambda Q^T.

    fn is applied element-wise to the 1-D array of eigenvalues.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def sqrtm_psd(S):
    """The symmetric positive semidefinite square root of a PSD matrix.

    Tiny negative eigenvalues from rounding (>= -1e-12 * max|eigenvalue|)
    are treated as 0.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def inv_sqrtm_pd(S):
    """S^{-1/2} for a symmetric positive definite S (the whitening matrix)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def expm(A, terms=20):
    """Matrix exponential of ANY square matrix (not only symmetric ones).

    Use the Taylor series with scaling and squaring: the result must be
    accurate (relative error < 1e-10) even for matrices of norm ~ 20.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def sherman_morrison(Ainv, u, v):
    """(A + u v^T)^{-1}, given A^{-1}, in O(n^2) operations (no new inverse)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def woodbury(Ainv, U, C, V):
    """(A + U C V)^{-1}, given A^{-1}, solving only a k x k system (C is k x k)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def block_inverse(A, B, C, D):
    """Inverse of the block matrix [[A, B], [C, D]] via the Schur complement of D.

    Returns the full (p+q) x (p+q) inverse.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def block_logabsdet(A, B, C, D):
    """ln |det [[A, B], [C, D]]| = ln|det D| + ln|det(A - B D^{-1} C)|."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError
