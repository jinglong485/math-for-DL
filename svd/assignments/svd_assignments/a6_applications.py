"""
Assignment 6 -- Applications: norms, low rank, regularisation, PCA.

Textbook: Chapters 7 and 10.   Lab Manual: Parts 6-7.

Rules: nothing from numpy.linalg (or scipy).  You may import your own
functions from a1-a5.
"""
import numpy as np

from .a4_svd import svd_jacobi, singular_values

EPS = np.finfo(float).eps


def _as_float_matrix(A):
    """Helper (given): return A as a 2-D float array copy (a 1-D input becomes a column)."""
    A = np.array(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A


def spectral_norm(A):
    """||A||_2 = sigma_1 (0.0 for the zero matrix)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def frobenius_norm(A):
    """||A||_F = sqrt(sum of squared entries) = sqrt(sum sigma_i^2)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def nuclear_norm(A):
    """||A||_* = sum of the singular values."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def condition_number(A):
    """kappa(A) = sigma_1 / sigma_r, where sigma_r is the smallest singular value
    above max(m, n) * EPS * sigma_1.  Return np.inf for the zero matrix."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def low_rank_approx(A, k):
    """Best rank-k approximation sum_{i<k} sigma_i u_i v_i^T (Eckart-Young).

    k = 0 gives the zero matrix; k >= min(m, n) gives A itself.
    Raises:
        ValueError: if k < 0.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def eckart_young_error(s, k, norm="fro"):
    """Error ||A - A_k|| of the best rank-k approximation, from the singular values.

    Args:
        s: singular values in decreasing order.
        k: rank of the approximation (0 <= k).
        norm: "2" (spectral) or "fro" (Frobenius).
    Returns:
        sigma_{k+1} for norm "2" (0.0 if k >= len(s)),
        sqrt(sigma_{k+1}^2 + ...) for norm "fro".
    Raises:
        ValueError: for any other norm.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def compression_ratio(m, n, k):
    """Storage of a rank-k truncated SVD relative to the full m x n matrix:
    k (m + n + 1) / (m n)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def truncated_svd_solve(A, b, k):
    """TSVD solution sum_{i<k} (u_i^T b / sigma_i) v_i.

    Raises:
        ValueError: if k < 1 or k exceeds the number of non-zero singular values.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def ridge_solve(A, b, lam):
    """Tikhonov / ridge solution  argmin ||Ax - b||^2 + lam ||x||^2
    = sum_i sigma_i / (sigma_i^2 + lam) (u_i^T b) v_i.

    lam = 0 must give the minimum-norm least-squares solution A^+ b
    (terms with sigma_i = 0 contribute nothing).
    Raises:
        ValueError: if lam < 0.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def pca(X, k):
    """Principal component analysis of the rows of X (N samples x d features).

    Returns (components, explained_variance, scores, mean):
        components         (k x d)  rows are the principal directions v_1..v_k
        explained_variance (k,)     sigma_i^2 / (N - 1), decreasing
        scores             (N x k)  coordinates of the centred data: (X - mean) @ components.T
        mean               (d,)     the column means
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError
