"""
Assignment 3 -- Derivatives with respect to matrices; Gaussians.

Textbook: Chapters 6-7.   Lab Manual: Part 3.

Every function returns (value, gradient) where the gradient has the shape
of the variable.  Treat every entry of X as an independent variable (the
tests perturb single entries, so X may be non-symmetric even when it starts
symmetric).  Compute VALUES with solve / slogdet (not Cholesky, which reads
only one triangle).  numpy.linalg is allowed.
"""
import numpy as np


def trace_AX(X, A):
    """f = tr(A X)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def trace_XtAX(X, A):
    """f = tr(X^T A X), A square and NOT necessarily symmetric."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def bilinear(X, a, b):
    """f = a^T X b."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def logabsdet(X):
    """f = ln |det X| for invertible X (any sign of the determinant)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def det(X):
    """f = det X."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def trace_inv(X, A):
    """f = tr(X^{-1} A)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def inv_bilinear(X, a, b):
    """f = a^T X^{-1} b."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def frobenius_lsq(X, A, B):
    """f = ||A X - B||_F^2."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def tanh_weighted(X, A):
    """f = sum_ij A_ij tanh(X_ij) = tr(A^T tanh(X))."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def gaussian_nll(X, mu, Sigma):
    """Negative log-likelihood of the rows of X (N x D) under N(mu, Sigma).

    value = N D/2 ln(2 pi) + N/2 ln|Sigma| + 1/2 sum_n (x_n-mu)^T Sigma^{-1} (x_n-mu)
    Returns (value, d/dmu, d/dSigma).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def gaussian_mle(X):
    """Maximum-likelihood mean and covariance (divide by N). Returns (mu, Sigma)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def gaussian_conditional(mu, Sigma, idx_b, x_b):
    """Parameters of p(x_a | x_b) for x ~ N(mu, Sigma).

    idx_b: list of observed coordinates; a = the remaining coordinates in
    increasing order.  Returns (mu_cond, Sigma_cond).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def kl_gaussians(mu1, S1, mu2, S2):
    """KL( N(mu1, S1) || N(mu2, S2) ) for full covariance matrices (a float)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def kl_diag_standard(mu, logvar):
    """KL( N(mu, diag(exp(logvar))) || N(0, I) ) -- the VAE term.

    Returns (value, d/dmu, d/dlogvar).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError
