import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, C3, INK2, savefig
from linalg_scratch import svd, tsvd_solve, ridge_solve

# a 1-D blurring operator: each output is a Gaussian-weighted average
n = 100
s_grid = np.linspace(0, 1, n)
A = np.exp(-((s_grid[:, None] - s_grid[None, :]) ** 2) / (2 * 0.02 ** 2))
A /= A.sum(axis=1, keepdims=True)
x_true = ((s_grid > 0.2) & (s_grid < 0.45)) * 1.0 + np.sin(np.pi * s_grid) * (s_grid > 0.55)
rng = np.random.default_rng(0)
b = A @ x_true + 1e-3 * rng.standard_normal(n)          # blurred + noise

usv = U, s, Vt = svd(A)                                 # compute the SVD once
print("sigma_1 = %.2e, sigma_n = %.2e, kappa = %.1e" % (s[0], s[-1], s[0] / s[-1]))
err = lambda x: np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
naive = Vt.T @ ((U.T @ b) / s)                          # A^{-1} b = sum (u_i^T b / sigma_i) v_i
print("naive A^+ b      relative error: %.1e" % err(naive))
best_k = min(range(1, n + 1), key=lambda k: err(tsvd_solve(A, b, k, usv)))
lams = np.logspace(-8, 0, 41)
best_lam = min(lams, key=lambda l: err(ridge_solve(A, b, l, usv)))
print("TSVD,  best k   = %3d    relative error: %.3f" % (best_k, err(tsvd_solve(A, b, best_k, usv))))
print("ridge, best lam = %.0e  relative error: %.3f" % (best_lam, err(ridge_solve(A, b, best_lam, usv))))

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].plot(s_grid, x_true, color=INK2, lw=2.4, label="true signal")
ax[0].plot(s_grid, b, color="#9a9890", lw=1, label="blurred + noise")
ax[0].plot(s_grid, tsvd_solve(A, b, best_k, usv), color=C1, label=f"TSVD, k={best_k}")
ax[0].plot(s_grid, ridge_solve(A, b, best_lam, usv), color=C2, ls="--", label="ridge")
ax[0].legend(fontsize=7); ax[0].set_xlabel("position")
ax[1].semilogy(s, "o", ms=2.5, color=C1, label="$\\sigma_i$")
ax[1].semilogy(np.abs(U.T @ b), "s", ms=2.5, color=C2, label="$|u_i^Tb|$")
ax[1].set_xlabel("index $i$"); ax[1].legend(fontsize=7)
savefig("regularisation")
