import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, C3, savefig
from linalg_scratch import svd_via_eig, svd_jacobi

eps = 1e-8
L = np.array([[1, 1], [eps, 0], [0, eps]])               # the Lauchli matrix
print("true sigma_2      = 1.0000e-08")
print("svd_via_eig  gives %.4e" % svd_via_eig(L)[1][1])
print("svd_jacobi   gives %.4e" % svd_jacobi(L)[1][1])

# relative error of the smallest singular value as kappa grows
rng = np.random.default_rng(0)
Q1 = np.linalg.qr(rng.standard_normal((30, 30)))[0]     # NumPy only builds the
Q2 = np.linalg.qr(rng.standard_normal((10, 10)))[0]     # test matrices here
kappas = np.logspace(1, 14, 14)
err_eig, err_jac = [], []
for kap in kappas:
    s = np.logspace(0, -np.log10(kap), 10)
    A = (Q1[:, :10] * s) @ Q2.T
    err_eig.append(abs(svd_via_eig(A)[1][-1] - s[-1]) / s[-1])
    err_jac.append(abs(svd_jacobi(A)[1][-1] - s[-1]) / s[-1])
plt.loglog(kappas, np.maximum(err_eig, 1e-17), "o-", color=C1, label="via eig$(A^TA)$")
plt.loglog(kappas, np.maximum(err_jac, 1e-17), "s-", color=C2, label="one-sided Jacobi")
eps_m = np.finfo(float).eps
plt.loglog(kappas, eps_m * kappas, color="#52514e", lw=0.8, ls=":")
plt.loglog(kappas, np.minimum(eps_m * kappas ** 2, 1), color="#52514e", lw=0.8, ls="--")
plt.text(3e11, 4e-5, "$\\kappa\\varepsilon$", color="#52514e")
plt.text(2e6, 1e-3, "$\\kappa^2\\varepsilon$", color="#52514e")
plt.xlabel("condition number $\\kappa(A)$")
plt.ylabel("relative error in $\\sigma_{\\min}$"); plt.legend(); savefig("svd_accuracy")
for k, e1, e2 in list(zip(kappas, err_eig, err_jac))[::3]:
    print("kappa=%.0e   eig-based error %.1e   Jacobi error %.1e" % (k, e1, e2))
