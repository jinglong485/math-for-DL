import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, INK2, savefig
from mcalc import gaussian_nll, gaussian_mle, fit_gaussian_gd, kl_diag_standard, num_grad, rel_error

np.set_printoptions(precision=4, suppress=True)
X = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 0.0]])                 # Worked Example 7.1
mu, Sigma = gaussian_mle(X)
print("mu_ML =", mu, "\n3 * Sigma_ML =\n", 3 * Sigma)
_, dmu, dSigma = gaussian_nll(X, mu, Sigma)
print("gradients at the MLE: |dmu| = %.1e, |dSigma| = %.1e" % (np.abs(dmu).max(), np.abs(dSigma).max()))

rng = np.random.default_rng(0)
Xr = rng.standard_normal((50, 3)) @ np.array([[1, 0, 0], [0.5, 1, 0], [0.2, 0.3, 0.5]])
m0 = rng.standard_normal(3); S0 = np.cov(rng.standard_normal((3, 20)))
_, dmu, dS = gaussian_nll(Xr, m0, S0)
print("gradient check: dmu %.1e, dSigma %.1e"
      % (rel_error(dmu, num_grad(lambda m: gaussian_nll(Xr, m, S0)[0], m0)),
         rel_error(dS, num_grad(lambda S: gaussian_nll(Xr, m0, S)[0], S0))))

# fit a correlated 2-D Gaussian by gradient descent on (mu, L)
C = np.array([[2.0, 1.2], [1.2, 1.0]])
D2 = rng.multivariate_normal([1.0, -1.0], C, size=400)
mu_gd, Sig_gd, hist = fit_gaussian_gd(D2, lr=0.2, steps=400)
mu_ml, Sig_ml = gaussian_mle(D2)
print("gradient descent: mu =", mu_gd, " Sigma =\n", Sig_gd)
print("closed form     : mu =", mu_ml, " Sigma =\n", Sig_ml)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].semilogx(np.arange(1, len(hist) + 1), hist, color=C1); ax[0].set_xlabel("step"); ax[0].set_ylabel("NLL per point")
ax[1].scatter(D2[:, 0], D2[:, 1], s=4, color="#9a9890")
g1, g2 = np.meshgrid(np.linspace(-3, 5, 200), np.linspace(-4, 2, 200))
G = np.stack([g1 - mu_gd[0], g2 - mu_gd[1]], axis=-1)
q = np.einsum("...i,ij,...j->...", G, np.linalg.inv(Sig_gd), G)
ax[1].contour(g1, g2, q, levels=[1, 4, 9], colors=C2, linewidths=1.2)
ax[1].set_aspect("equal"); ax[1].set_title("1, 2, 3 standard-deviation contours", fontsize=8)
savefig("gauss_fit")

# the VAE KL term (Worked Example 7.3) and the reparameterisation trick
val, dmu, dlv = kl_diag_standard(np.array([1.0, 0.0]), np.log(np.array([1.0, 4.0])))
print("KL = %.4f, dKL/dmu = %s, dKL/dlogvar = %s" % (val, dmu, dlv))
mu = np.array([0.5, -1.0]); L = np.array([[1.0, 0.0], [0.6, 0.8]])
eps = rng.standard_normal((200000, 2)); Z = mu + eps @ L.T            # z = mu + L eps
grad_f = 2 * Z                                                       # f(z) = ||z||^2
print("reparameterisation, d/dmu E||z||^2: MC %s, exact %s" % (grad_f.mean(0), 2 * mu))
print("d/dL E||z||^2: MC\n", np.tril(grad_f.T @ eps / len(eps)), "\nexact 2L (lower part)\n", 2 * L)
