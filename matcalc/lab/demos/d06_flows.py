import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, C3, INK2, savefig
from mcalc import (linear_flow_nll, coupling_forward, coupling_inverse, num_jacobian, num_grad,
                   rel_error, hutchinson_samples, rayleigh_quotient)

np.set_printoptions(precision=4, suppress=True)
A = np.array([[2.0, 0.0], [1.0, 1.0]])                               # Worked Example 11.1
val, _, _ = linear_flow_nll(np.array([[2.0, 3.0]]), A, np.zeros(2))
print("ln p(x) = %.4f   (-ln 2pi - 2.5 - ln 2 = %.4f)" % (-val, -np.log(2 * np.pi) - 2.5 - np.log(2)))

rng = np.random.default_rng(0)
C = np.array([[2.0, 1.2], [1.2, 1.0]])
X = rng.multivariate_normal([1.0, -1.0], C, size=500)
A0, b0 = np.eye(2) + 0.1 * rng.standard_normal((2, 2)), np.zeros(2)
_, dA, db = linear_flow_nll(X, A0, b0)
print("flow gradient check: dA %.1e, db %.1e" % (rel_error(dA, num_grad(lambda M: linear_flow_nll(X, M, b0)[0], A0)),
                                                rel_error(db, num_grad(lambda v: linear_flow_nll(X, A0, v)[0], b0))))
A, b = np.eye(2), np.zeros(2)
for _ in range(2000):
    _, dA, db = linear_flow_nll(X, A, b)
    A, b = A - 0.05 * dA, b - 0.05 * db
mu_ml = X.mean(0); S_ml = (X - mu_ml).T @ (X - mu_ml) / len(X)
print("trained flow: A A^T =\n", A @ A.T, "\nsample covariance =\n", S_ml, "\nb =", b, " mean =", mu_ml)

# a coupling layer: log|det J| from the formula vs a numerical Jacobian
D = 4
Ws, Wt = rng.standard_normal((2, 2)), rng.standard_normal((2, 2))
bs, bt = rng.standard_normal(2), rng.standard_normal(2)
x = rng.standard_normal((1, D))
z, logdet = coupling_forward(x, Ws, bs, Wt, bt)
J = num_jacobian(lambda v: coupling_forward(v.reshape(1, D), Ws, bs, Wt, bt)[0], x)
print("coupling: formula log|det J| = %.6f, numerical = %.6f" % (logdet[0], np.linalg.slogdet(J)[1]))
print("upper-right block of J (must be 0):\n", J[:2, 2:])
print("inverse recovers x:", np.allclose(coupling_inverse(z, Ws, bs, Wt, bt), x))

# Hutchinson's trace estimator: running mean with Rademacher and Gaussian vectors
n = 50
B = rng.standard_normal((n, n)); M = B @ B.T / n
k = np.arange(1, 2001)
fig, ax = plt.subplots(figsize=(6.0, 3.0))
for kind, c in (("rademacher", C1), ("gaussian", C2)):
    s = hutchinson_samples(lambda v: M @ v, n, 2000, np.random.default_rng(1), kind)
    ax.loglog(k, np.abs(np.cumsum(s) / k - np.trace(M)) / np.trace(M), color=c, label=kind, lw=1.2)
    print("%-10s sample std %.2f, estimate after 2000 samples %.3f (true trace %.3f)" % (kind, s.std(), s.mean(), np.trace(M)))
ax.loglog(k, 2 / np.sqrt(k), color=INK2, ls=":", lw=0.9)
ax.text(300, 0.15, "$\\propto 1/\\sqrt{k}$", color=INK2)
ax.set_xlabel("number of samples $k$"); ax.set_ylabel("relative error"); ax.legend(); savefig("hutchinson")

# PCA by gradient ascent on the Rayleigh quotient (Chapter 10)
S = np.array([[2.0, 1.0], [1.0, 2.0]])
x = np.array([1.0, 0.0])
for _ in range(50):
    R, g = rayleigh_quotient(S, x)
    x = x + 0.5 * g
    x = x / np.linalg.norm(x)
print("Rayleigh ascent: R = %.6f, x = %s   (top eigenpair: 3, [0.7071 0.7071])" % (rayleigh_quotient(S, x)[0], x))
