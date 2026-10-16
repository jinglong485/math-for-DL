import warnings
import numpy as np
from mcalc import logsumexp, softmax, softplus, log_sigmoid

np.set_printoptions(precision=4)
a = np.array([1000.0, 1001.0, 1002.0])                               # Worked Example 9.4
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    naive = np.exp(a) / np.sum(np.exp(a))
    print("naive softmax :", naive)
    print("naive lse     :", np.log(np.sum(np.exp(a))))
    print("naive softplus(1000):", np.log(1 + np.exp(1000.0)))
    print("naive log sigmoid(-800):", np.log(1 / (1 + np.exp(800.0))))
print("stable softmax:", softmax(a))
print("stable lse    : %.4f" % logsumexp(a))
print("stable softplus(1000) = %.1f, softplus(-1000) = %.1f" % (softplus(1000.0), softplus(-1000.0)))
print("stable log sigmoid(-800) = %.1f" % log_sigmoid(-800.0))

# binary cross-entropy from probabilities vs from logits
t, logit = 1.0, -40.0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    p = 1 / (1 + np.exp(-logit))
    print("BCE via probability: %.4f   (p = %.3e)" % (-(t * np.log(p) + (1 - t) * np.log(1 - p)), p))
print("BCE via logits     : %.4f" % (softplus(logit) - t * logit))

# log-determinant and solves: never invert
rng = np.random.default_rng(0)
B = rng.standard_normal((300, 300)); S = B @ B.T + 1e-3 * np.eye(300)
sign, ld = np.linalg.slogdet(S)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("log(det(S)) =", np.log(np.linalg.det(S)), "   slogdet =", round(ld, 4))
# solve versus inverse on the Hilbert matrix h_ij = 1/(i+j-1), kappa ~ 1.6e13
n = 10
Hm = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
x_true = np.ones(n); rhs = Hm @ x_true
y_solve = np.linalg.solve(Hm, rhs); y_inv = np.linalg.inv(Hm) @ rhs
print("Hilbert n=10: residual |H y - b|  solve %.1e, inv %.1e" % (np.linalg.norm(Hm @ y_solve - rhs), np.linalg.norm(Hm @ y_inv - rhs)))
print("              error    |y - x|    solve %.1e, inv %.1e" % (np.linalg.norm(y_solve - x_true), np.linalg.norm(y_inv - x_true)))
