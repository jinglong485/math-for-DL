import numpy as np
from mcalc import logdet_chol, sqrtm_sym, inv_sqrtm_sym, expm_sym, expm_taylor

np.set_printoptions(precision=4, suppress=True)
S = np.array([[4.0, 2.0], [2.0, 3.0]])                        # Worked Example 2.1
print("trace %.0f, det %.0f, eigenvalues %s" % (np.trace(S), np.linalg.det(S), np.linalg.eigvalsh(S)))
print("Cholesky L =\n", np.linalg.cholesky(S))
print("logdet via Cholesky %.6f   ln 8 = %.6f" % (logdet_chol(S), np.log(8)))

# determinants underflow; log-determinants do not
B = 0.1 * np.eye(400)
print("det(0.1 I_400)          =", np.linalg.det(B))
print("log(det(0.1 I_400))     =", np.log(np.linalg.det(B)) if np.linalg.det(B) > 0 else "-inf (log of 0)")
print("logdet_chol(0.1 I_400)  = %.4f   (400 ln 0.1 = %.4f)" % (logdet_chol(B), 400 * np.log(0.1)))

R = sqrtm_sym(np.array([[5.0, 4.0], [4.0, 5.0]]))              # Worked Example 2.2
print("sqrt([[5,4],[4,5]]) =\n", R, "\nsquared =\n", R @ R)
W = inv_sqrtm_sym(np.array([[5.0, 4.0], [4.0, 5.0]]))
print("3 * S^(-1/2) =\n", 3 * W)

E = expm_sym(np.array([[0.0, 1.0], [1.0, 0.0]]))               # Worked Example 2.3
print("exp([[0,1],[1,0]]) =\n", E, "\ncosh 1 = %.4f, sinh 1 = %.4f" % (np.cosh(1), np.sinh(1)))
th = 0.7
Rot = expm_taylor(np.array([[0.0, -th], [th, 0.0]]))
print("exp of skew matrix (theta=0.7) =\n", Rot, "\ncos 0.7 = %.4f, sin 0.7 = %.4f" % (np.cos(th), np.sin(th)))
M = np.random.default_rng(1).standard_normal((4, 4))
print("det(exp(M)) = %.6f   exp(tr M) = %.6f" % (np.linalg.det(expm_taylor(M)), np.exp(np.trace(M))))
