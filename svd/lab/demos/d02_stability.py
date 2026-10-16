import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, C3, savefig
from linalg_scratch import (householder_qr, gram_schmidt_classical,
                            gram_schmidt_modified)

def matrix_with_condition(kappa, m=40, n=20, seed=0):
    """A = U diag(s) V^T with singular values from 1 down to 1/kappa."""
    rng = np.random.default_rng(seed)
    U, _ = householder_qr(rng.standard_normal((m, m)))
    V, _ = householder_qr(rng.standard_normal((n, n)))
    s = np.logspace(0, -np.log10(kappa), n)
    return (U[:, :n] * s) @ V.T

def loss(Q):                      # how far Q is from orthonormal columns
    k = Q.shape[1]
    return np.abs(Q.T @ Q - np.eye(k)).max()

kappas = np.logspace(1, 15, 15)
res = {"classical GS": [], "modified GS": [], "Householder": []}
for kap in kappas:
    A = matrix_with_condition(kap)
    res["classical GS"].append(loss(gram_schmidt_classical(A)[0]))
    res["modified GS"].append(loss(gram_schmidt_modified(A)[0]))
    res["Householder"].append(loss(householder_qr(A)[0][:, :A.shape[1]]))

print(" kappa     classical    modified   Householder")
for i in range(0, 15, 2):
    print("%7.0e   %9.1e   %9.1e   %9.1e" % (kappas[i], res["classical GS"][i],
          res["modified GS"][i], res["Householder"][i]))

for (lab, val), c, mk in zip(res.items(), (C1, C2, C3), ("o", "s", "^")):
    plt.loglog(kappas, val, marker=mk, color=c, label=lab)
plt.xlabel("condition number $\\kappa(A)$")
plt.ylabel("$\\max|Q^TQ-I|$")
plt.legend(); savefig("gs_stability")
