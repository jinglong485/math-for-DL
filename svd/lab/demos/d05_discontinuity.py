import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, savefig
from linalg_scratch import pinv

eps_values = np.logspace(-16, 0, 33)
size_default, size_tol = [], []
for e in eps_values:
    A = np.array([[1.0, 0.0], [0.0, e]])
    size_default.append(np.abs(pinv(A)).max())          # rcond = 2 * machine eps
    size_tol.append(np.abs(pinv(A, rcond=1e-6)).max())   # rcond = 1e-6
plt.loglog(eps_values, size_default, color=C1, label="default rcond $=2\\varepsilon_{mach}$")
plt.loglog(eps_values, size_tol, color=C2, ls="--", label="rcond $=10^{-6}$")
plt.xlabel("$\\varepsilon$ in $A_\\varepsilon=$diag$(1,\\varepsilon)$")
plt.ylabel("largest entry of $A_\\varepsilon^+$"); plt.legend(); savefig("pinv_jump")
for e in (1e-3, 1e-7, 1e-17):
    A = np.array([[1.0, 0.0], [0.0, e]])
    print("eps=%.0e  default: %s   rcond=1e-6: %s"
          % (e, np.diag(pinv(A)), np.diag(pinv(A, rcond=1e-6))))
