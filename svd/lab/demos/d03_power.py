import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, C3, savefig
from linalg_scratch import power_iteration, rayleigh_quotient

S = np.array([[2.0, 1.0], [1.0, 2.0]])                   # Worked Example 4.4
x = np.array([1.0, 0.0])
for k in range(4):
    print("k=%d  x=%-12s  R(x)=%.6f  error=%.6f"
          % (k, x.astype(int), rayleigh_quotient(S, x), 3 - rayleigh_quotient(S, x)))
    x = S @ x

lam, v, hist = power_iteration(S, x0=[1.0, 0.0])
print("converged: lambda = %.12f, v = %s, steps = %d" % (lam, v, len(hist)))

# convergence speed depends on the ratio lambda_2 / lambda_1
for ratio, c in zip((0.3, 0.7, 0.9), (C1, C2, C3)):
    D = np.diag([1.0, ratio, ratio / 2])
    _, _, h = power_iteration(D, x0=[1.0, 1.0, 1.0], tol=1e-16, max_iter=60)
    plt.semilogy(range(1, len(h) + 1), np.abs(1.0 - np.array(h)) + 1e-17,
                 color=c, label="$\\lambda_2/\\lambda_1=%.1f$" % ratio)
plt.ylim(1e-16, 1); plt.xlabel("iteration $k$"); plt.ylabel("error in $\\lambda_1$")
plt.legend(); savefig("power_convergence")
