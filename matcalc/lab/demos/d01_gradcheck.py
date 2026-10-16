import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, C3, savefig
from mcalc import num_grad, num_jacobian, check_grad, check_directional, complex_step_grad

np.set_printoptions(precision=6, suppress=True)
f = lambda x: x[0] ** 2 * x[1] + 3 * x[1]                   # Worked Example 1.1
print("num_grad at (1,2):", num_grad(f, [1.0, 2.0]), "  exact: [4, 4]")

g = lambda x: np.array([x[0] ** 2 * x[1], 5 * x[0] + np.sin(x[1])])   # Worked Example 1.2
print("num_jacobian at (1,0):\n", num_jacobian(g, [1.0, 0.0]))

# a correct and a wrong gradient for f(x) = x^T A x with non-symmetric A
rng = np.random.default_rng(0)
A = rng.standard_normal((5, 5)); x = rng.standard_normal(5)
q = lambda x: x @ A @ x
print("relative error, (A + A^T) x :", "%.1e" % check_grad(q, (A + A.T) @ x, x))
print("relative error, 2 A x (bug) :", "%.1e" % check_grad(q, 2 * A @ x, x))
print("directional check, correct  :", "%.1e" % check_directional(q, (A + A.T) @ x, x))

# error of three derivative approximations as a function of h, for f(x) = exp(sin x) at x = 1
fs = lambda x: np.exp(np.sin(x))
exact = np.cos(1.0) * np.exp(np.sin(1.0))
hs = np.logspace(-16, -1, 61)
fwd = [abs((fs(1 + h) - fs(1)) / h - exact) for h in hs]
cen = [abs((fs(1 + h) - fs(1 - h)) / (2 * h) - exact) for h in hs]
cpx = [abs(np.imag(fs(1 + 1j * h)) / h - exact) for h in hs]
best_f, best_c = hs[int(np.argmin(fwd))], hs[int(np.argmin(cen))]
print("best h: forward %.0e (error %.1e), central %.0e (error %.1e)"
      % (best_f, min(fwd), best_c, min(cen)))
print("complex step at h=1e-20: error %.1e" % abs(complex_step_grad(lambda v: np.exp(np.sin(v[0])), [1.0], 1e-20)[0] - exact))
plt.loglog(hs, np.maximum(fwd, 1e-17), color=C1, label="forward difference")
plt.loglog(hs, np.maximum(cen, 1e-17), color=C2, label="central difference")
plt.loglog(hs, np.maximum(cpx, 1e-17), color=C3, label="complex step")
plt.xlabel("step $h$"); plt.ylabel("absolute error"); plt.legend(loc="upper center"); savefig("fd_error")
