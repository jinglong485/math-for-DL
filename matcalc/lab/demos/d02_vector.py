import numpy as np
from mcalc import (quadratic, least_squares, softmax, softmax_jacobian, logsumexp,
                   logistic_loss, num_grad, num_jacobian, check_grad, rel_error)

np.set_printoptions(precision=4, suppress=True)
A = np.array([[2.0, 1.0], [1.0, 3.0]]); b = np.array([1.0, -1.0])     # Worked Example 4.1
f, g, H = quadratic(np.array([1.0, 2.0]), A, b)
print("f = %.1f, gradient = %s, Hessian eigenvalues = %s" % (f, g, np.linalg.eigvalsh(H)))
print("minimiser -A^{-1} b / 2 =", -0.5 * np.linalg.solve(A, b))

a = np.array([0.0, np.log(2.0), 0.0])                                 # Worked Example 4.2
print("softmax =", softmax(a))
print("16 * Jacobian =\n", 16 * softmax_jacobian(a))
print("Jacobian vs finite differences: %.1e" % rel_error(softmax_jacobian(a), num_jacobian(softmax, a)))
print("grad of logsumexp equals softmax: %.1e" % rel_error(num_grad(logsumexp, a), softmax(a)))

Phi = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]); t = np.array([6.0, 0.0, 0.0])
w = np.linalg.solve(Phi.T @ Phi, Phi.T @ t)                          # normal equations
print("line fit w =", w, "  gradient there:", least_squares(w, Phi, t)[1])

rng = np.random.default_rng(0)
Phi = rng.standard_normal((20, 3)); tt = (rng.random(20) < 0.5).astype(float); w = rng.standard_normal(3)
E, g, H = logistic_loss(w, Phi, tt)
print("logistic loss: gradient error %.1e, Hessian error %.1e"
      % (check_grad(lambda v: logistic_loss(v, Phi, tt)[0], g, w),
         rel_error(H, num_jacobian(lambda v: logistic_loss(v, Phi, tt)[1], w))))
print("Hessian eigenvalues (all >= 0, convex):", np.linalg.eigvalsh(H))
