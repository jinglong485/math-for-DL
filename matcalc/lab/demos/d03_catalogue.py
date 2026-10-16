import numpy as np
from mcalc import CATALOGUE, num_grad, rel_error

rng = np.random.default_rng(0)
n = 4
P = {"A": rng.standard_normal((n, n)), "B": rng.standard_normal((n, n)),
     "a": rng.standard_normal(n), "b": rng.standard_normal(n)}
X = rng.standard_normal((n, n)) + 2 * np.eye(n)        # comfortably invertible
print("%-18s %12s" % ("function", "rel. error"))
for name, (f, grad) in CATALOGUE.items():
    err = rel_error(grad(X, P), num_grad(lambda Y: f(Y, P), X))
    print("%-18s %12.1e" % (name, err))

# Worked Examples 6.1 and 6.2
X1 = np.array([[2.0, 1.0], [1.0, 1.0]])
print("d ln|X| / dX =\n", CATALOGUE["ln|det X|"][1](X1, {}))
A2 = np.array([[1.0, 2.0], [0.0, 1.0]]); X2 = np.array([[1.0, 0.0], [1.0, 1.0]])
print("d tr(X^T A X) / dX =\n", CATALOGUE["tr(X^T A X)"][1](X2, {"A": A2}))
