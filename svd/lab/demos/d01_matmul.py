import time
import numpy as np
from linalg_scratch import matmul, outer

A = np.array([[1, 2], [0, 1]]); B = np.array([[3, 1], [1, 0]])  # Worked Ex. 2.2
print("matmul(A, B) =\n", matmul(A, B))
print("sum of outer products =\n",
      outer(A[:, 0], B[0, :]) + outer(A[:, 1], B[1, :]))

rng = np.random.default_rng(1)
X, Y = rng.standard_normal((120, 120)), rng.standard_normal((120, 120))
t0 = time.perf_counter(); C1 = matmul(X, Y); t1 = time.perf_counter()
C2 = X @ Y;                                   t2 = time.perf_counter()
print("max |difference| = %.2e" % np.max(np.abs(C1 - C2)))
print("triple loop: %.3f s,   @ operator: %.6f s" % (t1 - t0, t2 - t1))
