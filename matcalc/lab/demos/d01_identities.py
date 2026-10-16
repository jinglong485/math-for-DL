import time
import numpy as np
from mcalc import sherman_morrison, woodbury, block_inverse

np.set_printoptions(precision=4, suppress=True)
A = np.diag([2.0, 4.0]); u = v = np.array([1.0, 1.0])          # Worked Example 3.3
print("14 * (A + u u^T)^-1 =\n", 14 * sherman_morrison(np.linalg.inv(A), u, v))
print("determinant lemma: %.4f vs det %.4f" % ((1 + v @ np.linalg.solve(A, u)) * np.linalg.det(A),
                                               np.linalg.det(A + np.outer(u, v))))

M = np.array([[2.0, 1, 0], [1, 2, 1], [0, 1, 2]])               # Worked Example 3.2
Minv = block_inverse(M[:1, :1], M[:1, 1:], M[1:, :1], M[1:, 1:])
print("4 * M^-1 by blocks =\n", 4 * Minv, "\nmatches inv:", np.allclose(Minv, np.linalg.inv(M)))

# Woodbury: a diagonal matrix plus a rank-k update, n = 2000, k = 10
rng = np.random.default_rng(0)
n, k = 2000, 10
d = rng.uniform(1, 2, n); U = rng.standard_normal((n, k)); C = np.eye(k)
Ainv = np.diag(1 / d)
t0 = time.perf_counter(); direct = np.linalg.inv(np.diag(d) + U @ C @ U.T); t1 = time.perf_counter()
wood = woodbury(Ainv, U, C, U.T); t2 = time.perf_counter()
print("n=%d, k=%d: direct inverse %.3f s, Woodbury %.3f s, max difference %.1e"
      % (n, k, t1 - t0, t2 - t1, np.abs(direct - wood).max()))
