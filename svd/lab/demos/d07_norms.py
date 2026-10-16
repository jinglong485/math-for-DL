import numpy as np
from linalg_scratch import (spectral_norm, frobenius_norm, nuclear_norm,
                            condition_number, matrix_rank, low_rank_approx, svd)

A = np.array([[3, 0], [4, 5]])                           # Worked Example 7.2
print("||A||_2 = %.4f  ||A||_F = %.4f  ||A||_* = %.4f  kappa = %.4f  rank = %d"
      % (spectral_norm(A), frobenius_norm(A), nuclear_norm(A),
         condition_number(A), matrix_rank(A)))

B = np.array([[1, 1], [1, 1.0001]])                      # Worked Example 7.4
print("kappa(B) = %.1f" % condition_number(B))

# Eckart-Young: the truncated SVD beats every other rank-k matrix
rng = np.random.default_rng(7)
M = rng.standard_normal((30, 20))
U, s, Vt = svd(M)
for k in (1, 5, 10):
    Mk = low_rank_approx(M, k)
    other = rng.standard_normal((30, k)) @ rng.standard_normal((k, 20))
    other *= np.sum(M * other) / np.sum(other * other)    # best scaling of a random rank-k matrix
    print("k=%2d  ||M-M_k||_2 = %.4f  (sigma_%d = %.4f)   random rank-k error: %.4f"
          % (k, spectral_norm(M - Mk), k + 1, s[k], spectral_norm(M - other)))
