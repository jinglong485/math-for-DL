import numpy as np
from linalg_scratch import pinv, pinv_full_rank, penrose_residuals

np.set_printoptions(precision=4, suppress=True)
A = np.array([[3, 2, 2], [2, 3, -2]])                    # Worked Example 9.2
print("45 * A^+ =\n", 45 * pinv(A))
print("full-rank formula agrees:", np.allclose(pinv(A), pinv_full_rank(A)))

T = np.array([[1, 0], [0, 1], [1, 1]])                   # Worked Example 9.3
print("3 * T^+ =\n", 3 * pinv(T))
print("T^+ T =\n", pinv(T) @ T)

D = np.array([[1, 2], [2, 4], [3, 6]]); b = np.array([1, 2, 4])   # Worked Ex. 9.4
Dp = pinv(D)
print("70 * D^+ =\n", 70 * Dp)
x = Dp @ b
print("x+ = %s   (17/70 = %.4f, 17/35 = %.4f)" % (x, 17 / 70, 17 / 35))
print("residual =", b - D @ x)
print("Penrose residuals:", ["%.1e" % r for r in penrose_residuals(D, Dp)])
print("matches NumPy:", np.allclose(Dp, np.linalg.pinv(D)))
try:
    pinv_full_rank(D)
except ValueError as err:
    print("pinv_full_rank(D) ->", err)
