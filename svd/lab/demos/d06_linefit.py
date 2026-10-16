import numpy as np
from linalg_scratch import lstsq_normal, lstsq_qr, lstsq_svd

np.set_printoptions(precision=4, suppress=True)
A = np.array([[1, 0], [1, 1], [1, 2]]); b = np.array([6, 0, 0])   # Worked Ex. 8.1
for f in (lstsq_normal, lstsq_qr, lstsq_svd):
    print(f"{f.__name__:13s} C, D = {f(A, b)}")

A1 = np.array([[1, 2]]); print("min-norm solution of x1 + 2 x2 = 5:", lstsq_svd(A1, [5]))

# parabola fit, Exercise 8.4
t = np.array([-1, 0, 1, 2]); y = np.array([4, 6, 0, 6])
V = np.column_stack([np.ones(4), t, t ** 2])
c = lstsq_qr(V, y)
print("parabola coefficients:", c, "  residual:", y - V @ c)
