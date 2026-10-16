import numpy as np
from linalg_scratch import norm

print("0.1 + 0.2 == 0.3 ?", 0.1 + 0.2 == 0.3)
print("0.1 + 0.2       =", repr(0.1 + 0.2))
print("machine epsilon =", np.finfo(float).eps)
print("1 + 1e-16 == 1 ?", 1 + 1e-16 == 1)        # below eps/2: lost entirely
print("isclose(0.1+0.2, 0.3):", np.isclose(0.1 + 0.2, 0.3))

# catastrophic cancellation: subtracting nearly equal numbers
x = 1e-8
print("(1+x) - 1 =", (1 + x) - 1, " (exact answer 1e-8)")

# overflow in a naive length computation, and the scaled version
v = np.array([3e200, 4e200])
with np.errstate(over="ignore"):
    naive = np.sqrt(np.sum(v ** 2))
print("naive length :", naive)
print("scaled norm  :", norm(v))
