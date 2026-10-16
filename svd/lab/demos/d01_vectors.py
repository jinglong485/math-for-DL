import numpy as np
from linalg_scratch import dot, norm, normalize, angle, project_onto_vector

u, v = np.array([1, 2, 2]), np.array([2, 0, 1])          # Worked Example 1.1
print("u.v       =", dot(u, v))
print("|u|, |v|  =", norm(u), norm(v))
print("angle     = %.4f degrees" % np.degrees(angle(u, v)))
print("u/|u|     =", normalize(u))

x = np.array([3, -4, 0])                                  # Worked Example 1.2
print("norms 1,2,inf:", norm(x, 1), norm(x, 2), norm(x, np.inf))

a, b = np.array([1, 2, 2]), np.array([1, 1, 1])          # Worked Example 1.4
p = project_onto_vector(b, a)
e = b - p
print("p =", p, " (= 5/9 * a)")
print("e =", e, " e.a =", dot(e, a))

# compare with NumPy (allowed in demos, never inside linalg_scratch)
assert np.isclose(dot(u, v), u @ v)
assert np.isclose(norm(x), np.linalg.norm(x))
print("all checks against NumPy passed")
