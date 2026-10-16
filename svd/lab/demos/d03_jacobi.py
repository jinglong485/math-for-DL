import numpy as np
from linalg_scratch import jacobi_eigh

np.set_printoptions(precision=4, suppress=True)
S = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])           # Worked Example 4.3
lam, V = jacobi_eigh(S)
print("eigenvalues:", lam)
print("eigenvectors (columns):\n", V)
print("|S V - V diag(lam)| = %.1e" % np.abs(S @ V - V * lam).max())

rng = np.random.default_rng(3)
B = rng.standard_normal((8, 8)); S8 = B + B.T            # random symmetric 8 x 8
lam8, V8, hist = jacobi_eigh(S8, return_history=True)
print("off-diagonal norm after each sweep:")
print("  " + "  ".join("%.1e" % h for h in hist))
ref = np.sort(np.linalg.eigvalsh(S8))[::-1]               # NumPy, for checking only
print("max |our eigenvalues - NumPy's| = %.1e" % np.abs(lam8 - ref).max())
