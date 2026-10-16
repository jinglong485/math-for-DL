import numpy as np
from linalg_scratch import projection_matrix, complete_orthonormal_basis

np.set_printoptions(precision=4, suppress=True)
A = np.array([[1, 0], [1, 1], [1, 2]]); b = np.array([6, 0, 0])   # Worked Ex. 3.1
P = projection_matrix(A)
print("6 P =\n", 6 * P)
print("P b =", P @ b, "   error e =", b - P @ b)
print("|P@P - P| = %.1e,  |P - P^T| = %.1e"
      % (np.abs(P @ P - P).max(), np.abs(P - P.T).max()))

q = np.array([[1, 1, 1]]).T / np.sqrt(3)                 # extend one unit vector
Q = complete_orthonormal_basis(q)
print("completed basis Q =\n", Q)
print("|Q^T Q - I| = %.1e" % np.abs(Q.T @ Q - np.eye(3)).max())
