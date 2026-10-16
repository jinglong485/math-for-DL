import numpy as np
from linalg_scratch import gram_schmidt_modified, householder_qr

np.set_printoptions(precision=4, suppress=True)
A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])            # Worked Example 3.2
Q, R = gram_schmidt_modified(A)
print("Q =\n", Q)
print("R =\n", R)
print("sqrt(2), 1/sqrt(2), sqrt(3/2), 1/sqrt(6), 2/sqrt(3) =",
      np.round([np.sqrt(2), 1/np.sqrt(2), np.sqrt(1.5), 1/np.sqrt(6), 2/np.sqrt(3)], 4))
print("|Q^T Q - I| = %.1e,  |QR - A| = %.1e"
      % (np.abs(Q.T @ Q - np.eye(3)).max(), np.abs(Q @ R - A).max()))

Qh, Rh = householder_qr(A)
print("Householder R =\n", Rh)
print("(rows may differ from Gram-Schmidt by a sign: both are valid QRs)")
