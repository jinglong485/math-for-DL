import numpy as np
from linalg_scratch import svd_via_eig, svd_jacobi

np.set_printoptions(precision=4, suppress=True)
examples = {"Worked Ex. 6.1": [[3, 0], [4, 5]],
            "Worked Ex. 6.2": [[3, 2, 2], [2, 3, -2]],
            "Worked Ex. 6.3": [[1, 1], [1, 1], [1, 1]],
            "Worked Ex. 6.4": [[1, 1], [0, 1], [1, 0]]}
for name, A in examples.items():
    A = np.array(A, dtype=float)
    U, s, Vt = svd_jacobi(A, full_matrices=True)
    Sigma = np.zeros(A.shape); Sigma[:s.size, :s.size] = np.diag(s)
    print(f"--- {name}:  sigma = {s}")
    print("U =\n", U); print("V^T =\n", Vt)
    print("|U Sigma V^T - A| = %.1e   |U^T U - I| = %.1e   |V^T V - I| = %.1e"
          % (np.abs(U @ Sigma @ Vt - A).max(), np.abs(U.T @ U - np.eye(U.shape[0])).max(),
             np.abs(Vt @ Vt.T - np.eye(Vt.shape[0])).max()))
    s2 = svd_via_eig(A)[1]
    print("svd_via_eig sigma =", s2, "  NumPy:", np.linalg.svd(A, compute_uv=False))
