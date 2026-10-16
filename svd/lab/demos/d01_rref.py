import numpy as np
from linalg_scratch import rref, rank_by_elimination, null_space_by_elimination

np.set_printoptions(precision=4, suppress=True)
A = np.array([[1, 2, 3], [2, 4, 7], [3, 6, 10]])        # Worked Example 2.3
R, pivots = rref(A)
print("R =\n", R)
print("pivot columns:", pivots, "  rank:", rank_by_elimination(A))
N = null_space_by_elimination(A)
print("null space basis (columns):\n", N)
print("left null space basis:\n", null_space_by_elimination(A.T))
print("A @ N =", (A @ N).ravel())

B = np.array([[1, 2, 0, 1], [2, 4, 1, 4], [3, 6, 1, 5]])  # Exercise 2.3
print("Exercise 2.3: rank", rank_by_elimination(B), " null space:\n",
      null_space_by_elimination(B))
