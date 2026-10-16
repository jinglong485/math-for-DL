import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)
x = np.array([1.0, 2.0, 3.0])          # a vector: shape (3,)
X = rng.standard_normal((4, 3))        # a matrix: 4 rows (examples), 3 columns (features)
print("x.shape =", x.shape, "  X.shape =", X.shape)

# broadcasting: subtract the column means from every row
Xc = X - X.mean(axis=0)
print("column means after centring:", Xc.mean(axis=0))

# outer product u v^T, and a column vector vs a 1-D array
u, v = np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])
print("outer(u, v) =\n", np.outer(u, v))
print("u[:, None] * v[None, :] gives the same:", np.allclose(u[:, None] * v[None, :], np.outer(u, v)))

# einsum writes index notation directly: repeated indices are summed
A = rng.standard_normal((3, 3))
print("trace       :", np.einsum("ii->", A), "=", np.trace(A))
print("x^T A x     :", np.einsum("i,ij,j->", x, A, x), "=", x @ A @ x)
print("X A (matmul):", np.allclose(np.einsum("nk,kj->nj", X, A), X @ A))
print("row-wise x_n^T A x_n:", np.einsum("ni,ij,nj->n", X, A, X))

# the gradient of f(W) = a^T W b in index notation: df/dW_kl = a_k b_l
a, b = rng.standard_normal(3), rng.standard_normal(3)
print("grad of a^T W b equals outer(a, b):", np.allclose(np.einsum("k,l->kl", a, b), np.outer(a, b)))
