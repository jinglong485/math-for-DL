"""
linalg_scratch.py -- the SVD and the pseudo-inverse, implemented from scratch.

Companion code for the "Python Lab Manual".  Every function uses only basic
NumPy array arithmetic (+, -, *, /, sqrt, sum, slicing, and the @ operator
once we have written our own matmul).  Nothing from numpy.linalg is used;
numpy.linalg appears only in the demo scripts, to *check* our answers.

Sections (match the parts of the Lab Manual):
    Part 1  vectors, matrices, Gaussian elimination
    Part 2  Gram-Schmidt, Householder QR, triangular solves, projections
    Part 3  eigenvalues: Rayleigh quotient, power iteration, Jacobi method
    Part 4  the SVD: via eig(A^T A) and by one-sided Jacobi
    Part 5  the pseudo-inverse and the Penrose conditions
    Part 6  least squares and regularisation
    Part 7  applications: norms, low-rank approximation, PCA
"""
import numpy as np

EPS = np.finfo(float).eps          # machine epsilon, about 2.22e-16


def _as_float_matrix(A):
    """Return A as a 2-D float array (a copy, so callers' data is never changed)."""
    A = np.array(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A


# =====================================================================
# Part 1 -- vectors and matrices
# =====================================================================

# [begin:dot]
def dot(u, v):
    """Dot product u . v = sum_i u_i v_i, written as an explicit loop."""
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if u.shape != v.shape:
        raise ValueError(f"shape mismatch: {u.shape} vs {v.shape}")
    total = 0.0
    for ui, vi in zip(u, v):
        total += ui * vi
    return total
# [end:dot]


# [begin:norm]
def norm(x, p=2):
    """Vector norm: p = 1, 2 or np.inf.

    The 2-norm is computed as  s * sqrt(sum((x/s)^2))  with s = max|x_i|.
    Mathematically this is just sqrt(sum x_i^2); numerically, the scaling
    stops x_i^2 from overflowing when |x_i| is huge (e.g. 1e200).
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    if p == 1:
        return float(np.sum(np.abs(x)))
    if p == np.inf:
        return float(np.max(np.abs(x)))
    if p == 2:
        s = np.max(np.abs(x))
        if s == 0.0:
            return 0.0
        return float(s * np.sqrt(np.sum((x / s) ** 2)))
    raise ValueError("p must be 1, 2 or np.inf")
# [end:norm]


# [begin:normalize]
def normalize(x):
    """Unit vector x / ||x||.  Raises ValueError for the zero vector."""
    x = np.asarray(x, dtype=float)
    n = norm(x)
    if n == 0.0:
        raise ValueError("cannot normalise the zero vector")
    return x / n


def angle(u, v):
    """Angle between u and v in radians, in [0, pi]."""
    c = dot(u, v) / (norm(u) * norm(v))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))   # clip guards rounding


def project_onto_vector(b, a):
    """Projection of b onto the line through a:  (a.b / a.a) a."""
    a = np.asarray(a, dtype=float)
    return (dot(a, b) / dot(a, a)) * a
# [end:normalize]


# [begin:matmul]
def matmul(A, B):
    """Matrix product by the definition (AB)_ij = sum_k A_ik B_kj.

    Three nested loops: O(m n p) operations.  We write it once to see what
    the @ operator does; afterwards we use @, which runs the same arithmetic
    in optimised compiled code.
    """
    A = _as_float_matrix(A)
    B = _as_float_matrix(B)
    m, n = A.shape
    n2, p = B.shape
    if n != n2:
        raise ValueError(f"inner dimensions differ: {A.shape} @ {B.shape}")
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def outer(u, v):
    """Outer product u v^T: the (len(u) x len(v)) matrix with entries u_i v_j."""
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    return u[:, None] * v[None, :]
# [end:matmul]


# [begin:rref]
def rref(A, tol=1e-10):
    """Reduced row echelon form by Gauss-Jordan elimination.

    Partial pivoting: in each column we pick the row with the largest
    |entry| as pivot, which keeps the multipliers <= 1 in size.
    Entries smaller than tol * max|A| count as zero.

    Returns (R, pivot_columns).
    """
    R = _as_float_matrix(A)
    m, n = R.shape
    scale = np.max(np.abs(R)) if R.size else 0.0
    thresh = tol * max(scale, 1.0)
    pivots = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        # choose the pivot row: largest entry in this column, at or below `row`
        p = row + int(np.argmax(np.abs(R[row:, col])))
        if abs(R[p, col]) <= thresh:
            R[row:, col] = 0.0          # whole column is (numerically) zero
            continue
        R[[row, p]] = R[[p, row]]        # swap rows
        R[row] = R[row] / R[row, col]    # scale pivot to 1
        for i in range(m):               # eliminate above and below
            if i != row:
                R[i] = R[i] - R[i, col] * R[row]
        pivots.append(col)
        row += 1
    R[np.abs(R) <= thresh] = 0.0
    return R, pivots


def rank_by_elimination(A, tol=1e-10):
    """Rank = number of pivots in the reduced row echelon form."""
    return len(rref(A, tol)[1])


def null_space_by_elimination(A, tol=1e-10):
    """Basis of N(A): one 'special solution' per free variable (columns of the result)."""
    R, pivots = rref(A, tol)
    n = R.shape[1]
    free = [j for j in range(n) if j not in pivots]
    basis = []
    for f in free:
        x = np.zeros(n)
        x[f] = 1.0
        for i, pc in enumerate(pivots):  # pivot variable = -(its row's entry in column f)
            x[pc] = -R[i, f]
        basis.append(x)
    return np.array(basis).T if basis else np.zeros((n, 0))
# [end:rref]


# =====================================================================
# Part 2 -- orthogonalisation and QR
# =====================================================================

# [begin:gs]
def gram_schmidt_classical(A):
    """Classical Gram-Schmidt: A = Q R with Q (m x n) orthonormal columns.

    r_ij = q_i . a_j uses the ORIGINAL column a_j.
    """
    A = _as_float_matrix(A)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        w = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]       # project the original a_j
            w -= R[i, j] * Q[:, i]
        R[j, j] = norm(w)
        if R[j, j] == 0.0:
            raise ValueError("columns are linearly dependent")
        Q[:, j] = w / R[j, j]
    return Q, R


def gram_schmidt_modified(A):
    """Modified Gram-Schmidt: same result in exact arithmetic.

    r_ij = q_i . w uses the PARTIALLY REDUCED vector w, so each subtraction
    also removes the rounding errors made by the previous ones.
    """
    A = _as_float_matrix(A)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        w = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ w             # project the current w
            w -= R[i, j] * Q[:, i]
        R[j, j] = norm(w)
        if R[j, j] == 0.0:
            raise ValueError("columns are linearly dependent")
        Q[:, j] = w / R[j, j]
    return Q, R
# [end:gs]


# [begin:householder]
def householder_qr(A):
    """Full QR factorisation A = Q R by Householder reflections.

    Q is m x m orthogonal, R is m x n upper triangular.
    Step k reflects column k (from row k down) onto a multiple of e_1 with
    H = I - 2 v v^T / (v^T v),  v = x + sign(x_1) ||x|| e_1.
    The sign choice avoids subtracting two nearly equal numbers.
    """
    R = _as_float_matrix(A)
    m, n = R.shape
    Q = np.eye(m)
    for k in range(min(m - 1, n)):
        x = R[k:, k].copy()
        alpha = norm(x)
        if alpha == 0.0:
            continue                          # column already zero below the diagonal
        v = x.copy()
        v[0] += (1.0 if x[0] >= 0 else -1.0) * alpha
        vtv = v @ v
        # apply H = I - 2 v v^T / v^T v to the trailing rows of R ...
        R[k:, :] -= np.outer(v, (2.0 / vtv) * (v @ R[k:, :]))
        # ... and accumulate Q = H_1 H_2 ... H_k (apply H on the right)
        Q[:, k:] -= np.outer((Q[:, k:] @ v) * (2.0 / vtv), v)
        R[k + 1:, k] = 0.0                    # exact zeros below the diagonal
    return Q, R
# [end:householder]


# [begin:triangular]
def back_substitution(U, b):
    """Solve U x = b for upper-triangular U (b may be a vector or a matrix)."""
    U = _as_float_matrix(U)
    b = np.array(b, dtype=float)
    n = U.shape[0]
    x = np.zeros_like(b, dtype=float)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0.0:
            raise ValueError("singular triangular matrix")
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def forward_substitution(L, b):
    """Solve L x = b for lower-triangular L."""
    L = _as_float_matrix(L)
    b = np.array(b, dtype=float)
    n = L.shape[0]
    x = np.zeros_like(b, dtype=float)
    for i in range(n):
        if L[i, i] == 0.0:
            raise ValueError("singular triangular matrix")
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def solve(A, b):
    """Solve a square system A x = b via Householder QR: R x = Q^T b."""
    A = _as_float_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("solve() needs a square matrix")
    Q, R = householder_qr(A)
    return back_substitution(R, Q.T @ np.asarray(b, dtype=float))
# [end:triangular]


# [begin:projection]
def projection_matrix(A):
    """Orthogonal projector onto C(A) for A with independent columns: P = Q Q^T."""
    A = _as_float_matrix(A)
    n = A.shape[1]
    Q, _ = householder_qr(A)
    Qn = Q[:, :n]                 # first n columns: orthonormal basis of C(A)
    return Qn @ Qn.T


def complete_orthonormal_basis(Q, tol=1e-10):
    """Extend the orthonormal columns of Q (m x k) to an m x m orthogonal matrix.

    Try the standard basis vectors e_1, e_2, ... in turn; remove their
    components along the columns we already have (twice, for safety --
    'twice is enough'); keep the result if anything substantial is left.
    """
    Q = _as_float_matrix(Q)
    m, k = Q.shape
    cols = [Q[:, j] for j in range(k)]
    for i in range(m):
        if len(cols) == m:
            break
        w = np.zeros(m)
        w[i] = 1.0
        for _ in range(2):
            for q in cols:
                w -= (q @ w) * q
        nw = norm(w)
        if nw > tol:
            cols.append(w / nw)
    return np.column_stack(cols) if cols else np.zeros((m, 0))
# [end:projection]


# =====================================================================
# Part 3 -- eigenvalues of symmetric matrices
# =====================================================================

# [begin:power]
def rayleigh_quotient(S, x):
    """R(x) = x^T S x / x^T x."""
    x = np.asarray(x, dtype=float)
    return float(x @ (S @ x) / (x @ x))


def power_iteration(S, x0=None, tol=1e-12, max_iter=10_000, seed=0):
    """Dominant eigenpair of a symmetric matrix S.

    Returns (lam, v, history) where history[k] is the Rayleigh quotient
    after k+1 steps.  Converges when |lambda_1| > |lambda_2|.
    """
    S = _as_float_matrix(S)
    n = S.shape[0]
    if x0 is None:
        x0 = np.random.default_rng(seed).standard_normal(n)
    x = normalize(x0)
    lam = rayleigh_quotient(S, x)
    history = []
    for _ in range(max_iter):
        y = S @ x
        ny = norm(y)
        if ny == 0.0:                 # x is in the null space: eigenvalue 0
            return 0.0, x, history + [0.0]
        x = y / ny
        lam_new = rayleigh_quotient(S, x)
        history.append(lam_new)
        if abs(lam_new - lam) <= tol * max(1.0, abs(lam_new)):
            lam = lam_new
            break
        lam = lam_new
    return lam, x, history


def eigs_by_deflation(S, k, **kwargs):
    """Top-k eigenpairs of symmetric PSD S: power iteration + deflation.

    After finding (lam, v) we replace S by S - lam v v^T, which moves that
    eigenvalue to 0 and leaves the others unchanged.
    """
    S = _as_float_matrix(S).copy()
    lams, vecs = [], []
    for _ in range(k):
        lam, v, _ = power_iteration(S, **kwargs)
        lams.append(lam)
        vecs.append(v)
        S = S - lam * np.outer(v, v)
    return np.array(lams), np.column_stack(vecs)
# [end:power]


# [begin:jacobi]
def jacobi_eigh(S, tol=1e-14, max_sweeps=60, return_history=False):
    """All eigenvalues and eigenvectors of a symmetric matrix (cyclic Jacobi).

    Each step picks an off-diagonal entry s_pq and applies a plane rotation
    J (in the p,q plane) with  S <- J^T S J  chosen so that the new s_pq = 0.
    One 'sweep' visits every pair p < q once.  The off-diagonal mass
    off(S) = sqrt(sum_{i != j} s_ij^2) decreases to zero (quadratically
    near the end), and the product of all rotations holds the eigenvectors.

    Returns (eigenvalues sorted in DECREASING order, V with eigenvectors as
    columns) -- and the list of off(S) after each sweep if requested.
    """
    A = _as_float_matrix(S).copy()
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("matrix must be square")
    if not np.allclose(A, A.T, atol=1e-12 * max(1.0, np.max(np.abs(A)))):
        raise ValueError("matrix must be symmetric")
    V = np.eye(n)
    history = []

    def off(M):
        # sum the off-diagonal squares directly; computing sum(M*M) - sum(diag^2)
        # would cancel catastrophically once the off-diagonal part is tiny
        O = M - np.diag(np.diag(M))
        return np.sqrt(np.sum(O * O))

    scale = np.sqrt(np.sum(A * A)) or 1.0
    for _ in range(max_sweeps):
        history.append(off(A))
        if off(A) <= tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if abs(apq) <= 1e-300:
                    continue
                # rotation angle: t = tan(theta) solves t^2 + 2 t theta_ - 1 = 0
                theta = (A[q, q] - A[p, p]) / (2.0 * apq)
                if abs(theta) > 1e150:             # avoid overflow in theta^2
                    t = 0.5 / theta
                else:
                    t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + np.sqrt(theta * theta + 1.0))
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                # A <- A J  (columns p and q)
                Ap, Aq = A[:, p].copy(), A[:, q].copy()
                A[:, p] = c * Ap - s * Aq
                A[:, q] = s * Ap + c * Aq
                # A <- J^T A  (rows p and q)
                Ap, Aq = A[p, :].copy(), A[q, :].copy()
                A[p, :] = c * Ap - s * Aq
                A[q, :] = s * Ap + c * Aq
                A[p, q] = A[q, p] = 0.0        # exactly zero by construction
                # accumulate eigenvectors V <- V J
                Vp, Vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * Vp - s * Vq
                V[:, q] = s * Vp + c * Vq
    else:
        history.append(off(A))
    eigvals = np.diag(A).copy()
    order = np.argsort(eigvals)[::-1]          # largest first
    eigvals, V = eigvals[order], V[:, order]
    return (eigvals, V, history) if return_history else (eigvals, V)
# [end:jacobi]


# =====================================================================
# Part 4 -- the singular value decomposition
# =====================================================================

# [begin:svd_eig]
def svd_via_eig(A, full_matrices=False):
    """SVD from the eigen-decomposition of A^T A (the textbook hand method).

    1. eigenpairs of the smaller Gram matrix (A^T A, or A A^T for wide A)
    2. sigma_i = sqrt(lambda_i)  (clip tiny negative rounding to 0)
    3. u_i = A v_i / sigma_i     (never compute U independently!)
    4. complete U (and V) with orthonormal bases of the null spaces.

    Returns U, s, Vt like numpy.linalg.svd.  Simple, but it squares the
    condition number: singular values below ~ sqrt(eps) * sigma_1 are lost
    (they are returned as 0).
    """
    A = _as_float_matrix(A)
    m, n = A.shape
    if m < n:                                   # wide: work with A^T and swap back
        U, s, Vt = svd_via_eig(A.T, full_matrices)
        return Vt.T, s, U.T
    k = n                                       # now m >= n, so min(m, n) = n
    lam, V = jacobi_eigh(A.T @ A)
    s = np.sqrt(np.clip(lam, 0.0, None))
    # rank decision on the EIGENVALUES: lambda <= max(m,n) eps lambda_1 is noise,
    # i.e. this method cannot resolve sigma below about sqrt(eps) * sigma_1
    tol = max(m, n) * EPS * (lam[0] if lam.size and lam[0] > 0 else 1.0)
    r = int(np.sum(lam > tol))                  # numerical rank
    U = np.zeros((m, r))
    for i in range(r):
        U[:, i] = (A @ V[:, i]) / s[i]
    U = complete_orthonormal_basis(U)           # m x m orthogonal
    s[r:] = 0.0
    if full_matrices:
        return U, s, V.T
    return U[:, :k], s, V.T
# [end:svd_eig]


# [begin:svd_jacobi]
def svd_jacobi(A, full_matrices=False, tol=1e-15, max_sweeps=60):
    """SVD by one-sided Jacobi (Hestenes' method).

    Rotate PAIRS OF COLUMNS of a working copy W = A V until all columns of W
    are mutually orthogonal.  Then W = U Sigma: the column norms are the
    singular values and the normalised columns are the u_i.  A^T A is never
    formed, so small singular values are computed to high relative accuracy.

    For columns w_p, w_q let  alpha = |w_p|^2, beta = |w_q|^2, gamma = w_p.w_q.
    The rotation that makes them orthogonal is the Jacobi rotation that
    diagonalises the 2 x 2 matrix [[alpha, gamma], [gamma, beta]].
    """
    A = _as_float_matrix(A)
    m, n = A.shape
    if m < n:                                   # wide: decompose A^T and swap back
        U, s, Vt = svd_jacobi(A.T, full_matrices, tol, max_sweeps)
        return Vt.T, s, U.T
    W = A.copy()
    V = np.eye(n)
    for _ in range(max_sweeps):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = W[:, p] @ W[:, p]
                beta = W[:, q] @ W[:, q]
                gamma = W[:, p] @ W[:, q]
                if abs(gamma) <= tol * np.sqrt(alpha * beta) or gamma == 0.0:
                    continue                    # already orthogonal (to working accuracy)
                rotated = True
                zeta = (beta - alpha) / (2.0 * gamma)
                if abs(zeta) > 1e150:              # avoid overflow in zeta^2
                    t = 0.5 / zeta
                else:
                    t = (1.0 if zeta >= 0 else -1.0) / (abs(zeta) + np.sqrt(1.0 + zeta * zeta))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s_ = c * t
                Wp, Wq = W[:, p].copy(), W[:, q].copy()
                W[:, p] = c * Wp - s_ * Wq
                W[:, q] = s_ * Wp + c * Wq
                Vp, Vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * Vp - s_ * Vq
                V[:, q] = s_ * Vp + c * Vq
        if not rotated:
            break
    s = np.array([norm(W[:, j]) for j in range(n)])
    order = np.argsort(s)[::-1]                 # sort singular values, largest first
    s, W, V = s[order], W[:, order], V[:, order]
    tol_rank = max(m, n) * EPS * (s[0] if s.size and s[0] > 0 else 1.0)
    r = int(np.sum(s > tol_rank))
    U = np.zeros((m, r))
    for i in range(r):
        U[:, i] = W[:, i] / s[i]
    U = complete_orthonormal_basis(U)           # fill in u_{r+1}, ..., u_m
    s[r:] = 0.0
    if full_matrices:
        return U, s, V.T
    return U[:, :n], s, V.T


def svd(A, full_matrices=False):
    """Our default SVD: one-sided Jacobi."""
    return svd_jacobi(A, full_matrices=full_matrices)
# [end:svd_jacobi]


# =====================================================================
# Part 5 -- the pseudo-inverse
# =====================================================================

# [begin:pinv]
def pinv(A, rcond=None):
    """Moore-Penrose pseudo-inverse  A^+ = V Sigma^+ U^T.

    Singular values <= rcond * sigma_1 are treated as zero.
    Default rcond = max(m, n) * eps  (the same threshold as matrix rank).
    """
    A = _as_float_matrix(A)
    m, n = A.shape
    U, s, Vt = svd(A)
    if rcond is None:
        rcond = max(m, n) * EPS
    if s.size == 0 or s[0] == 0.0:
        return np.zeros((n, m))
    keep = s > rcond * s[0]
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    # V Sigma^+ U^T = sum_i (1/sigma_i) v_i u_i^T ; broadcasting scales the columns of V
    return (Vt.T * s_inv) @ U.T


def pinv_full_rank(A):
    """Pseudo-inverse from the full-rank formulas (no SVD needed).

    independent columns:  A^+ = (A^T A)^{-1} A^T      (left inverse)
    independent rows:     A^+ = A^T (A A^T)^{-1}      (right inverse)
    """
    A = _as_float_matrix(A)
    m, n = A.shape
    r = rank_by_elimination(A)
    if r == n:
        return solve(A.T @ A, A.T)
    if r == m:
        return solve(A @ A.T, A).T             # (A A^T)^{-1} is symmetric
    raise ValueError("A is rank deficient: use the SVD-based pinv")


def penrose_residuals(A, X):
    """Frobenius norms of the four Penrose-condition residuals (all ~0 for X = A^+)."""
    A = _as_float_matrix(A)
    X = _as_float_matrix(X)
    AX, XA = A @ X, X @ A
    fro = lambda M: float(np.sqrt(np.sum(M * M)))
    return (fro(A @ X @ A - A), fro(X @ A @ X - X), fro(AX - AX.T), fro(XA - XA.T))
# [end:pinv]


# =====================================================================
# Part 6 -- least squares
# =====================================================================

# [begin:cholesky]
def cholesky(S):
    """Cholesky factor L (lower triangular) with S = L L^T, for S positive definite."""
    S = _as_float_matrix(S)
    n = S.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        d = S[j, j] - L[j, :j] @ L[j, :j]
        if d <= 0.0:
            raise ValueError("matrix is not positive definite")
        L[j, j] = np.sqrt(d)
        L[j + 1:, j] = (S[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    return L
# [end:cholesky]


# [begin:lstsq]
def lstsq_normal(A, b):
    """Least squares via the normal equations A^T A x = A^T b (Cholesky solve)."""
    A = _as_float_matrix(A)
    L = cholesky(A.T @ A)
    y = forward_substitution(L, A.T @ np.asarray(b, dtype=float))
    return back_substitution(L.T, y)


def lstsq_qr(A, b):
    """Least squares via Householder QR:  R_1 x = (Q^T b)_1."""
    A = _as_float_matrix(A)
    n = A.shape[1]
    Q, R = householder_qr(A)
    return back_substitution(R[:n, :n], (Q.T @ np.asarray(b, dtype=float))[:n])


def lstsq_svd(A, b, rcond=None):
    """Minimum-norm least-squares solution  x = sum_i (u_i^T b / sigma_i) v_i = A^+ b."""
    return pinv(A, rcond) @ np.asarray(b, dtype=float)
# [end:lstsq]


# [begin:regularise]
def tsvd_solve(A, b, k, usv=None):
    """Truncated-SVD solution: keep only the k largest singular values.

    Pass usv=(U, s, Vt) to reuse an SVD you already computed (e.g. when
    trying many values of k).
    """
    U, s, Vt = svd(A) if usv is None else usv
    c = U[:, :k].T @ np.asarray(b, dtype=float)       # coefficients u_i^T b
    return Vt[:k].T @ (c / s[:k])


def ridge_solve(A, b, lam, usv=None):
    """Tikhonov / ridge solution  x = sum_i sigma_i/(sigma_i^2 + lam) (u_i^T b) v_i."""
    U, s, Vt = svd(A) if usv is None else usv
    c = U.T @ np.asarray(b, dtype=float)
    return Vt.T @ (s / (s ** 2 + lam) * c)
# [end:regularise]


def vandermonde(t, degree):
    """Columns 1, t, t^2, ..., t^degree (for polynomial fitting)."""
    t = np.asarray(t, dtype=float)
    return np.column_stack([t ** j for j in range(degree + 1)])


# =====================================================================
# Part 7 -- applications
# =====================================================================

# [begin:norms]
def singular_values(A):
    return svd(A)[1]


def spectral_norm(A):
    s = singular_values(A)
    return float(s[0]) if s.size else 0.0


def frobenius_norm(A):
    A = _as_float_matrix(A)
    return float(np.sqrt(np.sum(A * A)))


def nuclear_norm(A):
    return float(np.sum(singular_values(A)))


def condition_number(A):
    """kappa = sigma_1 / sigma_r (smallest NON-ZERO singular value); inf if A = 0."""
    s = singular_values(A)
    s = s[s > max(np.shape(A)) * EPS * (s[0] if s.size else 0.0)]
    return float(s[0] / s[-1]) if s.size else np.inf


def matrix_rank(A, tol=None):
    s = singular_values(A)
    if s.size == 0:
        return 0
    if tol is None:
        tol = max(np.shape(A)) * EPS * s[0]
    return int(np.sum(s > tol))
# [end:norms]


# [begin:lowrank]
def low_rank_approx(A, k):
    """Best rank-k approximation (Eckart-Young): sum_{i<=k} sigma_i u_i v_i^T."""
    U, s, Vt = svd(A)
    k = min(k, s.size)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def pca(X, k):
    """Principal component analysis of the rows of X (N samples x d features).

    Returns (components (k x d), explained_variance (k,), scores (N x k), mean).
    """
    X = _as_float_matrix(X)
    mean = X.mean(axis=0)
    Xc = X - mean                                     # centre the data
    U, s, Vt = svd(Xc)
    N = X.shape[0]
    explained_variance = s[:k] ** 2 / (N - 1)
    scores = U[:, :k] * s[:k]                         # = Xc @ Vt[:k].T
    return Vt[:k], explained_variance, scores, mean
# [end:lowrank]
