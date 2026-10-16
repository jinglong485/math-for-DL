"""
mcalc.py -- matrix calculus for deep learning, implemented in NumPy.

Companion code for the "NumPy Lab Manual".  Rules of the course:
  * NumPy only; numpy.linalg (solve, inv, det, slogdet, cholesky, eigh) is allowed;
  * nothing that differentiates for you (no JAX, PyTorch, TensorFlow, autograd):
    every gradient here is derived by hand and checked numerically.

Sections (match the parts of the Lab Manual):
    Part 1  numerical derivatives and gradient checking; functions of matrices; identities
    Part 2  vector derivatives; gradient descent and Newton's method
    Part 3  matrix derivatives; Gaussians
    Part 4  backpropagation: layers and a multilayer perceptron
    Part 5  automatic differentiation (dual numbers, a reverse-mode Tensor); stability
    Part 6  deep-learning layers: attention, layer norm, flows, Hutchinson, PCA
"""
import numpy as np


# =====================================================================
# Part 1 -- numerical derivatives
# =====================================================================

# [begin:numgrad]
def num_grad(f, X, h=1e-6):
    """Central-difference gradient of a scalar function f at X (any shape).

    Entry i of the result is (f(X + h e_i) - f(X - h e_i)) / 2h, so the
    result has the shape of X -- the rule 'the gradient has the shape of
    the variable'.  Costs 2 * X.size evaluations of f.
    """
    X = np.array(X, dtype=float)                 # a private copy we can perturb
    G = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        old = X[idx]
        X[idx] = old + h
        f_plus = f(X)
        X[idx] = old - h
        f_minus = f(X)
        X[idx] = old                             # restore
        G[idx] = (f_plus - f_minus) / (2 * h)
    return G


def num_jacobian(f, x, h=1e-6):
    """Central-difference Jacobian J[i, j] = d f_i / d x_j of f: R^n -> R^m.

    Inputs and outputs of any shape are flattened, so J is (m, n).
    """
    x = np.array(x, dtype=float)
    m = np.size(f(x))
    J = np.zeros((m, x.size))
    for j, idx in enumerate(np.ndindex(x.shape)):
        old = x[idx]
        x[idx] = old + h
        f_plus = np.ravel(f(x))
        x[idx] = old - h
        f_minus = np.ravel(f(x))
        x[idx] = old
        J[:, j] = (f_plus - f_minus) / (2 * h)
    return J
# [end:numgrad]


# [begin:check]
def rel_error(a, b, tiny=1e-12):
    """Relative error ||a - b|| / max(||a||, ||b||, tiny) (Frobenius norms)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), tiny))


def check_grad(f, grad, X, h=1e-6):
    """Relative error between an analytic gradient and central differences."""
    return rel_error(grad, num_grad(f, X, h))


def check_directional(f, grad, X, h=1e-6, seed=0):
    """Cheap check along one random direction v: two evaluations of f.

    Compares (f(X + h v) - f(X - h v)) / 2h with <grad, v>.
    """
    X = np.asarray(X, dtype=float)
    v = np.random.default_rng(seed).standard_normal(X.shape)
    fd = (f(X + h * v) - f(X - h * v)) / (2 * h)
    an = float(np.sum(grad * v))
    return abs(fd - an) / max(abs(fd), abs(an), 1e-12)


def complex_step_grad(f, x, h=1e-20):
    """Complex-step gradient Im f(x + i h e_j) / h: no subtraction, full accuracy.

    Only valid for f built from analytic operations (no abs, max, comparisons).
    """
    x = np.array(x, dtype=complex)
    g = np.zeros(x.shape)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + 1j * h
        g[idx] = np.imag(f(x)) / h
        x[idx] = old
    return g
# [end:check]


# =====================================================================
# Part 1 -- functions of matrices and identities
# =====================================================================

# [begin:matfun]
def logdet_chol(S):
    """ln|S| for symmetric positive definite S, as 2 * sum(log diag(L)), S = L L^T."""
    L = np.linalg.cholesky(S)
    return 2.0 * float(np.sum(np.log(np.diag(L))))


def sym_fn(S, fn):
    """Spectral function f(S) = Q f(Lambda) Q^T of a symmetric matrix."""
    w, Q = np.linalg.eigh(S)
    return (Q * fn(w)) @ Q.T                   # Q diag(f(w)) Q^T without building diag


def sqrtm_sym(S):
    return sym_fn(S, lambda w: np.sqrt(np.clip(w, 0.0, None)))


def inv_sqrtm_sym(S):
    return sym_fn(S, lambda w: 1.0 / np.sqrt(w))


def expm_sym(S):
    return sym_fn(S, np.exp)


def logm_sym(S):
    return sym_fn(S, np.log)


def expm_taylor(A, terms=20):
    """exp(A) for any square A: Taylor series with scaling and squaring.

    exp(A) = exp(A / 2^s)^(2^s); choosing s so that ||A / 2^s|| < 1/2 makes
    the series converge fast, then s squarings undo the scaling.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    norm1 = np.max(np.sum(np.abs(A), axis=0)) if A.size else 0.0
    s = max(0, int(np.ceil(np.log2(norm1))) + 1) if norm1 > 0 else 0
    B = A / 2.0 ** s
    E, term = np.eye(n), np.eye(n)
    for k in range(1, terms + 1):
        term = term @ B / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E
# [end:matfun]


# [begin:identities]
def sherman_morrison(Ainv, u, v):
    """(A + u v^T)^{-1} from A^{-1} in O(n^2)."""
    Au = Ainv @ u
    vA = v @ Ainv
    return Ainv - np.outer(Au, vA) / (1.0 + v @ Au)


def woodbury(Ainv, U, C, V):
    """(A + U C V)^{-1} from A^{-1}: only a k x k system is solved."""
    K = np.linalg.inv(C) + V @ Ainv @ U          # k x k
    return Ainv - Ainv @ U @ np.linalg.solve(K, V @ Ainv)


def block_inverse(A, B, C, D):
    """Inverse of [[A, B], [C, D]] via the Schur complement of D."""
    Dinv = np.linalg.inv(D)
    P = np.linalg.inv(A - B @ Dinv @ C)          # (M/D)^{-1}
    top = np.hstack([P, -P @ B @ Dinv])
    bottom = np.hstack([-Dinv @ C @ P, Dinv + Dinv @ C @ P @ B @ Dinv])
    return np.vstack([top, bottom])
# [end:identities]


# =====================================================================
# Part 2 -- vector derivatives and optimisation
# =====================================================================

# [begin:vecfun]
def quadratic(x, A, b):
    """f = x^T A x + b^T x with gradient (A + A^T) x + b and Hessian A + A^T."""
    f = x @ A @ x + b @ x
    return f, (A + A.T) @ x + b, A + A.T


def least_squares(w, Phi, t, lam=0.0):
    """E = 1/2 ||Phi w - t||^2 + lam/2 ||w||^2, its gradient and Hessian."""
    r = Phi @ w - t
    E = 0.5 * r @ r + 0.5 * lam * w @ w
    g = Phi.T @ r + lam * w
    H = Phi.T @ Phi + lam * np.eye(len(w))
    return E, g, H
# [end:vecfun]


# [begin:stable]
def logsumexp(a, axis=-1, keepdims=False):
    """Stable log(sum(exp(a))): subtract the maximum first."""
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(a, axis=-1):
    """Stable softmax along an axis."""
    a = np.asarray(a, dtype=float)
    e = np.exp(a - np.max(a, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def softplus(a):
    """log(1 + exp(a)) without overflow: max(a, 0) + log1p(exp(-|a|))."""
    a = np.asarray(a, dtype=float)
    return np.maximum(a, 0.0) + np.log1p(np.exp(-np.abs(a)))


def sigmoid(a):
    """Logistic sigmoid, evaluated so that exp never overflows."""
    a = np.asarray(a, dtype=float)
    e = np.exp(-np.abs(a))                       # always <= 1
    return np.where(a >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def log_sigmoid(a):
    return -softplus(-np.asarray(a, dtype=float))
# [end:stable]


# [begin:vecjac]
def softmax_jacobian(a):
    """d softmax(a) / d a = diag(s) - s s^T."""
    s = softmax(a)
    return np.diag(s) - np.outer(s, s)


def logistic_loss(w, Phi, t):
    """Cross-entropy of logistic regression: value, gradient Phi^T (y - t), Hessian Phi^T R Phi.

    The value is computed as sum(softplus(a) - t a), which never takes log(0).
    """
    a = Phi @ w
    E = float(np.sum(softplus(a) - t * a))
    y = sigmoid(a)
    g = Phi.T @ (y - t)
    H = Phi.T @ (Phi * (y * (1 - y))[:, None])
    return E, g, H
# [end:vecjac]


# [begin:optim]
def gradient_descent(grad_fn, x0, lr, n_steps):
    """Plain gradient descent; returns the array of iterates x_0, ..., x_n."""
    xs = [np.array(x0, dtype=float)]
    for _ in range(n_steps):
        xs.append(xs[-1] - lr * grad_fn(xs[-1]))
    return np.array(xs)


def newton(grad_fn, hess_fn, x0, n_steps):
    """Newton's method x <- x - H^{-1} g (a solve, never an inverse)."""
    xs = [np.array(x0, dtype=float)]
    for _ in range(n_steps):
        x = xs[-1]
        xs.append(x - np.linalg.solve(np.atleast_2d(hess_fn(x)), np.atleast_1d(grad_fn(x))))
    return np.array(xs)
# [end:optim]


# =====================================================================
# Part 3 -- matrix derivatives and Gaussians
# =====================================================================

# [begin:catalogue]
# Each entry: name -> (f(X, P), grad(X, P)), where P holds the constants.
CATALOGUE = {
    "tr(A X)":          (lambda X, P: np.trace(P["A"] @ X),
                         lambda X, P: P["A"].T),
    "tr(A X B)":        (lambda X, P: np.trace(P["A"] @ X @ P["B"]),
                         lambda X, P: P["A"].T @ P["B"].T),
    "tr(X^T A X)":      (lambda X, P: np.trace(X.T @ P["A"] @ X),
                         lambda X, P: (P["A"] + P["A"].T) @ X),
    "a^T X b":          (lambda X, P: P["a"] @ X @ P["b"],
                         lambda X, P: np.outer(P["a"], P["b"])),
    "ln|det X|":        (lambda X, P: np.linalg.slogdet(X)[1],
                         lambda X, P: np.linalg.inv(X).T),
    "det X":            (lambda X, P: np.linalg.det(X),
                         lambda X, P: np.linalg.det(X) * np.linalg.inv(X).T),
    "tr(X^-1 A)":       (lambda X, P: np.trace(np.linalg.solve(X, P["A"])),
                         lambda X, P: -np.linalg.inv(X).T @ P["A"].T @ np.linalg.inv(X).T),
    "a^T X^-1 b":       (lambda X, P: P["a"] @ np.linalg.solve(X, P["b"]),
                         lambda X, P: -np.linalg.inv(X).T @ np.outer(P["a"], P["b"]) @ np.linalg.inv(X).T),
    "||A X - B||_F^2":  (lambda X, P: np.sum((P["A"] @ X - P["B"]) ** 2),
                         lambda X, P: 2 * P["A"].T @ (P["A"] @ X - P["B"])),
    "tr(A^T tanh(X))":  (lambda X, P: np.sum(P["A"] * np.tanh(X)),
                         lambda X, P: P["A"] * (1 - np.tanh(X) ** 2)),
}
# [end:catalogue]


# [begin:gauss]
def gaussian_nll(X, mu, Sigma):
    """Negative log-likelihood of the rows of X under N(mu, Sigma), with gradients.

    Returns (value, d/dmu, d/dSigma) where
        d/dmu    = -Sigma^{-1} sum_n (x_n - mu)
        d/dSigma = 1/2 (N Sigma^{-1} - Sigma^{-1} S Sigma^{-1}),  S = sum (x_n-mu)(x_n-mu)^T
    """
    X = np.atleast_2d(X)
    N, D = X.shape
    diff = X - mu
    Sinv = np.linalg.inv(Sigma)
    S = diff.T @ diff
    # slogdet, not logdet_chol: Cholesky reads only the lower triangle, so a finite
    # difference that perturbs an upper entry would see a different function
    value = 0.5 * N * D * np.log(2 * np.pi) + 0.5 * N * np.linalg.slogdet(Sigma)[1] + 0.5 * np.sum(Sinv * S)
    dmu = -Sinv @ diff.sum(axis=0)
    dSigma = 0.5 * (N * Sinv - Sinv @ S @ Sinv)
    return float(value), dmu, dSigma


def gaussian_mle(X):
    """Maximum-likelihood mean and covariance (divide by N, as in Bishop)."""
    X = np.atleast_2d(X)
    mu = X.mean(axis=0)
    diff = X - mu
    return mu, diff.T @ diff / X.shape[0]


def fit_gaussian_gd(X, lr=0.1, steps=500):
    """Fit N(mu, L L^T) by gradient descent on (mu, L), L lower triangular.

    Chain rule through Sigma = L L^T:  dE/dL = 2 (dE/dSigma) L, keeping the
    lower triangle.  Returns mu, Sigma and the loss history (NLL per point).
    """
    X = np.atleast_2d(X)
    N, D = X.shape
    mu, L = np.zeros(D), np.eye(D)
    history = []
    for _ in range(steps):
        E, dmu, dSigma = gaussian_nll(X, mu, L @ L.T)
        history.append(E / N)
        mu = mu - lr * dmu / N
        L = L - lr * np.tril(2 * dSigma @ L) / N
    return mu, L @ L.T, np.array(history)


def kl_diag_standard(mu, logvar):
    """KL( N(mu, diag(exp(logvar))) || N(0, I) ) and its gradients (the VAE term)."""
    var = np.exp(logvar)
    value = 0.5 * float(np.sum(var + mu ** 2 - 1.0 - logvar))
    return value, mu.copy(), 0.5 * (var - 1.0)
# [end:gauss]


# =====================================================================
# Part 4 -- backpropagation
# =====================================================================

# [begin:layers]
class Linear:
    """Y = X W + b.   backward:  dW = X^T dY,  db = sum_rows(dY),  dX = dY W^T."""

    def __init__(self, d_in, d_out, rng, scale=None):
        scale = np.sqrt(1.0 / d_in) if scale is None else scale
        self.W = rng.standard_normal((d_in, d_out)) * scale
        self.b = np.zeros(d_out)

    def forward(self, X):
        self.X = X                                # keep the input for backward
        return X @ self.W + self.b

    def backward(self, dY):
        self.dW = self.X.T @ dY
        self.db = dY.sum(axis=0)
        return dY @ self.W.T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class Tanh:
    """H = tanh(Z).   backward:  dZ = dH * (1 - H^2)."""

    def forward(self, Z):
        self.H = np.tanh(Z)
        return self.H

    def backward(self, dH):
        return dH * (1.0 - self.H ** 2)

    def params(self):
        return []

    def grads(self):
        return []


class ReLU:
    """H = max(Z, 0).   backward:  dZ = dH * [Z > 0]."""

    def forward(self, Z):
        self.mask = Z > 0
        return Z * self.mask

    def backward(self, dH):
        return dH * self.mask

    def params(self):
        return []

    def grads(self):
        return []


class SoftmaxCrossEntropy:
    """Mean cross-entropy of softmax(A) with integer labels.   backward: (S - T) / N."""

    def forward(self, A, y):
        N = A.shape[0]
        logp = A - logsumexp(A, axis=1, keepdims=True)    # stable log-softmax
        self.S, self.y = np.exp(logp), y
        return float(-np.mean(logp[np.arange(N), y]))

    def backward(self):
        N = self.S.shape[0]
        G = self.S.copy()
        G[np.arange(N), self.y] -= 1.0                     # S - T for one-hot T
        return G / N
# [end:layers]


# [begin:mlp]
class MLP:
    """A multilayer perceptron with a hand-written backward pass."""

    def __init__(self, sizes, activation="tanh", seed=0):
        rng = np.random.default_rng(seed)
        act = {"tanh": Tanh, "relu": ReLU}[activation]
        self.layers = []
        for i, (d_in, d_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(d_in, d_out, rng))
            if i < len(sizes) - 2:                         # no activation after the last layer
                self.layers.append(act())
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X                                            # logits

    def loss(self, X, y):
        return self.loss_fn.forward(self.forward(X), y)

    def backward(self):
        g = self.loss_fn.backward()
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g                                            # gradient w.r.t. the input

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def grads(self):
        return [g for layer in self.layers for g in layer.grads()]


def make_spirals(n_per_class=100, n_classes=3, noise=0.2, seed=0):
    """The classic 2-D spiral classification problem."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        theta = np.linspace(k * 4.0, (k + 1) * 4.0, n_per_class) + rng.standard_normal(n_per_class) * noise
        X.append(np.column_stack([r * np.sin(theta), r * np.cos(theta)]))
        y.append(np.full(n_per_class, k))
    return np.vstack(X), np.concatenate(y)


def train(model, X, y, lr=0.5, epochs=2000, momentum=0.9):
    """Full-batch gradient descent with momentum. Returns the loss history."""
    velocity = [np.zeros_like(p) for p in model.params()]
    history = []
    for _ in range(epochs):
        history.append(model.loss(X, y))
        model.backward()
        for p, g, v in zip(model.params(), model.grads(), velocity):
            v *= momentum
            v -= lr * g
            p += v                                          # in-place: updates the layer
    return np.array(history)
# [end:mlp]


# =====================================================================
# Part 5 -- automatic differentiation
# =====================================================================

# [begin:dual]
class Dual:
    """Dual number a + b eps with eps^2 = 0: forward-mode automatic differentiation."""

    def __init__(self, val, der=0.0):
        self.val, self.der = float(val), float(der)

    @staticmethod
    def _lift(x):
        return x if isinstance(x, Dual) else Dual(x, 0.0)

    def __add__(self, o):
        o = Dual._lift(o)
        return Dual(self.val + o.val, self.der + o.der)
    __radd__ = __add__

    def __sub__(self, o):
        o = Dual._lift(o)
        return Dual(self.val - o.val, self.der - o.der)

    def __rsub__(self, o):
        return Dual._lift(o) - self

    def __mul__(self, o):
        o = Dual._lift(o)
        return Dual(self.val * o.val, self.val * o.der + self.der * o.val)
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Dual._lift(o)
        return Dual(self.val / o.val, (self.der * o.val - self.val * o.der) / o.val ** 2)

    def __rtruediv__(self, o):
        return Dual._lift(o) / self

    def __neg__(self):
        return Dual(-self.val, -self.der)

    def __pow__(self, p):                    # real exponent p
        return Dual(self.val ** p, p * self.val ** (p - 1) * self.der)

    def __repr__(self):
        return f"Dual({self.val:.6g}, {self.der:.6g})"


def _unary(fn, dfn):
    def op(x):
        if isinstance(x, Dual):
            return Dual(fn(x.val), dfn(x.val) * x.der)
        return fn(x)
    return op


sin = _unary(np.sin, np.cos)
cos = _unary(np.cos, lambda v: -np.sin(v))
exp = _unary(np.exp, np.exp)
log = _unary(np.log, lambda v: 1.0 / v)
tanh = _unary(np.tanh, lambda v: 1.0 - np.tanh(v) ** 2)


def derivative(f, x):
    """f'(x) by evaluating f on the dual number x + 1 eps."""
    return f(Dual(x, 1.0)).der
# [end:dual]


# [begin:tensor]
def _unbroadcast(grad, shape):
    """Sum a gradient over the axes that broadcasting expanded, back to `shape`."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, n in enumerate(shape):
        if n == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """A NumPy array that records operations, for reverse-mode differentiation."""

    def __init__(self, data, parents=(), op=""):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._backward = lambda: None
        self.op = op

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _lift(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    # ---- elementary operations: each defines the forward value and its backward rule
    def __add__(self, other):
        other = Tensor._lift(other)
        out = Tensor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out
    __radd__ = __add__

    def __mul__(self, other):
        other = Tensor._lift(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.shape)
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor._lift(other))

    def __rsub__(self, other):
        return Tensor._lift(other) + (-self)

    def __pow__(self, p):                         # real exponent p
        out = Tensor(self.data ** p, (self,), f"**{p}")
        def _backward():
            self.grad += out.grad * p * self.data ** (p - 1)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * Tensor._lift(other) ** -1.0

    def __matmul__(self, other):                  # 2-D matrices
        other = Tensor._lift(other)
        out = Tensor(self.data @ other.data, (self, other), "@")
        def _backward():
            self.grad += out.grad @ other.data.T          # dX = dY W^T
            other.grad += self.data.T @ out.grad          # dW = X^T dY
        out._backward = _backward
        return out

    @property
    def T(self):
        out = Tensor(self.data.T, (self,), "T")
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis) * (1.0 / n)

    def _elementwise(self, value, local_grad, op):
        out = Tensor(value, (self,), op)
        def _backward():
            self.grad += out.grad * local_grad
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        return self._elementwise(e, e, "exp")

    def log(self):
        return self._elementwise(np.log(self.data), 1.0 / self.data, "log")

    def tanh(self):
        t = np.tanh(self.data)
        return self._elementwise(t, 1.0 - t ** 2, "tanh")

    def relu(self):
        return self._elementwise(np.maximum(self.data, 0.0), (self.data > 0).astype(float), "relu")

    def sigmoid(self):
        s = sigmoid(self.data)
        return self._elementwise(s, s * (1.0 - s), "sigmoid")

    # ---- the reverse sweep
    def backward(self):
        """Fill .grad of every tensor this one depends on (self must be a scalar)."""
        order, seen = [], set()
        def visit(v):                              # depth-first topological sort
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = np.ones_like(self.data)        # seed: dL/dL = 1
        for v in reversed(order):
            v._backward()


def cross_entropy(logits, y):
    """Mean softmax cross-entropy as ONE primitive: stable forward, (S - T)/N backward."""
    N = logits.shape[0]
    logp = logits.data - logsumexp(logits.data, axis=1, keepdims=True)
    out = Tensor(-np.mean(logp[np.arange(N), y]), (logits,), "xent")
    def _backward():
        G = np.exp(logp)
        G[np.arange(N), y] -= 1.0
        logits.grad += out.grad * G / N
    out._backward = _backward
    return out


def zero_grad(params):
    for p in params:
        p.grad = np.zeros_like(p.data)
# [end:tensor]


# =====================================================================
# Part 6 -- deep-learning layers
# =====================================================================

# [begin:attention]
def softmax_rows_backward(A, dA):
    """Backward pass of a row-wise softmax: dS = A * (dA - rowsum(dA * A))."""
    return A * (dA - np.sum(dA * A, axis=1, keepdims=True))


def attention_forward(X, Wq, Wk, Wv, causal=False):
    """One self-attention head. Returns Y and a cache for the backward pass."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d_k = Wq.shape[1]
    S = Q @ K.T / np.sqrt(d_k)
    if causal:                                     # token i may only attend to j <= i
        S = np.where(np.tril(np.ones_like(S, dtype=bool)), S, -np.inf)
    A = softmax(S, axis=1)
    Y = A @ V
    return Y, (X, Wq, Wk, Wv, Q, K, V, A, d_k)


def attention_backward(dY, cache):
    """Gradients of self-attention: returns dX, dWq, dWk, dWv."""
    X, Wq, Wk, Wv, Q, K, V, A, d_k = cache
    dV = A.T @ dY
    dA = dY @ V.T
    dS = softmax_rows_backward(A, dA)
    dQ = dS @ K / np.sqrt(d_k)
    dK = dS.T @ Q / np.sqrt(d_k)
    dWq, dWk, dWv = X.T @ dQ, X.T @ dK, X.T @ dV
    dX = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T         # X is used three times: add
    return dX, dWq, dWk, dWv
# [end:attention]


# [begin:layernorm]
def layernorm_forward(X, gamma, beta, eps=1e-5):
    """Normalise each row of X, then scale and shift."""
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    sig = np.sqrt(var + eps)
    Xh = (X - mu) / sig
    return gamma * Xh + beta, (Xh, sig, gamma)


def layernorm_backward(dY, cache):
    """dX = (dXh - mean(dXh) - Xh mean(dXh * Xh)) / sigma, row by row."""
    Xh, sig, gamma = cache
    dXh = dY * gamma
    dX = (dXh - dXh.mean(axis=1, keepdims=True)
          - Xh * (dXh * Xh).mean(axis=1, keepdims=True)) / sig
    return dX, (dY * Xh).sum(axis=0), dY.sum(axis=0)
# [end:layernorm]


# [begin:flows]
def coupling_forward(x, Ws, bs, Wt, bt):
    """Affine coupling layer on the rows of x (N x D, D even).

    z_A = x_A,  z_B = x_B * exp(s) + t,  s = tanh(x_A Ws + bs),  t = x_A Wt + bt.
    Returns z and log|det J| = sum(s) for every row.
    """
    d = x.shape[1] // 2
    xa, xb = x[:, :d], x[:, d:]
    s = np.tanh(xa @ Ws + bs)
    t = xa @ Wt + bt
    return np.hstack([xa, xb * np.exp(s) + t]), s.sum(axis=1)


def coupling_inverse(z, Ws, bs, Wt, bt):
    d = z.shape[1] // 2
    za, zb = z[:, :d], z[:, d:]
    s = np.tanh(za @ Ws + bs)
    t = za @ Wt + bt
    return np.hstack([za, (zb - t) * np.exp(-s)])


def linear_flow_nll(X, A, b):
    """Mean NLL of x = A z + b, z ~ N(0, I), with gradients w.r.t. A and b.

    dA = A^{-T} (I - mean(z z^T)),   db = -A^{-T} mean(z).
    """
    N, D = X.shape
    Z = np.linalg.solve(A, (X - b).T).T            # rows z_n = A^{-1}(x_n - b)
    value = 0.5 * D * np.log(2 * np.pi) + 0.5 * np.mean(np.sum(Z ** 2, axis=1)) \
        + np.linalg.slogdet(A)[1]
    AinvT = np.linalg.inv(A).T
    dA = AinvT @ (np.eye(D) - Z.T @ Z / N)
    db = -AinvT @ Z.mean(axis=0)
    return float(value), dA, db


def hutchinson_samples(matvec, n, n_samples, rng, kind="rademacher"):
    """Individual Hutchinson samples eps^T A eps (their mean estimates tr A)."""
    out = np.empty(n_samples)
    for i in range(n_samples):
        eps = rng.choice([-1.0, 1.0], size=n) if kind == "rademacher" else rng.standard_normal(n)
        out[i] = eps @ matvec(eps)
    return out


def rayleigh_quotient(S, x):
    """R(x) = x^T S x / x^T x and its gradient 2 (S x - R x) / x^T x."""
    xx = x @ x
    R = x @ S @ x / xx
    return float(R), 2.0 * (S @ x - R * x) / xx
# [end:flows]
