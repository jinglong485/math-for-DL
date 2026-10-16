# Optional: compare our hand-derived gradients with JAX's automatic differentiation.
import numpy as np
from mcalc import CATALOGUE, MLP, make_spirals, softmax_jacobian

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)        # JAX defaults to float32
except ImportError:
    print("JAX is not installed -- this optional demo is skipped (pip install jax).")
    raise SystemExit

print("JAX version", jax.__version__)
rng = np.random.default_rng(0)
n = 4
A = rng.standard_normal((n, n)); X = rng.standard_normal((n, n)) + 2 * np.eye(n)
checks = {
    "ln|det X|":   (lambda X: jnp.linalg.slogdet(X)[1], {}),
    "tr(X^-1 A)":  (lambda X: jnp.trace(jnp.linalg.solve(X, A)), {"A": A}),
    "tr(X^T A X)": (lambda X: jnp.trace(X.T @ A @ X), {"A": A}),
}
for name, (fj, P) in checks.items():
    ours = CATALOGUE[name][1](X, P)
    print("%-12s max |ours - jax.grad| = %.1e" % (name, np.abs(ours - np.asarray(jax.grad(fj)(X))).max()))

# Jacobians: forward mode (jacfwd) and reverse mode (jacrev) give the same matrix
a = np.array([0.0, np.log(2.0), 0.0])
sm = lambda a: jax.nn.softmax(a)
print("softmax Jacobian: |ours - jacfwd| = %.1e, |ours - jacrev| = %.1e"
      % (np.abs(softmax_jacobian(a) - jax.jacfwd(sm)(a)).max(), np.abs(softmax_jacobian(a) - jax.jacrev(sm)(a)).max()))

# a whole MLP: our hand-written backward pass vs jax.grad
Xd, y = make_spirals(20, 3, seed=1)
net = MLP([2, 8, 8, 3], activation="tanh", seed=0)
net.loss(Xd, y); net.backward()
def loss(params):
    h = Xd
    for i in range(0, len(params) - 2, 2):
        h = jnp.tanh(h @ params[i] + params[i + 1])
    logits = h @ params[-2] + params[-1]
    logp = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
    return -jnp.mean(logp[jnp.arange(len(y)), y])
jg = jax.grad(loss)([jnp.asarray(p) for p in net.params()])
print("MLP: max |ours - jax.grad| per parameter:", " ".join("%.1e" % np.abs(g - np.asarray(j)).max() for g, j in zip(net.grads(), jg)))
