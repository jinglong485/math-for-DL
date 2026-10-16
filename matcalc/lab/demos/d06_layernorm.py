import numpy as np
from mcalc import layernorm_forward, layernorm_backward, num_grad, num_jacobian, rel_error

np.set_printoptions(precision=4, suppress=True)
x = np.array([[1.0, 2.0, 3.0]])                                      # Worked Example 12.2
J = num_jacobian(lambda v: layernorm_forward(v.reshape(1, 3), np.ones(3), np.zeros(3), eps=0.0)[0], x)
print("numerical Jacobian of the standardisation:\n", J)
print("J 1 =", J @ np.ones(3), "  J x =", J @ x.ravel())

rng = np.random.default_rng(0)
X = rng.standard_normal((4, 6)); gamma = rng.standard_normal(6); beta = rng.standard_normal(6)
G = rng.standard_normal((4, 6))
_, cache = layernorm_forward(X, gamma, beta)
dX, dgamma, dbeta = layernorm_backward(G, cache)
L = lambda X=X, g=gamma, b=beta: np.sum(G * layernorm_forward(X, g, b)[0])
print("layer norm: dX %.1e, dgamma %.1e, dbeta %.1e" % (
    rel_error(dX, num_grad(lambda Z: L(X=Z), X)),
    rel_error(dgamma, num_grad(lambda g: L(g=g), gamma)),
    rel_error(dbeta, num_grad(lambda b: L(b=b), beta))))
