import numpy as np
from mcalc import Linear, Tanh, ReLU, SoftmaxCrossEntropy, num_grad, rel_error

rng = np.random.default_rng(0)
X = rng.standard_normal((5, 4))
G = rng.standard_normal((5, 3))          # a random upstream gradient dL/dY

# For a test, use the scalar L = sum(G * layer(X)); then dL/dY = G exactly.
lin = Linear(4, 3, rng)
lin.forward(X); dX = lin.backward(G)
def L_of(W=None, b=None, Xin=None):
    W = lin.W if W is None else W; b = lin.b if b is None else b; Xin = X if Xin is None else Xin
    return np.sum(G * (Xin @ W + b))
print("Linear  dW %.1e  db %.1e  dX %.1e" % (
    rel_error(lin.dW, num_grad(lambda W: L_of(W=W), lin.W)),
    rel_error(lin.db, num_grad(lambda b: L_of(b=b), lin.b)),
    rel_error(dX, num_grad(lambda Z: L_of(Xin=Z), X))))

for Act in (Tanh, ReLU):
    act = Act(); Z = rng.standard_normal((5, 3))
    act.forward(Z); dZ = act.backward(G)
    print("%-7s dZ %.1e" % (Act.__name__, rel_error(dZ, num_grad(lambda Z: np.sum(G * Act().forward(Z)), Z))))

A = rng.standard_normal((5, 3)); y = np.array([0, 2, 1, 1, 0])
ce = SoftmaxCrossEntropy(); ce.forward(A, y)
print("SoftmaxCrossEntropy dA %.1e" % rel_error(ce.backward(), num_grad(lambda A: SoftmaxCrossEntropy().forward(A, y), A)))
