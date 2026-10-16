import numpy as np
from mcalc import Dual, derivative, sin, exp, Tensor, cross_entropy, zero_grad, MLP, make_spirals, num_grad, rel_error

np.set_printoptions(precision=4, suppress=True)
# forward mode: dual numbers (Worked Example 9.2, Exercise 9.2)
print("f(x) = x^3 + 2x at 2:", (lambda x: x ** 3 + 2 * x)(Dual(2.0, 1.0)))
print("f'(1) for (x^2+1)(3x-1):", derivative(lambda x: (x ** 2 + 1) * (3 * x - 1), 1.0))
g = lambda x: exp(sin(x)) * x
print("d/dx x exp(sin x) at 1: dual %.12f, central diff %.12f"
      % (derivative(g, 1.0), (g(1 + 1e-6) - g(1 - 1e-6)) / 2e-6))

# reverse mode: the sweep of Worked Example 9.3
x1, x2 = Tensor(1.0), Tensor(2.0)
f = (x1 * x2 + x1) ** 2
f.backward()
print("f = %.0f, df/dx1 = %.0f, df/dx2 = %.0f" % (f.data, x1.grad, x2.grad))

# a composite function of matrices, checked against finite differences
rng = np.random.default_rng(0)
A0, B0 = rng.standard_normal((3, 4)), rng.standard_normal((4, 2))
def F(A, B):
    return ((A @ B).tanh() * 2.0 + 1.0).exp().sum() / 10.0
A, B = Tensor(A0), Tensor(B0)
out = F(A, B); out.backward()
fA = lambda a: float(F(Tensor(a), Tensor(B0)).data)
fB = lambda b: float(F(Tensor(A0), Tensor(b)).data)
print("composite: dA error %.1e, dB error %.1e" % (rel_error(A.grad, num_grad(fA, A0)), rel_error(B.grad, num_grad(fB, B0))))

# broadcasting: a bias added to every row receives the sum of the row gradients
W, b = Tensor(rng.standard_normal((2, 3))), Tensor(np.zeros(3))
Xb = Tensor(rng.standard_normal((5, 2)))
((Xb @ W + b) ** 2).sum().backward()
print("bias gradient equals column sums:", np.allclose(b.grad, (2 * (Xb.data @ W.data + b.data)).sum(0)))

# the same MLP gradient two ways: hand-written backprop vs our autodiff engine
X, y = make_spirals(20, 3, seed=1)
net = MLP([2, 8, 3], activation="tanh", seed=0)
net.loss(X, y); net.backward()
W1, b1, W2, b2 = [Tensor(p.copy()) for p in net.params()]
loss = cross_entropy((Tensor(X) @ W1 + b1).tanh() @ W2 + b2, y)
loss.backward()
diffs = [np.abs(t.grad - g).max() for t, g in zip((W1, b1, W2, b2), net.grads())]
print("hand backprop vs autodiff, max differences:", " ".join("%.1e" % d for d in diffs))
