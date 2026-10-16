import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, C3, savefig
from mcalc import Linear, ReLU, MLP, make_spirals, train, num_grad, rel_error

np.set_printoptions(precision=4, suppress=True)
# Worked Example 8.1: the two-layer ReLU network by hand
x = np.array([[1.0, 1.0]]); t = 3.0
l1, relu, l2 = Linear(2, 2, np.random.default_rng(0)), ReLU(), Linear(2, 1, np.random.default_rng(0))
l1.W = np.array([[1.0, 2.0], [1.0, -1.0]]); l1.b = np.zeros(2)
l2.W = np.array([[1.0], [-1.0]]); l2.b = np.zeros(1)
y = l2.forward(relu.forward(l1.forward(x)))
print("forward: z =", l1.forward(x), " y =", y, " L =", 0.5 * (y - t) ** 2)
dz = relu.backward(l2.backward(y - t))
dx = l1.backward(dz)
print("dW2 =", l2.dW.ravel(), " dW1 =\n", l1.dW, "\ndx =", dx)

# gradient check of a full MLP, every parameter
X, yl = make_spirals(20, 3, seed=1)
net = MLP([2, 8, 8, 3], activation="tanh", seed=0)
net.loss(X, yl); net.backward()
for i, (p, g) in enumerate(zip(net.params(), net.grads())):
    def L(q, p=p):
        old = p.copy(); p[...] = q; val = net.loss(X, yl); p[...] = old; return val
    print("parameter %d %-8s rel. error %.1e" % (i, str(p.shape), rel_error(g, num_grad(L, p.copy()))))

# train on the spiral data set
X, yl = make_spirals(100, 3, seed=0)
net = MLP([2, 64, 64, 3], activation="tanh", seed=0)
hist = train(net, X, yl, lr=0.2, epochs=3000, momentum=0.9)
acc = np.mean(np.argmax(net.forward(X), axis=1) == yl)
print("loss: start %.3f, after 3000 epochs %.4f; training accuracy %.1f%%" % (hist[0], hist[-1], 100 * acc))
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
ax[0].semilogy(hist, color=C1); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy")
g1, g2 = np.meshgrid(np.linspace(-1.1, 1.1, 300), np.linspace(-1.1, 1.1, 300))
pred = np.argmax(net.forward(np.column_stack([g1.ravel(), g2.ravel()])), axis=1).reshape(g1.shape)
from matplotlib.colors import ListedColormap
ax[1].contourf(g1, g2, pred, levels=[-0.5, 0.5, 1.5, 2.5], cmap=ListedColormap(["#d3e3f6", "#fbd9cb", "#c9ecdd"]))
for k, c in zip(range(3), (C1, C2, C3)):
    ax[1].scatter(X[yl == k, 0], X[yl == k, 1], s=7, color=c)
ax[1].set_aspect("equal"); ax[1].set_xticks([]); ax[1].set_yticks([])
savefig("spirals")
