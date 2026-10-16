import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, C3, INK2, savefig
from mcalc import gradient_descent, newton, logistic_loss, sigmoid

np.set_printoptions(precision=4, suppress=True)
H = np.diag([1.0, 10.0]); grad = lambda w: H @ w                       # Worked Example 5.1
for lr in (0.1, 0.19, 0.21):
    W = gradient_descent(grad, [1.0, 1.0], lr, 3)
    print("lr=%.2f  iterates:" % lr, " ".join("(%.3f, %.3f)" % tuple(w) for w in W))

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
g1, g2 = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.2, 1.2, 200))
ax[0].contour(g1, g2, 0.5 * (g1 ** 2 + 10 * g2 ** 2), levels=12, colors="#c9c7c0", linewidths=0.6)
for lr, c in zip((0.05, 0.18), (C1, C2)):
    W = gradient_descent(grad, [-1.0, 1.0], lr, 25)
    ax[0].plot(W[:, 0], W[:, 1], "o-", ms=2.5, color=c, label="$\\eta=%.2f$" % lr)
ax[0].set_aspect("equal"); ax[0].legend(fontsize=7); ax[0].set_title("paths on $H=$diag$(1,10)$", fontsize=8)
for kappa, c in zip((2, 10, 100), (C1, C2, C3)):
    Hk = np.diag([1.0, kappa])
    lr = 2 / (1 + kappa)
    W = gradient_descent(lambda w: Hk @ w, [1.0, 1.0], lr, 60)
    ax[1].semilogy(np.linalg.norm(W, axis=1), color=c, label="$\\kappa=%d$" % kappa)
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("$\\|w_k-w^\\star\\|$"); ax[1].legend(fontsize=7)
ax[1].set_title("optimal step, three condition numbers", fontsize=8)
savefig("gd_paths")

# logistic regression: Newton (IRLS) versus gradient descent
x = np.array([1.0, 2.0, -1.0, -2.0]); t = np.array([1.0, 0.0, 0.0, 1.0])   # Worked Example 5.2
Phi = x[:, None]
g = lambda w: logistic_loss(w, Phi, t)[1]; h = lambda w: logistic_loss(w, Phi, t)[2]
print("Newton iterates:", newton(g, h, [0.0], 4).ravel())

rng = np.random.default_rng(0)
N = 200
Xd = rng.standard_normal((N, 2)) * [1.0, 4.0]                       # badly scaled features
Phi = np.column_stack([np.ones(N), Xd])
w_true = np.array([0.5, 2.0, -0.5])
t = (rng.random(N) < sigmoid(Phi @ w_true)).astype(float)
Wn = newton(lambda w: logistic_loss(w, Phi, t)[1], lambda w: logistic_loss(w, Phi, t)[2], np.zeros(3), 8)
Wg = gradient_descent(lambda w: logistic_loss(w, Phi, t)[1] / N, np.zeros(3), 0.5, 2000)
w_star = Wn[-1]
print("Newton: |w_k - w*| =", " ".join("%.1e" % np.linalg.norm(w - w_star) for w in Wn[:7]))
for k in (10, 100, 1000, 2000):
    print("GD after %4d steps: |w - w*| = %.1e" % (k, np.linalg.norm(Wg[k] - w_star)))
