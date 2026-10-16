import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, INK2, savefig
from linalg_scratch import pca

np.set_printoptions(precision=4, suppress=True)
X = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1], [0, 0]])   # Worked Example 10.4
comps, var, scores, mean = pca(X, 2)
print("principal directions (rows):\n", comps)
print("variances:", var, "  explained:", var / var.sum())
print("scores on PC1:", scores[:, 0])

rng = np.random.default_rng(0)                          # a larger, noisier cloud
Z = rng.standard_normal((300, 2)) * [2.0, 0.5]
th = np.radians(30)
D = Z @ np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) + [1, 2]
comps, var, _, mean = pca(D, 2)
print("cloud: directions\n", comps, "\n std devs:", np.sqrt(var))
plt.figure(figsize=(4.2, 3.4))
plt.scatter(D[:, 0], D[:, 1], s=6, color="#9a9890")
for c, sd, col, lab in zip(comps, np.sqrt(var), (C1, C2), ("PC1", "PC2")):
    plt.annotate("", mean + 2 * sd * c, mean,
                 arrowprops=dict(arrowstyle="->", color=col, lw=2))
    plt.text(*(mean + 2.2 * sd * c), lab, color=INK2)
plt.axis("equal"); savefig("pca_cloud")
