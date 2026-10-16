import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, savefig
from linalg_scratch import svd

# a synthetic 'photograph': sky gradient, sun, hills, buildings, tiled road
h, w = 180, 270
y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
img = 0.85 - 0.45 * y
img[(x - 0.75) ** 2 + (y - 0.22) ** 2 < 0.006] = 1.0
img[y > 0.55 + 0.08 * np.sin(9 * x)] = 0.45
for x0, x1, top in [(0.08, 0.2, 0.35), (0.24, 0.31, 0.25), (0.42, 0.55, 0.4)]:
    img[(x > x0) & (x < x1) & (y > top) & (y < 0.75)] = 0.2
img[(y > 0.4 - 0.8 * np.abs(x - 0.85)) & (y < 0.75) & (np.abs(x - 0.85) < 0.1)] = 0.35  # a roof
tiles = 0.3 + 0.25 * ((np.floor(x * 24) + np.floor(y * 24)) % 2)
road = y > 0.82
img[road] = tiles[road]
img += 0.01 * np.random.default_rng(0).standard_normal(img.shape)   # sensor noise
img = np.clip(img, 0, 1)

U, s, Vt = svd(img)
ks = (2, 10, 30)
fig, ax = plt.subplots(1, 4, figsize=(8.0, 1.75))
for a, k in zip(ax, (None,) + ks):
    shown = img if k is None else (U[:, :k] * s[:k]) @ Vt[:k]
    a.imshow(shown, cmap="gray", vmin=0, vmax=1); a.axis("off")
    a.set_title("original" if k is None else f"rank {k}", fontsize=8)
savefig("image_compression")
plt.figure(figsize=(6.0, 2.6))
plt.semilogy(range(1, s.size + 1), s, color=C1)
plt.xlabel("$i$"); plt.ylabel("$\\sigma_i$"); savefig("image_singular_values")

total = np.sum(s ** 2)
print("image size %d x %d = %d numbers" % (h, w, h * w))
print("  k   storage   % of original   rel. Frobenius error")
for k in (1, 2, 5, 10, 20, 30, 50):
    store = k * (h + w + 1)
    print("%3d  %8d   %6.1f%%          %.4f"
          % (k, store, 100 * store / (h * w), np.sqrt(np.sum(s[k:] ** 2) / total)))
