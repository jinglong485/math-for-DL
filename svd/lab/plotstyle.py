"""Shared Matplotlib style for the Lab Manual figures (import it before plotting)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical colours, always used in this order
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.figsize": (6.0, 3.4),
    "figure.dpi": 150,
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def savefig(name):
    """Save the current figure as figs/<name>.pdf and close it."""
    os.makedirs("figs", exist_ok=True)
    plt.savefig(os.path.join("figs", name + ".pdf"))
    plt.close()
