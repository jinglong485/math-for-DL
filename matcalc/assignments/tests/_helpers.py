"""Shared helpers for the test suite (not collected as tests).

The reference numerical gradient here is independent of your A1 code, so a
bug in your numerical_gradient cannot hide a bug elsewhere.
"""
import ast
import inspect
import textwrap

import numpy as np

FORBIDDEN_MODULES = ("jax", "torch", "tensorflow", "autograd", "scipy", "sklearn", "mcalc")


def num_grad(f, X, h=1e-6):
    """Reference central-difference gradient (any shape)."""
    X = np.array(X, dtype=float)
    G = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        old = X[idx]
        X[idx] = old + h; fp = f(X.copy())
        X[idx] = old - h; fm = f(X.copy())
        X[idx] = old
        G[idx] = (fp - fm) / (2 * h)
    return G


def num_jac(f, x, h=1e-6):
    """Reference Jacobian, flattened to (m, n)."""
    x = np.array(x, dtype=float)
    cols = []
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h; fp = np.ravel(f(x.copy()))
        x[idx] = old - h; fm = np.ravel(f(x.copy()))
        x[idx] = old
        cols.append((fp - fm) / (2 * h))
    return np.column_stack(cols)


def rel_err(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-12))


def assert_grad(analytic, numeric, tol=1e-6, what="gradient"):
    analytic = np.asarray(analytic, dtype=float); numeric = np.asarray(numeric, dtype=float)
    assert analytic.shape == numeric.shape, \
        f"{what} has shape {analytic.shape}, expected {numeric.shape} (the shape of the variable)"
    e = rel_err(analytic, numeric)
    assert e < tol, f"{what} disagrees with finite differences: relative error {e:.1e}"


def rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def spd(n, seed=0, cond=None):
    """A random symmetric positive definite matrix."""
    rng = np.random.default_rng(seed)
    if cond is None:
        B = rng.standard_normal((n, n))
        return B @ B.T + n * np.eye(n) * 0.1 + np.eye(n)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    return (Q * np.logspace(0, -np.log10(cond), n)) @ Q.T


def make_spirals(n_per_class=100, n_classes=3, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        th = np.linspace(k * 4.0, (k + 1) * 4.0, n_per_class) + rng.standard_normal(n_per_class) * noise
        X.append(np.column_stack([r * np.sin(th), r * np.cos(th)]))
        y.append(np.full(n_per_class, k))
    return np.vstack(X), np.concatenate(y)


# ------------------------------------------------------------- rule checks
def forbidden_usages(module):
    tree = ast.parse(inspect.getsource(module))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    problems.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in FORBIDDEN_MODULES:
                problems.append(f"line {node.lineno}: from {node.module} import ...")
    return problems


def _still_a_stub(func):
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return True
    return False


def require_implemented(module):
    """Raise NotImplementedError while any public function or method of a module is a stub."""
    funcs = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj):
            funcs.append(obj)
        elif inspect.isclass(obj):
            for m in vars(obj).values():
                if isinstance(m, (staticmethod, classmethod)):
                    m = m.__func__
                elif isinstance(m, property):
                    m = m.fget
                if inspect.isfunction(m):
                    funcs.append(m)
    todo = sorted({f.__qualname__ for f in funcs if _still_a_stub(f)})
    if todo:
        raise NotImplementedError("rules are checked once these are implemented: " + ", ".join(todo))
