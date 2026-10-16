"""Shared helpers for the test suite (not collected as tests)."""
import ast
import inspect

import numpy as np

# Reference implementations, captured BEFORE the test fixtures block numpy.linalg.
# Tests use these to compute expected answers; your code may not use numpy.linalg.
ref_svd = np.linalg.svd
ref_eigvalsh = np.linalg.eigvalsh
ref_pinv = np.linalg.pinv
ref_lstsq = np.linalg.lstsq
ref_solve = np.linalg.solve
ref_qr = np.linalg.qr
ref_norm = np.linalg.norm
ref_cond = np.linalg.cond
ref_matrix_rank = np.linalg.matrix_rank

FORBIDDEN_MODULES = ("scipy", "numpy.linalg", "sklearn", "torch", "jax", "tensorflow")


def forbidden_usages(module):
    """Return a list of human-readable descriptions of forbidden imports/usages in a module."""
    tree = ast.parse(inspect.getsource(module))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(FORBIDDEN_MODULES):
                    problems.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [a.name for a in node.names]
            if mod.startswith(FORBIDDEN_MODULES) or (mod == "numpy" and "linalg" in names):
                problems.append(f"line {node.lineno}: from {mod} import {', '.join(names)}")
        elif isinstance(node, ast.Attribute) and node.attr == "linalg":
            problems.append(f"line {node.lineno}: uses .linalg")
    return problems


def builtin_product_usages(func):
    """Find uses of @, np.dot, np.matmul, ... inside one function (for A1's dot and matmul)."""
    banned = {"dot", "matmul", "inner", "einsum", "tensordot", "vdot"}
    tree = ast.parse(inspect.getsource(func).lstrip())
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            problems.append(f"the @ operator (line {node.lineno} of the function)")
        elif isinstance(node, ast.Attribute) and node.attr in banned:
            problems.append(f".{node.attr} (line {node.lineno} of the function)")
    return problems


def random_matrix(m, n, rank=None, seed=0, kappa=None):
    """Random m x n test matrix, optionally of a given rank or condition number."""
    rng = np.random.default_rng(seed)
    if rank is None and kappa is None:
        return rng.standard_normal((m, n))
    k = min(m, n) if rank is None else rank
    U = ref_qr(rng.standard_normal((m, m)))[0][:, :k]
    V = ref_qr(rng.standard_normal((n, n)))[0][:, :k]
    s = np.logspace(0, -np.log10(kappa), k) if kappa else np.sort(rng.uniform(0.5, 3.0, k))[::-1]
    return (U * s) @ V.T


def max_abs(M):
    M = np.asarray(M, dtype=float)
    return float(np.max(np.abs(M))) if M.size else 0.0


def assert_orthonormal_columns(Q, tol=1e-10, what="matrix"):
    Q = np.asarray(Q, dtype=float)
    k = Q.shape[1]
    err = max_abs(Q.T @ Q - np.eye(k))
    assert err <= tol, f"columns of {what} are not orthonormal: max|Q^T Q - I| = {err:.2e}"


def assert_parallel(v, w, tol=1e-8):
    """v and w are the same unit direction up to sign."""
    v = np.asarray(v, dtype=float).ravel(); w = np.asarray(w, dtype=float).ravel()
    v = v / np.sqrt(v @ v); w = w / np.sqrt(w @ w)
    assert abs(abs(v @ w) - 1.0) <= tol, f"{v} is not parallel to {w}"


def _still_a_stub(func):
    tree = ast.parse(inspect.getsource(func).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return True
    return False


def require_implemented(obj):
    """Raise NotImplementedError while a function (or any public function of a module)
    is still a stub, so that the 'rules' tests only score once the work is done."""
    if inspect.ismodule(obj):
        funcs = [f for name, f in inspect.getmembers(obj, inspect.isfunction)
                 if f.__module__ == obj.__name__ and not name.startswith("_")]
    else:
        funcs = [obj]
    todo = [f.__name__ for f in funcs if _still_a_stub(f)]
    if todo:
        raise NotImplementedError("rules are checked once these are implemented: " + ", ".join(todo))
