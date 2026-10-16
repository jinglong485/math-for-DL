"""
Test-suite configuration:
  * blocks numpy.linalg (and scipy.linalg) while YOUR code runs,
  * stops any single test after 60 seconds (on systems with SIGALRM),
  * prints a score report per assignment at the end of every run.
"""
import os
import signal
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))          # make _helpers importable
import _helpers  # noqa: E402,F401  (captures the reference functions first)

WEIGHTS = {
    "a1": ("A1  Vectors & matrices", 10),
    "a2": ("A2  Orthogonality & QR", 15),
    "a3": ("A3  Eigenvalues", 15),
    "a4": ("A4  The SVD", 25),
    "a5": ("A5  Pseudo-inverse & least squares", 20),
    "a6": ("A6  Applications", 15),
}
_results = {}            # assignment -> {"passed": n, "failed": n, "todo": n}


class ForbiddenFunctionError(AssertionError):
    pass


def _forbidden(name):
    def f(*args, **kwargs):
        raise ForbiddenFunctionError(
            f"numpy.linalg.{name} was called. Your solution must not use numpy.linalg "
            f"-- implement it from scratch.")
    return f


@pytest.fixture(autouse=True)
def block_numpy_linalg(monkeypatch):
    for name in dir(np.linalg):
        obj = getattr(np.linalg, name)
        if not name.startswith("_") and callable(obj) and not isinstance(obj, type):
            monkeypatch.setattr(np.linalg, name, _forbidden(name))
    try:
        import scipy.linalg as sla
        for name in dir(sla):
            obj = getattr(sla, name)
            if not name.startswith("_") and callable(obj) and not isinstance(obj, type):
                monkeypatch.setattr(sla, name, _forbidden("scipy.linalg." + name))
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def time_limit():
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    def handler(signum, frame):
        raise TimeoutError("test took longer than 60 s -- is there an infinite loop?")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _assignment_of(nodeid):
    base = os.path.basename(nodeid.split("::")[0])
    for key in WEIGHTS:
        if base.startswith(f"test_{key}_"):
            return key
    return None


def pytest_runtest_logreport(report):
    key = _assignment_of(report.nodeid)
    if key is None:
        return
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        r = _results.setdefault(key, {"passed": 0, "failed": 0, "todo": 0})
        if report.passed:
            r["passed"] += 1
        elif "NotImplementedError" in str(report.longrepr):
            r["todo"] += 1
        else:
            r["failed"] += 1


def pytest_terminal_summary(terminalreporter):
    if not _results:
        return
    tr = terminalreporter
    tr.write_sep("=", "SVD assignments: score report")
    tr.write_line(f"{'Assignment':38s}{'passed':>8s}{'failed':>8s}{'not done':>10s}{'points':>14s}")
    total = maximum = 0.0
    for key, (title, weight) in WEIGHTS.items():
        if key not in _results:
            continue
        r = _results[key]
        n = r["passed"] + r["failed"] + r["todo"]
        pts = weight * r["passed"] / n if n else 0.0
        total += pts
        maximum += weight
        tr.write_line(f"{title:38s}{r['passed']:>8d}{r['failed']:>8d}{r['todo']:>10d}"
                      f"{pts:>9.1f} / {weight:<3d}")
    tr.write_line("-" * 78)
    tr.write_line(f"{'TOTAL':38s}{'':26s}{total:>9.1f} / {maximum:<3.0f}")
    if maximum < 100:
        tr.write_line("(only some assignments were run; run plain `pytest` for the full score)")
