"""
pytest plugin that writes a machine-readable score report.

It is loaded by tools/collect_scores.py with `-p score_plugin` and writes the
JSON file named by the SCORE_JSON environment variable. It reads the
assignment titles and weights from the course's own tests/conftest.py
(WEIGHTS), and scores exactly like the text report printed there:

    points = weight * passed / (passed + failed + not done)

A test counts as "not done" when it fails with NotImplementedError.
The tests themselves are not changed.
"""
import json
import os
import re
import time

_tests = {}              # nodeid -> {"assignment", "outcome", "message"}
_collect_errors = []     # [{"file", "message"}]
_t0 = time.time()


def _assignment_of(nodeid, weights):
    base = os.path.basename(nodeid.split("::")[0])
    for key in weights:
        if base.startswith(f"test_{key}_"):
            return key
    return None


def _short(report):
    """A one-line description of why a test failed."""
    crash = getattr(report.longrepr, "reprcrash", None)
    text = crash.message if crash is not None else str(report.longrepr)
    lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
    line = lines[0] if lines else "failed"
    return line if len(line) <= 300 else line[:297] + "..."


def _collect_message(report):
    """'SyntaxError: invalid syntax (mc_assignments/a3_....py, line 108)' from a collection error."""
    text = str(report.longrepr)
    elines = [ln[1:].strip() for ln in text.splitlines() if ln.startswith("E ")]
    if not elines:
        return _short(report)
    msg = elines[-1]
    for ln in elines:
        m = re.search(r'File "(.+?)", line (\d+)', ln)
        if m:
            where = "/".join(m.group(1).replace("\\", "/").split("/")[-2:])
            msg += f" ({where}, line {m.group(2)})"
            break
    return msg if len(msg) <= 300 else msg[:297] + "..."


def pytest_collectreport(report):
    if report.failed:
        _collect_errors.append({"file": os.path.basename(str(report.nodeid)) or str(report.nodeid),
                                "message": _collect_message(report)})


def pytest_runtest_logreport(report):
    # same rule as the course conftest: the call phase, or a setup phase that did not pass
    if not (report.when == "call" or (report.when == "setup" and report.outcome != "passed")):
        return
    if report.passed:
        outcome, message = "passed", ""
    elif "NotImplementedError" in str(report.longrepr):
        outcome, message = "todo", ""
    else:
        outcome, message = "failed", _short(report)
    _tests[report.nodeid] = {"outcome": outcome, "message": message}


def _weights(config):
    for plugin in config.pluginmanager.get_plugins():
        w = getattr(plugin, "WEIGHTS", None)
        if isinstance(w, dict) and w:
            return w
    return {}


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("SCORE_JSON")
    if not path:
        return
    weights = _weights(session.config)
    assignments, failures = [], []
    total = 0.0
    for key, (title, weight) in weights.items():
        rows = {nid: t for nid, t in _tests.items() if _assignment_of(nid, weights) == key}
        counts = {k: sum(t["outcome"] == k for t in rows.values()) for k in ("passed", "failed", "todo")}
        n = sum(counts.values())
        errors = [e for e in _collect_errors if e["file"].startswith(f"test_{key}_")]
        points = weight * counts["passed"] / n if n else 0.0
        total += points
        code, _, name = title.partition(" ")
        assignments.append({
            "key": key, "code": code, "title": name.strip(), "weight": weight,
            "tests": n, **counts, "points": round(points, 2),
            "error": errors[0]["message"] if errors else None,
        })
        for nid, t in sorted(rows.items()):
            if t["outcome"] == "failed":
                failures.append({"assignment": key, "test": nid.split("::", 1)[-1],
                                 "file": nid.split("::")[0], "message": t["message"]})
    result = {
        "total": round(total, 2),
        "max": sum(w for _, w in weights.values()),
        "assignments": assignments,
        "failures": failures,
        "collection_errors": _collect_errors,
        "seconds": round(time.time() - _t0, 2),
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
