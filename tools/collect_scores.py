"""
Run every course's test suite and write the data behind the progress page.

    python tools/collect_scores.py            (from the repository root)

Writes, inside site/:
    data/scores.json       the latest result of every course and assignment
    data/history.json      one entry per commit (appended to the previous file)
    data/badge-*.json      shields.io endpoint badges
    books/*.pdf            copies of the course books, so the page can link them

On GitHub Actions it also writes a job summary and two step outputs
(`regressed`, `message`) that the workflow uses to flag a score that went down.
It always exits with status 0 unless something is wrong with the tooling
itself: unfinished assignments are expected to fail their tests.
"""
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
SITE = os.path.join(ROOT, "site")
DATA = os.path.join(SITE, "data")

COURSES = [
    {
        "id": "svd",
        "name": "The SVD and the Pseudo-Inverse",
        "short": "SVD",
        "dir": "svd/assignments",
        "books": [["Textbook", "SVD_Textbook.pdf"],
                  ["Python Lab Manual", "SVD_Python_Lab_Manual.pdf"],
                  ["Assignment handout", "SVD_Coding_Assignments.pdf"]],
    },
    {
        "id": "matcalc",
        "name": "Matrix Calculus for Deep Learning",
        "short": "Matrix calculus",
        "dir": "matcalc/assignments",
        "books": [["Textbook", "MatCalc_Textbook.pdf"],
                  ["NumPy Lab Manual", "MatCalc_NumPy_Lab_Manual.pdf"],
                  ["Assignment handout", "MatCalc_Coding_Assignments.pdf"]],
    },
]
MAX_HISTORY = 2000


# ------------------------------------------------------------------ helpers
def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def repo_url():
    server, repo = os.environ.get("GITHUB_SERVER_URL"), os.environ.get("GITHUB_REPOSITORY")
    if server and repo:
        return f"{server}/{repo}"
    remote = git("remote", "get-url", "origin")
    if not remote:
        return None
    m = re.match(r"(?:git@|https://)([^/:]+)[/:](.+?)(?:\.git)?/?$", remote)
    return f"https://{m.group(1)}/{m.group(2)}" if m else None


def tests_fingerprint(course_dir):
    """sha256 over the course's test files (names and contents)."""
    h = hashlib.sha256()
    tests = os.path.join(course_dir, "tests")
    for name in sorted(os.listdir(tests)):
        if name.endswith(".py"):
            h.update(name.encode())
            with open(os.path.join(tests, name), "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def r1(x):
    """A score rounded to one decimal, without trailing zeros: 97.65 -> '97.7', 10.0 -> '10'."""
    return f"{round(x, 1):g}"


def badge(label, total, maximum):
    frac = total / maximum if maximum else 0.0
    color = ("brightgreen" if frac >= 0.9 else "green" if frac >= 0.6 else
             "yellow" if frac >= 0.3 else "orange" if frac > 0 else "lightgrey")
    return {"schemaVersion": 1, "label": label, "message": f"{total:g} / {maximum:g}", "color": color}


# ------------------------------------------------------------------ run one course
def run_course(course):
    cdir = os.path.join(ROOT, course["dir"])
    out_json = os.path.join(DATA, f"_raw_{course['id']}.json")
    if os.path.exists(out_json):
        os.remove(out_json)
    env = dict(os.environ, SCORE_JSON=out_json,
               PYTHONPATH=os.pathsep.join([TOOLS, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
               PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=line", "-rN", "-p", "no:cacheprovider",
           "--continue-on-collection-errors", "-p", "score_plugin"]
    print(f"\n### {course['name']}  ({course['dir']})", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=cdir, env=env, capture_output=True, text=True, timeout=1800)
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        output = "the test suite took longer than 30 minutes and was stopped"
    start = output.rfind("\n", 0, output.find("score report")) + 1
    report = output[start:] if "score report" in output else output[-3000:]
    print(report.rstrip(), flush=True)

    raw = load_json(out_json, None)
    if os.path.exists(out_json):
        os.remove(out_json)
    if raw is None:                                   # pytest did not even start properly
        raw = {"total": 0.0, "max": 100, "assignments": [], "failures": [],
               "collection_errors": [{"file": "", "message": output.strip().splitlines()[-1] if output.strip() else "pytest failed"}],
               "seconds": 0.0}
    fp = tests_fingerprint(cdir)
    shipped = load_json(os.path.join(TOOLS, "fingerprints.json"), {}).get(course["id"])
    raw.update({
        "id": course["id"], "name": course["name"], "short": course["short"], "dir": course["dir"],
        "books": [{"label": lbl, "file": "books/" + fn} for lbl, fn in course["books"]],
        "tests_fingerprint": fp,
        "tests_modified": shipped is not None and fp != shipped,
    })
    if raw["tests_modified"]:
        print(f"::warning title=Tests changed::The tests of {course['dir']} differ from the shipped version.")
    return raw


# ------------------------------------------------------------------ main
def main():
    os.makedirs(DATA, exist_ok=True)
    courses = [run_course(c) for c in COURSES]

    # copies of the books for the page
    books_out = os.path.join(SITE, "books")
    os.makedirs(books_out, exist_ok=True)
    for c in COURSES:
        for _, fn in c["books"]:
            src = os.path.join(ROOT, "books", fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(books_out, fn))

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    sha = os.environ.get("GITHUB_SHA") or git("rev-parse", "HEAD")
    message = git("log", "-1", "--format=%s") if sha else None
    url = repo_url()
    run_url = None
    if os.environ.get("GITHUB_RUN_ID") and url:
        run_url = f"{url}/actions/runs/{os.environ['GITHUB_RUN_ID']}"

    totals = {c["id"]: c["total"] for c in courses}
    points = {c["id"]: {a["key"]: a["points"] for a in c["assignments"]} for c in courses}

    # history: one entry per commit; a re-run of the same commit replaces its entry
    hist_path = os.path.join(DATA, "history.json")
    history = load_json(hist_path, [])
    if not isinstance(history, list):
        history = []
    if history and sha and history[-1].get("sha") == sha:
        history.pop()
    previous = history[-1] if history else None
    entry = {"time": now, "sha": sha, "message": message, "run_url": run_url,
             "totals": totals, "points": points}
    history = (history + [entry])[-MAX_HISTORY:]

    # did anything go down since the previous commit?
    drops = []
    if previous:
        for c in courses:
            before = previous.get("totals", {}).get(c["id"])
            if before is not None and c["total"] < before - 1e-9:
                parts = []
                for a in c["assignments"]:
                    pb = previous.get("points", {}).get(c["id"], {}).get(a["key"])
                    if pb is not None and a["points"] < pb - 1e-9:
                        parts.append(f"{a['code']} {r1(pb)} -> {r1(a['points'])}")
                drops.append(f"{c['short']} {r1(before)} -> {r1(c['total'])}" + (f" ({', '.join(parts)})" if parts else ""))
    regressed = bool(drops)
    summary = ("Score went down: " + "; ".join(drops)) if regressed else \
        "Scores: " + ", ".join(f"{c['short']} {r1(c['total'])}/{c['max']:g}" for c in courses)

    scores = {
        "generated": now,
        "commit": {"sha": sha, "short": sha[:7] if sha else None, "message": message,
                   "url": f"{url}/commit/{sha}" if url and sha else None},
        "run_url": run_url,
        "repo_url": url,
        "previous": {"time": previous["time"], "totals": previous.get("totals", {}),
                     "points": previous.get("points", {})} if previous else None,
        "regressed": regressed,
        "message": summary,
        "courses": courses,
    }
    write_json(os.path.join(DATA, "scores.json"), scores)
    write_json(hist_path, history)
    grand = sum(c["total"] for c in courses)
    grand_max = sum(c["max"] for c in courses)
    write_json(os.path.join(DATA, "badge-total.json"), badge("course progress", round(grand, 1), grand_max))
    for c in courses:
        write_json(os.path.join(DATA, f"badge-{c['id']}.json"), badge(c["short"], round(c["total"], 1), c["max"]))

    # GitHub Actions extras
    gh_out, gh_sum = os.environ.get("GITHUB_OUTPUT"), os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"regressed={'true' if regressed else 'false'}\n")
            f.write(f"message={summary}\n")
    if gh_sum:
        lines = []
        for c in courses:
            lines += [f"### {c['name']}: {r1(c['total'])} / {c['max']:g}", "",
                      "| Assignment | Passed | Failed | Not done | Points |",
                      "|---|---:|---:|---:|---:|"]
            for a in c["assignments"]:
                name = f"{a['code']} {a['title']}" + (" (error)" if a["error"] else "")
                lines.append(f"| {name} | {a['passed']} | {a['failed']} | {a['todo']} | {r1(a['points'])} / {a['weight']} |")
            if c["tests_modified"]:
                lines.append("\n> **Note:** the tests of this course have been modified.")
            lines.append("")
        lines.append(f"**{summary}**")
        with open(gh_sum, "a") as f:
            f.write("\n".join(lines) + "\n")

    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
