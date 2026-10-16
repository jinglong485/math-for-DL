"""
Run the test suite and print your score.

    python grade.py            all six assignments
    python grade.py a3         only Assignment 3
    python grade.py a4 -v      Assignment 4, with the full failure messages
"""
import glob
import os
import sys

import pytest

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
args = sys.argv[1:]
verbose = "-v" in args
selected = [a.lower() for a in args if a.lower() in {"a1", "a2", "a3", "a4", "a5", "a6"}]
files = []
for a in selected:
    files += glob.glob(os.path.join("tests", f"test_{a}_*.py"))
options = ["--tb=short", "-q"] if verbose else ["--tb=no", "-q", "--no-header", "-rN"]
sys.exit(pytest.main(options + ["-p", "no:cacheprovider"] + (files or ["tests"])))
