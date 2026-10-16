"""Run every demo script and capture its output in out/<name>.txt (figures go to figs/)."""
import glob, os, subprocess, sys, time
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(here, "out"), exist_ok=True)
env = dict(os.environ, PYTHONPATH=here)
for path in sorted(glob.glob(os.path.join(here, "demos", "d*.py"))):
    name = os.path.splitext(os.path.basename(path))[0]
    t0 = time.time()
    r = subprocess.run([sys.executable, path], cwd=here, env=env, capture_output=True, text=True)
    with open(os.path.join(here, "out", name + ".txt"), "w") as f:
        f.write(r.stdout.rstrip() + "\n")
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"{name:22s} {status}  ({time.time() - t0:.1f}s)")
    if r.returncode:
        print(r.stderr)
