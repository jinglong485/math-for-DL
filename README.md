# Math for Deep Learning

Two self-study courses with textbooks, NumPy labs and auto-graded coding
assignments:

* **The SVD and the Pseudo-Inverse**: linear algebra from vectors to PCA,
  every algorithm written from scratch.
* **Matrix Calculus for Deep Learning**: the derivatives behind Bishop &
  Bishop, *Deep Learning: Foundations and Concepts*, from gradient checking
  to backpropagation, autodiff and attention.

Every push to GitHub runs both test suites and publishes your scores to a
progress page on GitHub Pages.

<!-- After setup, replace <you> with your GitHub user name to show live score badges:
![Progress](https://img.shields.io/endpoint?url=https://<you>.github.io/math-for-dl/data/badge-total.json)
![SVD](https://img.shields.io/endpoint?url=https://<you>.github.io/math-for-dl/data/badge-svd.json)
![Matrix calculus](https://img.shields.io/endpoint?url=https://<you>.github.io/math-for-dl/data/badge-matcalc.json)
-->

## What is in here

```
books/                         the six PDFs: textbook, lab manual and assignment handout per course
svd/
  assignments/                 YOUR CODE for the SVD course (svd_assignments/), its tests, grade.py
  lab/                         the lab manual's code (linalg_scratch.py and demos)
matcalc/
  assignments/                 YOUR CODE for the matrix-calculus course (mc_assignments/), tests, grade.py
  lab/                         the lab manual's code (mcalc.py and demos)
site/index.html                the progress page
tools/collect_scores.py        runs both test suites and writes the page's data
tools/score_plugin.py          pytest plugin that records each test result as JSON
tools/fingerprints.json        checksums of the original tests (the page shows if they changed)
.github/workflows/grade.yml    the automation: test, publish, check
```

Start with the SVD course (`books/SVD_Textbook.pdf`) if linear algebra is
rusty, or go straight to `books/MatCalc_Textbook.pdf`. Each course's
`ASSIGNMENTS.pdf` (also in `books/`) explains the assignments and the rules.

---

## One-time setup (about 10 minutes)

You need **Git**, **Python 3.10 or newer**, and a free **GitHub account**.

### 1. Make this folder a Git repository

```bash
cd math-for-dl
git init
git add .
git commit -m "Initial commit"
git branch -M main
```

If Git asks who you are, run these once and commit again:

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

### 2. Create an empty repository on GitHub

Go to <https://github.com/new>:

* **Repository name:** `math-for-dl` (any name works; the page address uses it)
* **Public**: GitHub Pages on a free account only publishes from public
  repositories. Your code and scores will be visible to anyone who looks.
* Do **not** tick "Add a README", ".gitignore" or "license". The repository
  must be empty.

### 3. Push

```bash
git remote add origin https://github.com/<you>/math-for-dl.git
git push -u origin main
```

The first push asks you to sign in. A browser window (Git Credential Manager)
or a personal access token used as the password both work. GitHub Desktop
or `gh auth login` also work.

### 4. Turn on GitHub Pages

On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

### 5. Run the workflow once

Your push in step 3 already started the workflow, before Pages was switched on,
so that first run fails at **Configure Pages**. That is expected. Now:

**Actions → Grade and publish → Run workflow → Run workflow**

After a minute or two all three jobs are green. Your page is at

```
https://<you>.github.io/math-for-dl/
```

(the link is also shown on the run's summary page and under Settings → Pages).
A fresh repository shows 0 / 100 for both courses.

### 6. (Optional) Score badges

Edit the comment block near the top of this README: put your user name in
place of `<you>` and move the three badge lines out of the comment.

---

## Day-to-day

```bash
cd matcalc/assignments                 # or svd/assignments
python -m pip install -r requirements.txt     # once
python grade.py a2                     # work on an assignment until you are happy
cd ../..
git add -A
git commit -m "Matcalc A2: logistic loss and Newton"
git push
```

* The **Actions** tab shows the run: yellow while it runs, then green or red.
  Each run's summary page has a score table.
* The page updates about a minute after the run finishes. If it looks stale,
  hard-refresh (Ctrl+Shift+R or Cmd+Shift+R).
* Commit messages appear on the page and in its chart's tooltips, so write
  ones you will want to read later.

### What the three jobs do

| Job | What it does | When it is red |
|---|---|---|
| **Run the tests** | Runs both test suites on a fresh Linux machine and computes the scores. Failing tests are normal while you work, so they never make this job fail. | Only if the tooling itself breaks. |
| **Publish the page** | Deploys `site/` with the new scores to GitHub Pages. | Pages is not set up (see Troubleshooting). |
| **Score did not go down** | Compares with your previous push. | A course's total dropped, e.g. an edit broke an assignment that used to pass. The page is still updated; the red ✗ is the alarm. |

### Where the history lives

The runner starts empty every time, so each run downloads `data/history.json`
from your live page, adds the new result and publishes it again. One point
per push; re-running the same commit replaces its point instead of adding a
new one. If you ever switch Pages off, the history starts again from zero.

### Preview the page on your computer

```bash
python tools/collect_scores.py          # from the repository root
python -m http.server -d site 8000      # then open http://localhost:8000
```

Local runs write `site/data/` and `site/books/`, which Git ignores.

---

## The honor system

* The score comes from tests that live in your own repository. Nothing stops
  you from editing them, so the page shows a check for each course: "tests
  unchanged from the course release" or a warning if they differ.
* The reference solutions are **not** in this repository. They are in the
  `solutions-DO-NOT-PUSH` folder next to it, as two zip files. Keep them out
  of this folder: anything you commit here is public.

## Troubleshooting

**"Get Pages site failed" or "Not Found" at Configure Pages.** Pages is not
switched on with the GitHub Actions source yet. Do step 4, then re-run the
workflow (step 5).

**Nothing happens after a push.** The workflow runs on the `main` branch only.
`git branch` should show `* main`; if it shows `master`, run
`git branch -M main` and `git push -u origin main`.

**The `.github` folder seems to be missing.** Folders starting with a dot are
hidden on macOS and Linux. `ls -a` shows it. It must be committed, or
there is no automation.

**"Branch is not allowed to deploy to github-pages".** Pages only accepts
deployments from the default branch. Make sure `main` is the default branch
(Settings → General).

**The repository is private and Pages is unavailable.** On a free account,
Pages needs a public repository. GitHub Pro allows private repositories, but
the page itself is still public.

**A test passes locally but fails on GitHub.** The workflow uses Python 3.12
on Linux with the latest NumPy. Check the failure message in the run log
(Actions → the run → Run the tests). Typical causes are a file you forgot to
`git add`, or code that depends on your local folder layout.

**Cost.** GitHub Actions and Pages are free for public repositories.
