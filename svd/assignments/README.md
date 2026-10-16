# SVD & Pseudo-Inverse — Coding Assignments

Six assignments that build, from scratch, everything from dot products to the
SVD, the pseudo-inverse, least squares and PCA. The full handout is
`ASSIGNMENTS.pdf` in this folder.

## Setup

```bash
python -m pip install -r requirements.txt     # numpy, pytest (matplotlib optional)
```

## Layout

```
svd_assignments/        <- YOUR CODE: fill in the functions marked TODO
    a1_vectors.py           A1  vectors & matrices            (10 points)
    a2_orthogonal.py        A2  Gram-Schmidt, Householder QR  (15 points)
    a3_eigen.py             A3  power iteration, Jacobi       (15 points)
    a4_svd.py               A4  the SVD, two ways             (25 points)
    a5_pinv.py              A5  pseudo-inverse, least squares (20 points)
    a6_applications.py      A6  norms, low rank, ridge, PCA   (15 points)
tests/                  <- the test suite (do not edit)
grade.py                <- runs the tests and prints your score
```

## Running the tests

```bash
python grade.py              # score for all six assignments
python grade.py a3           # only Assignment 3
python grade.py a4 -v        # Assignment 4 with full failure messages

pytest tests/test_a2_orthogonal.py -x        # stop at the first failure
pytest -k jacobi -v                          # only tests whose name contains "jacobi"
```

A fresh copy scores 0 / 100 with every test reported as "not done".

## The rules

* **No `numpy.linalg` and no `scipy`** inside `svd_assignments/`. The tests
  check your source code and also make every `numpy.linalg` function raise an
  error while your code runs.
* In A1, `dot` and `matmul` must not use `@`, `np.dot`, `np.matmul`, ... .
  From A2 on, `@` is allowed.
* Later assignments may (and should) import your earlier functions, e.g.
  `from .a3_eigen import jacobi_eigh`. Do the assignments in order.
* Do not edit the tests. If a test seems wrong, re-read the docstring of the
  function: the docstrings are the specification.
