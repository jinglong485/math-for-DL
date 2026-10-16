# Matrix Calculus for Deep Learning — Coding Assignments

Six assignments that build, in NumPy and from scratch, the matrix calculus
behind Bishop & Bishop, *Deep Learning: Foundations and Concepts*: gradient
checking, matrix functions, vector and matrix derivatives, Gaussians,
backpropagation, a small automatic-differentiation engine, and the backward
passes of attention, layer norm and normalising flows. The full handout is
`ASSIGNMENTS.pdf` in this folder.

## Setup

```bash
python -m pip install -r requirements.txt     # numpy, pytest (matplotlib optional)
```

## Layout

```
mc_assignments/              <- YOUR CODE: fill in everything marked TODO
    a1_matrix_functions.py       A1  gradient checking, matrix functions, identities  (10 points)
    a2_vector_derivatives.py     A2  gradients, Jacobians, Hessians, optimisation     (15 points)
    a3_matrix_derivatives.py     A3  derivatives w.r.t. matrices, Gaussians, KL       (20 points)
    a4_backprop.py               A4  layers, losses, an MLP trained by hand           (20 points)
    a5_autodiff.py               A5  dual numbers and a reverse-mode Tensor           (20 points)
    a6_layers.py                 A6  attention, layer/batch norm, flows, traces       (15 points)
tests/                       <- the test suite (do not edit)
grade.py                     <- runs the tests and prints your score
```

## Running the tests

```bash
python grade.py              # score for all six assignments
python grade.py a3           # only Assignment 3
python grade.py a4 -v        # Assignment 4 with full failure messages

pytest tests/test_a2_vector_derivatives.py -x    # stop at the first failure
pytest -k attention -v                           # only tests whose name contains "attention"
```

A fresh copy scores 0 / 100 with every test reported as "not done".
The whole suite runs in a few seconds once everything is implemented.

## The rules

* **NumPy only.** `numpy.linalg` (solve, inv, slogdet, cholesky, eigh, ...)
  is allowed. No `jax`, `torch`, `tensorflow`, `autograd`, `scipy`,
  `sklearn`, and not the Lab Manual's `mcalc` module: the tests read your
  source code and fail on those imports.
* **Every derivative is derived by hand.** The tests compare your gradients
  with finite differences computed by their own independent code.
* **Gradients have the shape of the variable** (the course convention): the
  gradient with respect to a 3 x 4 matrix is a 3 x 4 matrix.
* A test file's `test_rules` waits until every function of that assignment
  is implemented, then checks the imports.
* You may add private helper functions (names starting with `_`).
* Do not edit the tests. If a test seems wrong, re-read the docstring of the
  function: the docstrings are the specification.

## Suggested order and reading

| Assignment | Textbook chapters | Lab Manual part |
|---|---|---|
| A1 | 1–3, 9 (finite differences) | Parts 0–1 |
| A2 | 4–5 | Part 2 |
| A3 | 6–7 | Part 3 |
| A4 | 8 | Part 4 |
| A5 | 8–9 | Part 5 |
| A6 | 10–12 | Part 6 |

Do A1 first: its `numerical_gradient` is the tool you will use to debug
everything else.
