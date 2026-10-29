# SVD Lab: Usage & Testing Guide

This guide explains the project layout in `svd/lab`, how to work through the exercises, how to run tests, and key tips for debugging.

---

## 1. Structure & Progression

The lab is divided into 7 sequential parts located in `svd/lab/exercises/`. Each part builds directly on the previous ones:

```text
svd/lab/
├── exercises/
│   ├── part1_vectors_matrices.py         <-- Exercise 1
│   ├── test_part1.py                     <-- Tests for Part 1
│   ├── part2_orthogonalisation_qr.py     <-- Exercise 2
│   ├── test_part2.py                     <-- Tests for Part 2
│   ├── part3_eigenvalues.py              <-- Exercise 3
│   ├── test_part3.py                     <-- Tests for Part 3
│   ├── part4_svd.py                      <-- Exercise 4
│   ├── test_part4.py                     <-- Tests for Part 4
│   ├── part5_pseudoinverse.py            <-- Exercise 5
│   ├── test_part5.py                     <-- Tests for Part 5
│   ├── part6_least_squares.py            <-- Exercise 6
│   ├── test_part6.py                     <-- Tests for Part 6
│   ├── part7_applications.py             <-- Exercise 7
│   └── test_part7.py                     <-- Tests for Part 7
├── demos/                                <-- Visual demos & numerical experiments
└── linalg_scratch.py                     <-- Complete reference implementations
```

---

## 2. How to Work Through the Exercises

1. Open the exercise file: e.g. `exercises/part1_vectors_matrices.py`.
2. Look for functions containing:
   ```python
   # === YOUR CODE HERE ===
   raise NotImplementedError("Implement me!")
   ```
3. Read the docstring and comments—they specify input shapes, return types, numerical edge cases, and hints.
4. Replace `raise NotImplementedError` with your implementation using pure NumPy array operations (avoid using `numpy.linalg`).
5. Reference code and algorithms can be studied in `svd/lab/linalg_scratch.py`.

> [!IMPORTANT]
> **Import note for Parts 4 to 7:**
> When implementing Parts 4, 5, 6, and 7, ensure `_as_float_matrix` is imported:
> ```python
> from part1_vectors_matrices import _as_float_matrix
> ```
> Wildcard imports (`from ... import *`) in Python skip names that start with an underscore unless specified in `__all__`.

---

## 3. How to Run the Tests

You can test each part individually as a standalone script or test the entire suite.

### Option A: Running parts individually (Recommended for exercises)

Run the corresponding test file from inside `svd/lab/exercises`:

```bash
cd /home/jinglong/coding/my_adventures/math-for-dl/svd/lab/exercises

# Part 1: Vectors, Matrices, and Gaussian Elimination
python test_part1.py

# Part 2: Gram-Schmidt & Householder QR
python test_part2.py

# Part 3: Eigenvalues & Jacobi
python test_part3.py

# Part 4: Singular Value Decomposition
python test_part4.py

# Part 5: Moore-Penrose Pseudo-Inverse
python test_part5.py

# Part 6: Least Squares & Regularisation
python test_part6.py

# Part 7: Applications (PCA, Norms, Low-rank)
python test_part7.py
```

### Option B: Running via Pytest

From the `svd/lab` root directory:

```bash
cd /home/jinglong/coding/my_adventures/math-for-dl/svd/lab

# Run all exercise tests
pytest -v exercises/

# Run a specific part's tests
pytest -v exercises/test_part1.py
```

---

## 4. Visual Demos

The `demos/` directory contains ready-to-run scripts illustrating geometric and numerical properties:

```bash
cd /home/jinglong/coding/my_adventures/math-for-dl/svd/lab

# Run all demos
python run_demos.py

# Or run individual demos:
python demos/d00_float.py           # Machine precision & floating point
python demos/d02_stability.py       # Numerical stability
python demos/d03_power.py           # Power iteration animation / convergence
python demos/d04_svd_examples.py    # Geometric transformation of unit circle/sphere
python demos/d07_pca.py             # Dimensionality reduction with PCA
```
