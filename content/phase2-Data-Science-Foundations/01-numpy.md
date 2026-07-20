# Phase 2 · Lesson 1 — NumPy

> Prerequisite: Phase 1 (especially Python Fundamentals, Advanced Python)

---

## 1. Introduction

### What is NumPy?
NumPy ("Numerical Python") is the foundational library for numerical computing in Python, providing the `ndarray` — a fixed-type, contiguous, multi-dimensional array — and a huge library of vectorized operations over it. It is the substrate literally everything else in this curriculum sits on: pandas DataFrames are backed by NumPy arrays internally (or Arrow, increasingly, but NumPy remains foundational); PyTorch tensors mirror NumPy's API almost 1:1 by deliberate design; scikit-learn expects NumPy arrays as its core data representation.

### Why does it exist?
Pure Python lists are arrays of *pointers to objects* — flexible but catastrophically slow and memory-inefficient for numerical work (each element is a full Python object with type info, refcount, etc.). NumPy (Travis Oliphant, 2005, unifying earlier Numeric/Numarray libraries) stores raw, homogeneously-typed data in contiguous memory and pushes loops down into compiled C, achieving 10-100x+ speedups for numerical code.

### Historical background
NumPy descends from Numeric (1995) → Numarray → NumPy (2005 merger). It became the de facto numerical standard partly because SciPy, matplotlib, pandas, and scikit-learn all built directly on its array interface, creating a powerful network effect that persists into 2026's LLM/deep-learning stack (PyTorch's tensor API is intentionally NumPy-familiar).

### Real-world motivation
Every feature matrix you feed into an XGBoost model, every gradient computation in a from-scratch neural network (Phase 5), and every large-scale actuarial simulation (Monte Carlo mortality/lapse modeling) either directly uses NumPy or a library whose API and mental model is NumPy's.

---

## 2. Theory

### The `ndarray`: what makes it fast
An `ndarray` is: a pointer to a single contiguous block of memory, a `dtype` (fixed element type — `float64`, `int32`, etc.), a `shape` tuple, and a `strides` tuple (bytes to step to move one index along each axis). Because every element has the same type and fixed size, NumPy can compute any element's memory address in $O(1)$ via:
$$
\text{address}(i_1, \dots, i_n) = \text{base} + \sum_{k=1}^{n} i_k \cdot \text{strides}_k
$$

### Vectorization
"Vectorization" means expressing a computation as whole-array operations instead of explicit Python-level loops, letting NumPy dispatch to tight compiled C loops (often using SIMD instructions on the CPU). `a + b` on two arrays performs the loop entirely in C; `[x + y for x, y in zip(a, b)]` performs it with full Python interpreter overhead per element.

### Broadcasting
A set of rules letting NumPy operate on arrays of different (but compatible) shapes without copying data:
1. Align shapes from the *trailing* dimension.
2. Dimensions are compatible if equal, or one of them is 1.
3. Size-1 dimensions are (virtually) stretched to match, with **no actual memory duplication** — implemented via zero strides.

### Views vs. copies
Slicing (`a[1:3]`) returns a **view** (shares memory with the original — mutating the view mutates the original); fancy indexing (`a[[1, 3, 5]]`) and boolean masking return **copies**. This distinction is a very common, very subtle bug source.

---

## 3. Mathematical Foundations

### Linear algebra as NumPy's core vocabulary
NumPy directly implements vector/matrix operations foundational to virtually all ML math (deepened formally in Phase 3):
$$
(A B)_{ij} = \sum_{k} A_{ik} B_{kj} \qquad \text{— matrix multiplication, } O(n^3) \text{ naively}
$$
NumPy's `@`/`np.matmul` dispatches to BLAS (Basic Linear Algebra Subprograms — highly optimized C/Fortran, often further using multi-threading and hardware-specific SIMD/AVX instructions), which is why `A @ B` on 1000×1000 matrices is dramatically faster than any hand-written Python triple loop performing the same $O(n^3)$ work — same complexity class, vastly better constant factor.

### Numerical stability
Floating-point arithmetic (IEEE-754, revisited from Phase 1 Lesson 1) means naive formulas can be numerically unstable. Example: computing variance via
$$
\text{Var}(X) = E[X^2] - (E[X])^2
$$
can suffer catastrophic cancellation when $E[X^2]$ and $(E[X])^2$ are both large and close in value, losing precision. NumPy's `np.var` uses a numerically stable two-pass algorithm (compute mean first, then sum squared deviations) specifically to avoid this — a small but real illustration of why "just use the library function" beats reimplementing textbook formulas naively.

### Complexity of core operations
| Operation | Complexity |
|---|---|
| Element-wise op (`a + b`) | $O(n)$, highly parallelized constant factor |
| Matrix multiply ($n \times n$) | $O(n^3)$ naive; $O(n^{2.37})$ theoretical (Coppersmith–Winograd family, not used in practice); libraries use $O(n^3)$ with excellent constants (BLAS) |
| Sorting (`np.sort`) | $O(n \log n)$ |
| `np.linalg.solve` (linear system) | $O(n^3)$ via LU decomposition |

---

## 4. Algorithm — Broadcasting Resolution (step by step)

```
GIVEN arrays with shapes A: (8, 1, 6, 1), B: (7, 1, 5)
1. Right-align shapes (pad shorter shape with 1s on the LEFT):
     A: (8, 1, 6, 1)
     B: (1, 7, 1, 5)
2. Compare each dimension pair (from the right):
     dim -1: 1 vs 5 -> broadcast to 5
     dim -2: 6 vs 1 -> broadcast to 6
     dim -3: 1 vs 7 -> broadcast to 7
     dim -4: 8 vs 1 -> broadcast to 8
3. If any pair is neither equal NOR one of them 1 -> ValueError: shapes incompatible
4. Result shape: (8, 7, 6, 5)
```
This deterministic procedure is why `np.broadcast_shapes` can predict the output shape *before* any computation runs — worth internalizing so broadcasting errors stop feeling mysterious.

---

## 5. Python Implementation

```python
"""numpy_core.py — vectorized patterns you will use constantly"""
import numpy as np

# --- Creation ---
ages = np.array([45, 62, 30, 71, 55], dtype=np.float64)
claims = np.random.default_rng(seed=42).normal(loc=5000, scale=1500, size=1000)  # modern RNG API

# --- Vectorized operations (no Python-level loop) ---
age_normalized = (ages - ages.mean()) / ages.std()          # z-score normalization
risk_flag = (ages > 60) & (claims[:5] > 4000)                # vectorized boolean logic, elementwise AND

# --- Broadcasting example: subtract a per-column mean from a 2D matrix ---
feature_matrix = np.random.default_rng(0).normal(size=(1000, 4))    # 1000 samples, 4 features
col_means = feature_matrix.mean(axis=0)                              # shape (4,)
centered = feature_matrix - col_means                                 # broadcasts (4,) across (1000, 4)

# --- Efficient aggregation without loops ---
total_claims_by_bucket = np.zeros(5)
bucket_idx = np.clip((ages // 20).astype(int), 0, 4)
np.add.at(total_claims_by_bucket, bucket_idx, claims[:5])   # scatter-add, vectorized histogram-style op

# --- Linear algebra ---
X = feature_matrix
y = np.random.default_rng(1).normal(size=1000)
# Closed-form OLS: beta = (X^T X)^-1 X^T y  (Normal Equation, revisited formally in Phase 3/4)
beta = np.linalg.solve(X.T @ X, X.T @ y)

print("z-scored ages:", age_normalized.round(2))
print("OLS coefficients:", beta.round(3))
```

**Line-by-line notes:**
- `np.random.default_rng(seed=42)` is the modern (2019+) NumPy random API — preferred over the legacy `np.random.seed()` global-state API because it gives reproducible, independent random streams (critical for parallel Monte Carlo simulations, directly relevant to actuarial work).
- `np.linalg.solve(A, b)` solves $Ax = b$ directly via LU decomposition — always prefer this over explicitly computing `np.linalg.inv(A) @ b`, which is both slower and less numerically stable.

---

## 6. Build From Scratch

**A minimal strided-array class (to demystify what makes `ndarray` fast):**
```python
import ctypes

class TinyArray:
    """Simplified 1D float array backed by a raw contiguous C buffer — the core idea behind ndarray."""
    def __init__(self, data: list[float]):
        self.n = len(data)
        self.buffer = (ctypes.c_double * self.n)(*data)   # contiguous C memory, not Python objects

    def __getitem__(self, i: int) -> float:
        return self.buffer[i]              # O(1) direct memory access, no per-element Python object

    def __add__(self, other: "TinyArray") -> "TinyArray":
        result = [self.buffer[i] + other.buffer[i] for i in range(self.n)]
        return TinyArray(result)             # still Python-loop bound (real NumPy pushes this into C)

a = TinyArray([1.0, 2.0, 3.0])
b = TinyArray([10.0, 20.0, 30.0])
c = a + b
print([c[i] for i in range(3)])
```
This exposes the essential idea (contiguous, homogeneously-typed memory) while honestly falling short of real NumPy's speed, since the `__add__` loop here still runs in the Python interpreter — real NumPy's advantage comes from pushing that loop entirely into compiled C/SIMD, which pure Python (even with `ctypes` buffers) cannot replicate.

---

## 7. Library Implementation (Comparison)

| From scratch (`TinyArray`) | Real NumPy |
|---|---|
| 1D only, manual `ctypes` buffer | N-dimensional, full stride/shape machinery |
| `__add__` loop runs in Python (slow) | element-wise ops compiled in C, SIMD-vectorized |
| No broadcasting | full broadcasting rule engine |
| No BLAS/LAPACK | matrix ops dispatch to hardware-optimized BLAS/LAPACK |
| Educational only | 10-1000x faster in practice; use always in real code |

---

## 8. Visual Explanations

**Memory layout: row-major (C order) vs. column-major (Fortran order):**
```
Array:            Row-major (C, NumPy default):      Column-major (Fortran):
[[1, 2, 3],        memory: 1 2 3 4 5 6                memory: 1 4 2 5 3 6
 [4, 5, 6]]        (row 0 fully, then row 1)           (col 0 fully, then col 1)
```

**Broadcasting a (4,) vector across a (1000, 4) matrix:**
```
Matrix (1000, 4)          Vector (4,)  -- virtually stretched, NO copy --
[ x00 x01 x02 x03 ]        [ m0  m1  m2  m3 ]
[ x10 x11 x12 x13 ]   -    [ m0  m1  m2  m3 ]   <- same row repeated via stride=0
[ ...              ]       [ ... 1000 times ]
```

---

## 9. Practical Examples

**Simple:** compute the mean and standard deviation of a claims array without any loop.
**Medium:** one-hot encode a categorical array of region codes using `np.eye()[codes]` (a classic vectorized trick).
**Real-world:** vectorized Monte Carlo simulation of exchange rate paths (directly relevant to your DZD forward-simulation work):
```python
n_paths, n_steps = 10_000, 252
rng = np.random.default_rng(7)
shocks = rng.normal(loc=0, scale=0.01, size=(n_paths, n_steps))     # (10000, 252) all at once
log_paths = np.cumsum(shocks, axis=1)                                # vectorized cumulative sum per path
final_rates = 100 * np.exp(log_paths[:, -1])                         # terminal simulated rate per path
print(final_rates.mean(), final_rates.std())
```
This entire 10,000-path, 252-step simulation runs as a handful of vectorized calls — no explicit Python loop over paths or time steps.

---

## 10. Real Industry Use Cases

- **PyTorch/TensorFlow**: tensor APIs are deliberately NumPy-shaped; NumPy interoperability (`torch.from_numpy`, zero-copy where possible) is a first-class feature.
- **Quant finance / actuarial modeling**: Monte Carlo simulation, portfolio risk aggregation, and stochastic differential equation solvers (your SDE coursework) are essentially all vectorized NumPy under the hood.
- **NVIDIA RAPIDS (`cuPy`)**: a near-drop-in GPU-accelerated reimplementation of the NumPy API — proof of how foundational NumPy's *interface* has become, independent of its original CPU implementation.
- **Every scikit-learn estimator**: expects and returns NumPy arrays as its core numerical contract.

---

## 11. Common Mistakes

- Writing explicit Python `for` loops over array elements instead of vectorizing — often 50-500x slower for large arrays.
- Confusing views and copies: mutating a slice unexpectedly mutates the original array; conversely, expecting a fancy-indexed result to share memory when it doesn't.
- Silent broadcasting bugs: adding a `(1000,)` vector to a `(1000, 1)` column when you meant elementwise `(1000,)` + `(1000,)` — produces a silently wrong `(1000, 1000)` broadcasted result instead of an error.
- Ignoring `dtype` — mixing `int` arrays in a division can silently floor-divide or overflow depending on dtype in some contexts; always be explicit about `float64` for numerical work needing precision.

---

## 12. Best Practices (2026)

- Use the modern `np.random.Generator` API (`np.random.default_rng`) — the legacy global `np.random.seed` API is considered legacy/deprecated style in new code.
- Prefer `np.linalg.solve` over `np.linalg.inv` for solving linear systems — faster and more numerically stable.
- Use `einsum` (`np.einsum`) for complex multi-array tensor contractions — often clearer AND faster than chained `reshape`/`matmul`, and its exact same notation reappears in PyTorch for attention mechanisms (Phase 5).
- Know when to "drop down" to NumPy vs. staying in pandas (Lesson 2) — heavy numerical/linear-algebra work belongs in NumPy; labeled, heterogeneous tabular manipulation belongs in pandas.

---

## 13. Exercises

**Easy:** Vectorize a Celsius→Fahrenheit conversion over a 10,000-element array and compare timing against a Python loop.
**Medium:** Implement min-max normalization and z-score normalization as vectorized functions, verifying against `sklearn.preprocessing` equivalents.
**Hard:** Implement a vectorized pairwise Euclidean distance matrix computation between two sets of points ($n \times d$ and $m \times d$) using broadcasting, without any explicit loop — this is the core operation behind k-NN (Phase 4) and vector search (Phase 7).
**Mathematical:** Derive why $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 a \cdot b$ lets you compute a full pairwise distance matrix using only matrix multiplication (`X @ X.T`) plus broadcasting, avoiding an explicit $O(n \times m \times d)$ triple loop's Python overhead.
**Coding:** Implement `TinyArray` (Section 6) support for 2D shape and broadcasting-lite (stretching a `(1,)` array across a longer array).

---

## 14. Mini Project

Build a **vectorized Monte Carlo mortality/lapse simulator**: given per-age mortality and lapse probability vectors, simulate 100,000 policyholder paths over 30 years entirely via vectorized NumPy operations (no explicit Python loop over policyholders), compute the expected present value of a life insurance payout using vectorized discounting, and benchmark your vectorized version against a naive nested-Python-loop version to empirically demonstrate the speedup — directly extending your earlier actuarial/EVT coursework into a reusable simulation tool.

---

## 15. Interview Preparation

- Explain what makes a NumPy array faster than a Python list for numerical work.
- What is broadcasting, and what are its exact compatibility rules?
- What is the difference between a view and a copy in NumPy, and how would you detect which one an operation returns?
- System design/coding: implement a vectorized pairwise cosine similarity matrix between two matrices of embeddings (directly relevant to Phase 7's vector search).

---

## 16. Summary

NumPy's entire value proposition is: represent numerical data as contiguous, homogeneously-typed memory, and express computation as whole-array operations (vectorization + broadcasting) so the actual looping happens in compiled C rather than the Python interpreter. Every later phase's numerical code — from-scratch neural networks (Phase 5), embeddings (Phase 6), simulation-heavy actuarial work — is, underneath any higher-level library, ultimately NumPy-shaped array manipulation.

---

## 17. References

- Harris, C.R. et al. — "Array programming with NumPy" (Nature, 2020) — the official NumPy paper
- Official NumPy documentation, especially the "Broadcasting" and "Array creation" guides
- VanderPlas, J. — *Python Data Science Handbook*, NumPy chapters
- 100 NumPy Exercises (community resource, excellent for drilling vectorization fluency)
