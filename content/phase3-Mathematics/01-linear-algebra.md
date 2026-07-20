# Phase 3 · Lesson 1 — Linear Algebra

> Prerequisite: Phase 2 (especially NumPy). This lesson formalizes the math NumPy operations implement.

---

## 1. Introduction

### What is linear algebra?
The study of vector spaces and linear maps between them — vectors, matrices, linear transformations, eigenvalues/eigenvectors, and matrix decompositions. It is, without exaggeration, the mathematical language every ML model is written in: a neural network layer is a matrix multiplication plus a non-linearity; PCA is an eigendecomposition; word/token embeddings are vectors in a learned space; attention (Phase 5-6) is a sequence of matrix multiplications and softmax-weighted sums.

### Why does it exist?
Linear algebra formalizes the intuitive idea of "combining and scaling" quantities in a way that generalizes cleanly from 2D/3D geometric intuition to arbitrarily high dimensions — exactly what's needed to reason about datasets with hundreds or thousands of features, or embedding spaces with thousands of dimensions.

### Historical background
Matrix theory formalized in the 19th century (Cayley, Sylvester); the abstract vector-space axiomatization came later (Peano, 1888; fully modern treatment mid-20th century). Its centrality to computing exploded with the development of efficient numerical linear algebra libraries (LINPACK/LAPACK, 1970s-80s) — the direct ancestors of the BLAS/LAPACK routines NumPy calls today (Phase 2 Lesson 1).

### Real-world motivation
Every model you'll build from Phase 4 onward — from a simple linear regression closed-form solution to a full transformer's attention mechanism — is linear algebra performing the actual computation, with everything else (activation functions, loss functions) layered on top.

---

## 2. Theory

### Vectors and vector spaces
A vector space $V$ over $\mathbb{R}$ is a set closed under vector addition and scalar multiplication, satisfying axioms (associativity, distributivity, existence of a zero vector, etc.). In ML practice, $V = \mathbb{R}^n$ almost always — an $n$-dimensional feature vector, an embedding, a gradient.

### Linear independence, basis, dimension
A set of vectors is **linearly independent** if no vector in the set can be written as a combination of the others. A **basis** is a linearly independent set that spans the whole space; the **dimension** is the number of vectors in any basis (invariant regardless of which basis you pick).

### Matrices as linear transformations
A matrix $A \in \mathbb{R}^{m \times n}$ represents a linear map $T: \mathbb{R}^n \to \mathbb{R}^m$, $T(x) = Ax$. "Linear" means $T(\alpha x + \beta y) = \alpha T(x) + \beta T(y)$ — this single property is what makes matrix operations composable and analytically tractable (versus arbitrary non-linear functions).

### Eigenvalues and eigenvectors
For a square matrix $A$, a nonzero vector $v$ is an **eigenvector** with **eigenvalue** $\lambda$ if:
$$
Av = \lambda v
$$
i.e., $A$ acts on $v$ purely by *scaling*, not rotating/shearing it. Eigenvectors reveal a matrix's "natural axes" — directly the mathematical basis of PCA (Phase 4) and central to understanding neural network weight-matrix conditioning (Phase 5).

### Matrix decompositions (the practical toolkit)
| Decomposition | Form | Used for |
|---|---|---|
| Eigendecomposition | $A = Q \Lambda Q^{-1}$ (symmetric: $Q$ orthogonal) | PCA, understanding quadratic forms |
| Singular Value Decomposition (SVD) | $A = U \Sigma V^T$ (any matrix, not just square) | dimensionality reduction, recommender systems, pseudo-inverse |
| LU decomposition | $A = LU$ | efficient linear system solving (`np.linalg.solve`) |
| Cholesky | $A = LL^T$ (symmetric positive-definite) | efficient sampling from multivariate Gaussians, faster than general LU when applicable |

---

## 3. Mathematical Foundations

### The Normal Equation (closed-form linear regression, derived)
Given design matrix $X \in \mathbb{R}^{n \times p}$ and target $y \in \mathbb{R}^n$, minimize squared error $\|Xβ - y\|^2$. Taking the gradient and setting to zero:
$$
\nabla_\beta \|X\beta - y\|^2 = 2X^T(X\beta - y) = 0 \implies X^TX\beta = X^Ty \implies \beta = (X^TX)^{-1}X^Ty
$$
This is the "Normal Equation" you already used numerically in Phase 2's NumPy lesson (`np.linalg.solve(X.T@X, X.T@y)`) — now you see exactly where it comes from.

### SVD and PCA (the connection made explicit)
For a mean-centered data matrix $X$, the covariance matrix is $\Sigma = \frac{1}{n-1}X^TX$. The eigenvectors of $\Sigma$ (equivalently, the right singular vectors $V$ from $X = U\Sigma_{svd}V^T$) are the **principal components** — directions of maximum variance, ordered by their eigenvalues/singular values. This is the exact mathematical machinery behind Phase 4's PCA lesson.

### Eigendecomposition and quadratic forms
A quadratic form $x^TAx$ (appearing constantly in loss functions, e.g., ridge regression penalties, Gaussian log-likelihoods) can be analyzed via $A$'s eigenvalues: if all eigenvalues $\lambda_i > 0$, $A$ is **positive definite** and $x^TAx > 0$ for all $x \ne 0$ — meaning the corresponding quadratic loss surface is a convex bowl with a unique minimum, a property optimization algorithms (Lesson 5, this phase) depend on heavily.

### Norms
$$
\|x\|_1 = \sum_i |x_i| \quad \text{(L1, sparsity-inducing)}, \qquad \|x\|_2 = \sqrt{\sum_i x_i^2} \quad \text{(L2, Euclidean)}
$$
L1 regularization (Lasso, Phase 4) drives coefficients exactly to zero because of the L1 norm's non-differentiable "corner" at zero; L2 (Ridge) shrinks coefficients smoothly toward zero without eliminating them — a direct geometric consequence of each norm's unit-ball shape (a diamond vs. a circle in 2D).

---

## 4. Algorithm — Computing Eigenvalues (Power Iteration, conceptual)

```
GIVEN a symmetric matrix A, find the dominant eigenvector:
1. Start with a random vector v_0
2. REPEAT:
     v_{k+1} = A @ v_k
     v_{k+1} = v_{k+1} / ||v_{k+1}||     # normalize to prevent overflow/underflow
   UNTIL convergence (v_{k+1} ≈ v_k up to sign)
3. The dominant eigenvalue is: lambda = v^T A v  (the Rayleigh quotient)
```
**Why it works:** writing $v_0$ in the eigenbasis, $v_0 = \sum_i c_i e_i$, repeated multiplication by $A$ scales each component by its eigenvalue: $A^k v_0 = \sum_i c_i \lambda_i^k e_i$. As $k \to \infty$, the term with the *largest* $|\lambda_i|$ dominates the sum — the vector converges to that dominant eigenvector. Convergence rate is $O((\lambda_2/\lambda_1)^k)$ — the more separated the top two eigenvalues, the faster convergence.

---

## 5. Python Implementation

```python
"""linear_algebra_core.py"""
import numpy as np

X = np.random.default_rng(0).normal(size=(1000, 5))
X_centered = X - X.mean(axis=0)

# --- Eigendecomposition (symmetric case: covariance matrix) ---
cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)     # 'eigh' exploits symmetry, more stable than 'eig'
order = np.argsort(eigenvalues)[::-1]                 # sort descending — largest variance first
eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

# --- SVD (general case, directly gives principal components too) ---
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
explained_variance_ratio = (S ** 2) / np.sum(S ** 2)

# --- Solving a linear system the RIGHT way (never explicitly invert) ---
y = np.random.default_rng(1).normal(size=1000)
beta = np.linalg.solve(X_centered.T @ X_centered + 0.1 * np.eye(5), X_centered.T @ y)  # ridge-regularized

print("Eigenvalues:", eigenvalues.round(3))
print("Explained variance ratio (SVD):", explained_variance_ratio.round(3))
print("Ridge coefficients:", beta.round(3))
```

**Notes:** `np.linalg.eigh` (not `np.linalg.eig`) is used specifically because the covariance matrix is symmetric — `eigh` guarantees real eigenvalues and is numerically more stable/faster, exploiting that guarantee. Adding `0.1 * np.eye(5)` before solving is exactly Ridge regression's regularization term, making $(X^TX + \lambda I)$ always invertible even when $X^TX$ is singular/ill-conditioned.

---

## 6. Build From Scratch

**Power iteration (Section 4) implemented directly:**
```python
import numpy as np

def power_iteration(A: np.ndarray, n_iters: int = 100, tol: float = 1e-10) -> tuple[float, np.ndarray]:
    n = A.shape[0]
    v = np.random.default_rng(0).normal(size=n)
    v = v / np.linalg.norm(v)
    eigenvalue = 0.0
    for _ in range(n_iters):
        v_new = A @ v
        v_new_norm = np.linalg.norm(v_new)
        v_new = v_new / v_new_norm
        new_eigenvalue = v_new @ A @ v_new    # Rayleigh quotient
        if abs(new_eigenvalue - eigenvalue) < tol:
            break
        v, eigenvalue = v_new, new_eigenvalue
    return eigenvalue, v

A = np.array([[4.0, 1.0], [1.0, 3.0]])
val, vec = power_iteration(A)
print(val, vec)
# Compare against np.linalg.eigh(A) -- should match the LARGEST eigenvalue/eigenvector
```

**Gram-Schmidt orthogonalization from scratch (mirrors what QR decomposition does):**
```python
def gram_schmidt(vectors: list[np.ndarray]) -> list[np.ndarray]:
    basis = []
    for v in vectors:
        w = v.copy()
        for u in basis:
            w = w - (np.dot(v, u) / np.dot(u, u)) * u   # subtract projection onto each existing basis vector
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))
    return basis
```

---

## 7. Library Implementation (Comparison)

| From scratch | Production (NumPy/SciPy/LAPACK) |
|---|---|
| `power_iteration` | `np.linalg.eigh`/`eig` — full spectrum via QR algorithm, far more efficient for all eigenvalues, not just the dominant one |
| `gram_schmidt` | `np.linalg.qr` — numerically stable Householder-reflection-based QR, avoids Gram-Schmidt's known numerical instability with nearly-parallel vectors |
| Manual normal equation | `np.linalg.lstsq` — uses SVD internally, robust even when $X^TX$ is singular/ill-conditioned (unlike direct inversion) |

---

## 8. Visual Explanations

**Eigenvectors as "natural axes" of a linear transformation:**
```
Before (unit circle):        After applying A (ellipse):
      ↑                              ↑
   ●──┼──●                       ●───┼──────●   <- stretched along eigenvector 1 (large eigenvalue)
      │                              │
                                  <- compressed along eigenvector 2 (small eigenvalue)
Eigenvectors = the axes that only get SCALED (not rotated) by A.
```

**SVD as three sequential transformations:**
```
X = U  Σ  V^T
    │  │   │
    │  │   └─ ROTATE input space (orthonormal basis change)
    │  └───── SCALE along each new axis (singular values, descending)
    └──────── ROTATE into output space (orthonormal basis change)
```

---

## 9. Practical Examples

**Simple:** compute the eigenvalues/eigenvectors of a 2x2 covariance matrix by hand and verify with `np.linalg.eigh`.
**Medium:** perform PCA via SVD on a 5-feature actuarial dataset and report how many components are needed to explain 95% of variance.
**Real-world:** use SVD-based dimensionality reduction on a high-dimensional claims feature matrix (many correlated regional/occupational indicator features) before feeding into a model — directly useful given your actuarial portfolio work often has many correlated categorical-derived features.

---

## 10. Real Industry Use Cases

- **PCA/dimensionality reduction** (Netflix's original recommendation algorithms used SVD-based matrix factorization extensively).
- **PyTorch/TensorFlow**: every layer's forward pass is matrix multiplication; backpropagation (Phase 5) is the chain rule applied through a sequence of these linear maps.
- **Embeddings** (Phase 6): word/token embeddings are literally vectors in $\mathbb{R}^d$; similarity search (Phase 7) is inner-product/cosine-similarity computation — pure linear algebra.
- **Actuarial/risk modeling**: covariance matrices of correlated risk factors (interest rate, mortality, lapse) require eigendecomposition-based techniques for risk aggregation (e.g., Solvency II's correlation-matrix-based capital aggregation).

---

## 11. Common Mistakes

- Explicitly computing `np.linalg.inv(A) @ b` instead of `np.linalg.solve(A, b)` — slower and less numerically stable.
- Using `np.linalg.eig` (general, complex-valued-capable) on a known-symmetric matrix instead of `np.linalg.eigh` — slower and can return spurious tiny imaginary components due to floating-point error.
- Forgetting to mean-center data before PCA/covariance computation — produces a fundamentally different (and usually meaningless for "variance direction" purposes) decomposition.
- Ignoring matrix conditioning (ratio of largest to smallest eigenvalue) — an ill-conditioned $X^TX$ makes the Normal Equation numerically unstable, motivating ridge regularization or SVD-based least squares instead.

---

## 12. Best Practices (2026)

- Always use `np.linalg.solve` (or `lstsq`/`pinv` for non-square/singular cases) over explicit matrix inversion.
- Use `eigh`/`svd` (exploiting symmetry/structure) rather than the fully general `eig` whenever applicable.
- Check condition numbers (`np.linalg.cond`) on any matrix you're about to invert/solve in a production pipeline — a very large condition number is an early warning of numerical instability.
- Understand that GPU-accelerated linear algebra (cuBLAS, used under the hood by PyTorch) follows the exact same mathematical structure — the math here transfers directly to understanding neural network computation in Phase 5.

---

## 13. Exercises

**Easy:** By hand, compute the eigenvalues of $\begin{pmatrix}2&0\\0&3\end{pmatrix}$ and $\begin{pmatrix}2&1\\1&2\end{pmatrix}$.
**Medium:** Implement PCA from scratch using `np.linalg.eigh` on the covariance matrix and verify your result matches `sklearn.decomposition.PCA`.
**Hard:** Implement the power iteration method (Section 6) and empirically verify its convergence rate matches the theoretical $O((\lambda_2/\lambda_1)^k)$ prediction by varying the eigenvalue gap of a test matrix.
**Mathematical:** Prove that for a symmetric matrix, eigenvectors corresponding to distinct eigenvalues are orthogonal.
**Coding:** Implement Gram-Schmidt (Section 6) and compare its numerical stability against `np.linalg.qr` on a set of nearly-parallel vectors.

---

## 14. Mini Project

Build a **from-scratch PCA + risk-factor decomposition tool** for a simulated multi-factor actuarial dataset (interest rate, mortality shock, lapse shock, each correlated): compute the covariance matrix, perform eigendecomposition, identify the principal risk factors (directions of maximum joint variance), and compare your from-scratch implementation's output against `sklearn.decomposition.PCA` and `np.linalg.svd` for consistency — directly relevant to Solvency-II-style risk aggregation methodology.

---

## 15. Interview Preparation

- Explain eigenvalues/eigenvectors intuitively and their role in PCA.
- Derive the Normal Equation for linear regression from the squared-error objective.
- What's the difference between eigendecomposition and SVD, and why does SVD work on non-square matrices?
- Coding: implement power iteration to find a matrix's dominant eigenvalue.

---

## 16. Summary

Linear algebra provides the exact computational vocabulary of ML: matrices as linear transformations, eigendecomposition/SVD as tools for finding a dataset's "natural axes" (PCA), and the Normal Equation as the closed-form solution underlying linear regression. Every neural network forward pass, every embedding similarity computation, and every dimensionality-reduction technique in later phases is this same mathematics, scaled up and composed.

---

## 17. References

- Strang, G. — *Introduction to Linear Algebra*
- Trefethen & Bau — *Numerical Linear Algebra* (for the numerical-stability perspective)
- 3Blue1Brown — "Essence of Linear Algebra" video series (exceptional geometric intuition)
- Golub & Van Loan — *Matrix Computations* (the definitive reference for decompositions)
