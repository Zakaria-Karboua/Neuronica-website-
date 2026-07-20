# Phase 3 · Lesson 5 — Optimization

> Prerequisite: Linear Algebra, Calculus, Probability, Statistics (Lessons 1–4)

---

## 1. Introduction

### What is optimization?
The mathematics of finding the input(s) that minimize (or maximize) a function — in ML, almost always minimizing a **loss function** over model parameters. Every training run of every model in this curriculum — from a closed-form linear regression to a billion-parameter transformer — is, at its core, an optimization procedure.

### Why does it exist?
"Learning" in machine learning is not a mystical process — it is the mechanical search for parameter values that make a model's predictions match observed data as closely as possible, formalized as minimizing a loss function. Optimization theory provides the guarantees (or lack thereof) about whether, and how fast, that search actually converges to a good solution.

### Historical background
Gradient descent traces to Cauchy (1847). Convex optimization theory matured through the 20th century (linear programming — Kantorovich, Dantzig; interior-point methods — Karmarkar, 1984). Stochastic Gradient Descent (Robbins-Monro, 1951) and its modern deep-learning-era variants (Momentum, Adam — Kingma & Ba, 2014) are what make training on datasets far too large to fit in memory, using far more parameters than classical optimization theory originally anticipated, computationally tractable.

### Real-world motivation
Every hyperparameter you'll tune in Phase 4-6 (learning rate, batch size, optimizer choice) is a lever on this lesson's machinery. Understanding *why* a learning rate that's too high diverges and one that's too low converges glacially isn't optional trivia — it's the difference between a training run that works and one that silently wastes a day of GPU time.

---

## 2. Theory

### Convex vs. non-convex optimization
A function $f$ is **convex** if, for any two points, the line segment connecting them lies *above* the function's graph:
$$
f(\lambda x + (1-\lambda) y) \le \lambda f(x) + (1-\lambda) f(y), \quad \forall \lambda \in [0,1]
$$
Convex functions have a crucial guarantee: **any local minimum is the global minimum**. Linear/logistic regression's loss functions are convex (guaranteed convergence to the global optimum); neural network loss surfaces (Phase 5) are decidedly **non-convex** (many local minima, saddle points — Lesson 2's Hessian discussion) — a fundamentally different, harder optimization regime with far weaker theoretical guarantees, yet empirically still trainable.

### Gradient Descent and its variants
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$
where $\eta$ is the **learning rate**. Variants:
- **Batch GD**: use the full dataset's gradient each step — accurate but slow/memory-heavy for large datasets.
- **Stochastic GD (SGD)**: use a single random sample's gradient — noisy but cheap, and the noise itself can help escape shallow local minima/saddle points.
- **Mini-batch GD**: use a small batch (the standard in deep learning) — a practical middle ground exploiting vectorized hardware (Phase 2's NumPy vectorization ideas, now on GPUs).

### Momentum and adaptive learning rates
- **Momentum**: accumulates a running average of past gradients, smoothing out oscillations in narrow, curved loss valleys (a common non-convex loss landscape feature).
- **Adam** (Adaptive Moment Estimation): maintains per-parameter adaptive learning rates based on estimates of both the first moment (mean, like momentum) and second moment (uncentered variance) of gradients — the dominant default optimizer for training neural networks in 2026, precisely because it requires far less learning-rate tuning than plain SGD.

### Constrained optimization and Lagrange multipliers
For minimizing $f(x)$ subject to $g(x) = 0$:
$$
\mathcal{L}(x, \lambda) = f(x) - \lambda g(x), \qquad \nabla_x \mathcal{L} = 0, \quad \nabla_\lambda \mathcal{L} = 0
$$
At the optimum, the gradients of $f$ and $g$ are parallel — the intuition being that if they weren't parallel, you could slide along the constraint surface to further improve $f$. This is the foundation of Support Vector Machines' margin-maximization formulation (Phase 4) and appears throughout constrained ML formulations.

---

## 3. Mathematical Foundations

### Convergence rate of gradient descent (convex, smooth case)
For an $L$-smooth convex function (gradient doesn't change too fast, formally $\|\nabla f(x) - \nabla f(y)\| \le L\|x-y\|$), gradient descent with $\eta = 1/L$ achieves:
$$
f(\theta_t) - f(\theta^*) \le \frac{L\|\theta_0 - \theta^*\|^2}{2t}
$$
i.e., $O(1/t)$ convergence — the error shrinks *sublinearly*. For **strongly convex** functions (Hessian eigenvalues bounded below by $\mu > 0$, a stronger curvature guarantee), convergence improves to *linear* (geometric): $O(\rho^t)$ for some $\rho < 1$ depending on the condition number $L/\mu$ — this ratio is *why* ill-conditioned problems (Lesson 1's eigenvalue-ratio discussion) converge painfully slowly under plain gradient descent, motivating momentum/adaptive methods and preconditioning.

### Adam's update rule, derived
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \qquad \text{(bias correction, crucial early in training)}
$$
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$
The $\sqrt{\hat{v}_t}$ term in the denominator gives each parameter an *individually scaled* effective learning rate: parameters with historically large/noisy gradients get dampened, parameters with small/consistent gradients get relatively boosted — directly addressing the "different parameters need different learning rates" problem that plagues plain SGD on ill-conditioned loss surfaces.

### Newton's method (second-order optimization)
$$
\theta_{t+1} = \theta_t - H^{-1}\nabla f(\theta_t)
$$
Using the Hessian $H$ (Lesson 2) directly accounts for curvature, achieving **quadratic** convergence near the optimum (far faster than gradient descent's linear/sublinear rates) — but computing/inverting an $n \times n$ Hessian is $O(n^3)$, completely intractable for models with millions/billions of parameters, which is *exactly why* first-order methods (gradient descent variants) dominate deep learning despite their theoretically slower convergence rate — the per-step cost tradeoff decisively favors them at scale.

---

## 4. Algorithm — Mini-Batch SGD with Momentum (fully specified)

```
INITIALIZE: theta (parameters), velocity v = 0, learning rate eta, momentum coefficient beta (e.g. 0.9)
FOR each epoch:
    SHUFFLE training data
    FOR each mini-batch B:
        compute gradient: g = (1/|B|) * sum_{i in B} grad_theta( loss(theta, x_i, y_i) )
        v = beta * v + (1 - beta) * g          # exponential moving average of gradients
        theta = theta - eta * v                 # update using the SMOOTHED gradient, not raw g
    (optionally: evaluate on validation set, check for early stopping -- Phase 4)
```
**Why momentum helps, intuitively:** in a long, narrow valley (a common non-convex loss landscape shape near saddle regions), raw gradients oscillate back and forth across the narrow direction while making slow progress along the valley's length; momentum's exponential averaging cancels out the oscillating components (which flip sign each step) while reinforcing the consistent-direction component (which doesn't) — net effect: faster progress along the valley, dampened oscillation across it.

---

## 5. Python Implementation

```python
"""optimization_core.py — gradient descent variants implemented explicitly"""
import numpy as np


def gradient_descent(grad_fn, theta0: np.ndarray, lr: float = 0.1, n_iters: int = 200) -> np.ndarray:
    theta = theta0.copy()
    history = [theta.copy()]
    for _ in range(n_iters):
        theta = theta - lr * grad_fn(theta)
        history.append(theta.copy())
    return np.array(history)


def sgd_with_momentum(grad_fn, theta0: np.ndarray, lr: float = 0.1, beta: float = 0.9, n_iters: int = 200):
    theta = theta0.copy()
    velocity = np.zeros_like(theta)
    history = [theta.copy()]
    for _ in range(n_iters):
        g = grad_fn(theta)
        velocity = beta * velocity + (1 - beta) * g
        theta = theta - lr * velocity
        history.append(theta.copy())
    return np.array(history)


def adam(grad_fn, theta0: np.ndarray, lr: float = 0.1, beta1: float = 0.9, beta2: float = 0.999,
          eps: float = 1e-8, n_iters: int = 200):
    theta = theta0.copy()
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    history = [theta.copy()]
    for t in range(1, n_iters + 1):
        g = grad_fn(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(theta.copy())
    return np.array(history)


# Test on an ILL-CONDITIONED quadratic (elongated bowl -- exposes the difference between methods)
A = np.array([[50.0, 0.0], [0.0, 1.0]])   # condition number 50 -- one axis MUCH steeper than the other
def grad(theta): return A @ theta

theta0 = np.array([1.0, 1.0])
gd_path = gradient_descent(grad, theta0, lr=0.03)
mom_path = sgd_with_momentum(grad, theta0, lr=0.03)
adam_path = adam(grad, theta0, lr=0.1)

print("Plain GD final:", gd_path[-1])
print("Momentum final:", mom_path[-1])
print("Adam final:", adam_path[-1])
print("GD iterations to near-convergence:", np.argmax(np.linalg.norm(gd_path, axis=1) < 1e-3))
print("Adam iterations to near-convergence:", np.argmax(np.linalg.norm(adam_path, axis=1) < 1e-3))
```

**Expected finding:** on this ill-conditioned quadratic, plain gradient descent converges much more slowly (or oscillates/diverges if `lr` is too large for the steep axis) than momentum or Adam — a direct, hands-on demonstration of Section 3's condition-number-dependent convergence theory.

---

## 6. Build From Scratch

This lesson's Section 5 already *is* the from-scratch implementation (gradient descent, momentum, and Adam are genuinely simple enough to implement directly and are shown in full above, not simplified toy versions) — the natural "build from scratch" exercise here is instead **line search**, a technique real optimizers use to automatically choose a good step size rather than a fixed learning rate:

```python
def backtracking_line_search(f, grad_fn, theta: np.ndarray, direction: np.ndarray,
                               alpha: float = 1.0, rho: float = 0.5, c: float = 1e-4) -> float:
    """Armijo backtracking line search: shrink step size until a sufficient-decrease condition holds."""
    fx = f(theta)
    grad_dot_dir = grad_fn(theta) @ direction
    while f(theta + alpha * direction) > fx + c * alpha * grad_dot_dir:
        alpha *= rho          # shrink the step size and try again
    return alpha
```
This automatically finds a step size satisfying the Armijo condition (sufficient decrease in $f$) at every iteration, rather than requiring a hand-tuned fixed learning rate — the classical alternative to the fixed/scheduled learning rates used almost universally in deep learning, where line search's extra function evaluations per step are usually too expensive at scale.

---

## 7. Library/Tool Comparison

| From scratch | PyTorch `torch.optim` |
|---|---|
| `gradient_descent`/`sgd_with_momentum`/`adam` (Section 5) | `torch.optim.SGD(momentum=...)`, `torch.optim.Adam(...)` — identical formulas, GPU-accelerated, integrated with autograd |
| `backtracking_line_search` | `torch.optim.LBFGS` (uses line search internally); rarely used in deep learning due to cost, common in classical optimization |
| Manual convergence checking | Learning rate schedulers (`torch.optim.lr_scheduler`) — automate reducing $\eta$ over training, a standard practice in Phase 5 onward |

---

## 8. Visual Explanations

**Gradient descent on an ill-conditioned (elongated) loss surface — oscillation vs. smooth momentum path:**
```
Plain GD (zigzag, slow):        With Momentum (smoothed, faster):
    \  /\  /\  /                        \
     \/  \/  \/                          \___
  (steep axis oscillates,            (oscillations cancel,
   shallow axis crawls)               net progress along the valley)
```

**Convergence rate comparison (schematic, error vs. iteration):**
```
Error
  │  \
  │   \_  <- sublinear (O(1/t), plain GD, convex non-strongly-convex)
  │     \___
  │         \
  │          \___  <- linear/geometric (strongly convex, well-conditioned, or momentum-assisted)
  │              \______
  │                     \___________  <- quadratic (Newton's method, near optimum)
  └──────────────────────────────────  iterations
```

---

## 9. Practical Examples

**Simple:** minimize $f(x) = (x-3)^2$ via gradient descent, printing the trajectory.
**Medium:** compare plain GD, momentum, and Adam on a 2D Rosenbrock function (a classic, deliberately ill-conditioned, non-convex optimization benchmark) and visualize each method's path.
**Real-world:** implement gradient descent for logistic regression from scratch (deriving the gradient of the cross-entropy loss, Phase 3 Lesson 3's MLE connection made concrete) and compare convergence speed/behavior of plain SGD vs. Adam on your actuarial claims classification data.

---

## 10. Real Industry Use Cases

- **Every neural network training run** (PyTorch/TensorFlow/JAX): Adam (or its variants, AdamW with weight decay) is the default optimizer for the overwhelming majority of deep learning training in 2026.
- **Large-scale distributed training** (foundation model pretraining, Phase 6): involves sophisticated learning-rate schedules (warmup + cosine decay) and often specialized optimizers (LAMB, Lion) designed specifically for the extreme-batch-size, extreme-parameter-count regime.
- **Classical operations research / actuarial optimization**: portfolio optimization, reserve calculation under constraints, and pricing optimization problems are often *convex* (linear/quadratic programming), where global-optimum guarantees (Section 2) genuinely apply, unlike neural network training.
- **AutoML/hyperparameter tuning systems** (Optuna, Ray Tune): themselves optimization problems, often solved via Bayesian optimization — a more sophisticated technique built on top of this lesson's foundations.

---

## 11. Common Mistakes

- Using a single fixed learning rate across an entire long training run without any schedule/decay — often leaves substantial performance on the table compared to warmup + decay schedules.
- Choosing a learning rate too large for an ill-conditioned problem, causing divergence (loss exploding to `NaN`) rather than slow convergence — an extremely common early debugging experience in Phase 5.
- Assuming a non-convex training loss reaching a plateau means "converged to the global optimum" — it may simply be near a saddle point or local minimum; empirically, this matters less for large neural networks than classical theory once suggested, but the distinction is still real.
- Forgetting Adam's bias correction terms ($\hat m_t$, $\hat v_t$) when re-implementing it from scratch — omitting them causes underestimated updates specifically in early training steps.

---

## 12. Best Practices (2026)

- Default to AdamW (Adam with decoupled weight decay, now the standard variant over vanilla Adam) for most neural network training unless you have a specific reason to use plain SGD+momentum (sometimes still preferred for certain vision architectures' generalization properties).
- Use a learning-rate warmup followed by cosine or linear decay for any non-trivial training run — a near-universal 2026 default, especially for transformer training (Phase 5-6).
- Monitor gradient norms during training, not just the loss — exploding or vanishing gradient norms are often visible earlier and more diagnostically than the loss curve alone.
- For genuinely convex problems (classical statistical/actuarial optimization), prefer specialized convex solvers (interior-point methods, `cvxpy`) over generic gradient descent — they exploit convexity's guarantees for far faster, more reliable convergence.

---

## 13. Exercises

**Easy:** Implement gradient descent for $f(x) = x^4 - 3x^2$ and observe it can converge to different local minima depending on initialization (non-convexity in action).
**Medium:** Implement and compare plain SGD, momentum, and Adam on a synthetic ill-conditioned quadratic (Section 5), plotting convergence curves for each.
**Hard:** Implement the backtracking line search (Section 6) combined with plain gradient descent, and empirically show it avoids divergence even with a poorly-chosen initial step size, unlike fixed-learning-rate GD.
**Mathematical:** Derive the convergence rate of gradient descent on a strongly convex quadratic $f(\theta) = \frac{1}{2}\theta^T A \theta$ as a function of $A$'s condition number, and verify it empirically using Section 5's ill-conditioned example.
**Coding:** Implement RMSProp (Adam's predecessor, using only the second-moment adaptive scaling without momentum) from scratch and compare it against your Adam implementation.

---

## 14. Mini Project

Build a **from-scratch logistic regression trainer with a full optimizer comparison**: derive and implement the cross-entropy loss gradient, train the same model using plain SGD, SGD+momentum, and Adam on a real (or your actuarial) binary classification dataset, plot training loss curves for all three, and empirically characterize each optimizer's convergence speed and sensitivity to learning-rate choice — directly setting up the foundation for Phase 4's more advanced supervised learning models and Phase 5's neural network training.

---

## 15. Interview Preparation

- Explain the difference between convex and non-convex optimization, and why neural network training lacks convex guarantees.
- Derive the Adam update rule and explain what problem each of its components (momentum, adaptive scaling, bias correction) solves.
- Why does an ill-conditioned loss surface slow down gradient descent, and how does momentum help?
- Coding: implement mini-batch stochastic gradient descent from scratch for a linear model.

---

## 16. Summary

Optimization is the mechanical engine of "learning" in machine learning: gradient descent and its momentum/adaptive variants (culminating in Adam, the dominant 2026 default) navigate loss landscapes toward good parameter values, with convergence speed governed by convexity, curvature/conditioning (directly connecting back to Lesson 1's eigenvalues and Lesson 2's Hessian), and the specific optimizer's ability to smooth out ill-conditioned, oscillation-prone directions. Every training run in Phase 5 onward is a direct, larger-scale application of exactly the algorithms implemented from scratch in this lesson.

---

## 17. References

- Boyd, S. & Vandenberghe, L. — *Convex Optimization* (free online, the definitive convex optimization reference)
- Kingma, D. & Ba, J. — "Adam: A Method for Stochastic Optimization" (2014, the original Adam paper)
- Nocedal, J. & Wright, S. — *Numerical Optimization* (comprehensive classical + modern optimization theory)
- Ruder, S. — "An Overview of Gradient Descent Optimization Algorithms" (widely-read practical survey)
