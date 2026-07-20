# Phase 4 · Lesson 1 — Supervised Learning

> Prerequisite: Phase 3 (all lessons — this is where the math becomes models)

---

## 1. Introduction

### What is supervised learning?
Learning a function $f: X \to Y$ from labeled examples $(x_i, y_i)$, such that $f$ generalizes to predict $y$ for new, unseen $x$. This is the workhorse paradigm behind the overwhelming majority of production tabular ML systems — your mortality model, credit scoring, fraud detection, demand forecasting.

### Why does it exist?
Supervised learning formalizes "learning from examples with known answers" — the natural framing for problems where historical labeled outcomes exist (did this claim get approved, did this patient survive, what was the actual price). It's distinguished from unsupervised learning (Lesson 2, no labels) and reinforcement learning (Lesson 7, learning from reward signals rather than direct labels).

### Historical background
Linear regression predates "machine learning" by over a century (Legendre/Gauss, ~1800s, for astronomical calculations). Logistic regression (Cox, 1958) extended this to classification. Decision trees (1960s-80s, CART/ID3/C4.5) and their modern gradient-boosted descendants (XGBoost, 2016; LightGBM, CatBoost) represent the other major supervised learning lineage, dominant on tabular data even in the 2026 deep-learning era.

### Real-world motivation
Your XGBoost mortality model is a supervised learning system. This lesson formalizes the theory (bias-variance tradeoff, regularization, the exact mechanics of the algorithms you're already using) underneath work you've already done practically.

---

## 2. Theory

### Regression vs. classification
- **Regression**: $Y$ is continuous (e.g., predicted claim severity).
- **Classification**: $Y$ is categorical (e.g., claim approved/denied, mortality within N years).

### The bias-variance tradeoff (the central theoretical concept of supervised learning)
Expected test error decomposes as:
$$
E[(y - \hat{f}(x))^2] = \underbrace{(\text{Bias}[\hat f(x)])^2}_{\text{systematic error}} + \underbrace{\text{Var}[\hat f(x)]}_{\text{sensitivity to training data}} + \underbrace{\sigma^2}_{\text{irreducible noise}}
$$
- **High bias** (underfitting): model too simple to capture true structure (e.g., linear model on genuinely non-linear data).
- **High variance** (overfitting): model too flexible, fits training-set noise, generalizes poorly.
- Model complexity trades one against the other — this single tradeoff explains regularization, ensemble methods (Lesson 5), and cross-validation's entire purpose.

### Core algorithm families
| Family | Examples | Character |
|---|---|---|
| Linear models | Linear/Logistic Regression, Ridge, Lasso | high bias, low variance, interpretable, fast |
| Instance-based | k-Nearest Neighbors | no training phase, prediction cost scales with data size |
| Tree-based | Decision Trees, Random Forest, XGBoost/LightGBM | handle non-linearity/interactions natively, dominant on tabular data |
| Kernel methods | Support Vector Machines | effective in high dimensions, kernel trick enables non-linear boundaries |

### Regularization
Adding a penalty term to the loss to control variance/complexity:
$$
L_{\text{ridge}}(\beta) = \|y - X\beta\|^2 + \lambda\|\beta\|_2^2, \qquad L_{\text{lasso}}(\beta) = \|y - X\beta\|^2 + \lambda\|\beta\|_1
$$
(Directly reusing Phase 3 Lesson 1's L1/L2 norm geometry: Lasso's diamond-shaped constraint region has corners on the axes, driving some coefficients exactly to zero — automatic feature selection; Ridge's circular constraint shrinks smoothly without eliminating any.)

---

## 3. Mathematical Foundations

### Logistic regression, derived via MLE (connecting Phase 3 Lessons 3 and 6)
Model $P(y=1|x) = \sigma(w^Tx + b) = \frac{1}{1+e^{-(w^Tx+b)}}$. The likelihood for $n$ i.i.d. examples:
$$
L(w,b) = \prod_i \sigma(w^Tx_i+b)^{y_i}(1-\sigma(w^Tx_i+b))^{1-y_i}
$$
Negative log-likelihood (= cross-entropy loss, Phase 3 Lesson 6):
$$
-\log L = -\sum_i \big[y_i \log \hat y_i + (1-y_i)\log(1-\hat y_i)\big]
$$
Gradient with respect to $w$ (derived via the chain rule, Phase 3 Lesson 2), using the convenient fact that $\sigma'(z) = \sigma(z)(1-\sigma(z))$:
$$
\nabla_w(-\log L) = \sum_i (\hat y_i - y_i)x_i
$$
— a remarkably clean result (error times input, summed) that falls directly out of the sigmoid+cross-entropy combination, exactly why this pairing is so ubiquitous.

### Decision tree splitting criteria
- **Gini impurity**: $G = 1 - \sum_k p_k^2$ (probability two randomly drawn samples from a node have different classes).
- **Entropy/Information gain** (Phase 3 Lesson 6 directly reused): $IG = H(\text{parent}) - \sum_{\text{children}} \frac{n_{\text{child}}}{n_{\text{parent}}}H(\text{child})$.
Both measure node "purity"; trees greedily choose the split maximizing purity gain at each node — a greedy algorithm (Phase 1 Lesson 4), not globally optimal, but computationally tractable.

### Support Vector Machines and the kernel trick
SVMs maximize the margin between classes, formulated as a constrained (Phase 3 Lesson 5) optimization:
$$
\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^Tx_i+b) \ge 1 \;\forall i
$$
The **kernel trick** replaces $x_i^Tx_j$ with a kernel function $K(x_i,x_j)$ computing the inner product *as if* the data were mapped into a much higher-dimensional (even infinite-dimensional, for the RBF kernel) space — without ever explicitly computing that mapping, exploiting the fact that the SVM optimization only ever needs inner products, not the transformed vectors themselves.

---

## 4. Algorithm — Gradient-Boosted Trees (conceptual, the XGBoost family)

```
INITIALIZE: F_0(x) = constant (e.g., mean of y for regression, log-odds for classification)
FOR m = 1 to M (number of boosting rounds):
    1. Compute PSEUDO-RESIDUALS: r_i = -[∂L(y_i, F(x_i))/∂F(x_i)]  evaluated at F = F_{m-1}
       (for squared error loss, this is simply y_i - F_{m-1}(x_i) -- the ordinary residual)
    2. FIT a new decision tree h_m(x) to predict these residuals r_i
    3. UPDATE: F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
RETURN F_M(x) = the final ensemble prediction
```
Each new tree corrects the *errors* of the ensemble so far — a sequential, additive, gradient-descent-in-function-space procedure (this is precisely why it's called "gradient" boosting: each tree approximates the negative gradient of the loss, exactly Phase 3 Lesson 5's optimization theory, but the "parameter" being optimized is the function $F$ itself, updated by adding trees rather than by adjusting a fixed-size parameter vector).

---

## 5. Python Implementation

```python
"""supervised_learning_core.py — from-first-principles model comparison"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


rng = np.random.default_rng(42)
n = 5000
age = rng.normal(50, 15, n).clip(18, 90)
smoker = rng.binomial(1, 0.25, n)
region_risk = rng.normal(0, 1, n)
# Non-additive interaction: smoking's mortality effect is amplified at older ages (a real actuarial pattern)
logit = -4 + 0.05 * age + 1.2 * smoker + 0.03 * age * smoker + 0.3 * region_risk
p_mortality = 1 / (1 + np.exp(-logit))
y = rng.binomial(1, p_mortality)
X = np.column_stack([age, smoker, region_risk])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric="logloss"),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"{name}: AUC = {auc:.4f}")
```

**Expected finding:** logistic regression (a linear model in the *original* features) will underperform tree-based methods here specifically *because* of the `age * smoker` interaction term baked into the data-generating process — logistic regression can't discover this interaction automatically unless you engineer it explicitly (Phase 2 Lesson 6), while tree-based models discover interactions natively via sequential splits.

---

## 6. Build From Scratch

**Logistic regression via gradient descent, entirely from scratch (directly using Section 3's derived gradient):**
```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iters: int = 1000):
    n, p = X.shape
    X_bias = np.column_stack([np.ones(n), X])   # add intercept term
    w = np.zeros(p + 1)
    for _ in range(n_iters):
        y_hat = sigmoid(X_bias @ w)
        gradient = X_bias.T @ (y_hat - y) / n      # EXACTLY the Section 3 derived gradient
        w -= lr * gradient
    return w

def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X_bias = np.column_stack([np.ones(len(X)), X])
    return sigmoid(X_bias @ w)
```

**A minimal decision tree (CART-style, Gini impurity) from scratch:**
```python
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature, self.threshold, self.left, self.right, self.value = feature, threshold, left, right, value

def gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)

def best_split(X: np.ndarray, y: np.ndarray):
    best_gain, best_feat, best_thresh = -1, None, None
    parent_impurity = gini(y)
    for feat in range(X.shape[1]):
        thresholds = np.unique(X[:, feat])
        for t in thresholds:
            left_mask = X[:, feat] <= t
            if left_mask.sum() == 0 or (~left_mask).sum() == 0:
                continue
            left_impurity = gini(y[left_mask])
            right_impurity = gini(y[~left_mask])
            n = len(y)
            weighted_impurity = (left_mask.sum()/n)*left_impurity + ((~left_mask).sum()/n)*right_impurity
            gain = parent_impurity - weighted_impurity
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, t
    return best_feat, best_thresh, best_gain

def build_tree(X: np.ndarray, y: np.ndarray, depth: int = 0, max_depth: int = 4) -> TreeNode:
    if depth >= max_depth or len(np.unique(y)) == 1:
        return TreeNode(value=np.bincount(y).argmax())
    feat, thresh, gain = best_split(X, y)
    if feat is None or gain <= 0:
        return TreeNode(value=np.bincount(y).argmax())
    left_mask = X[:, feat] <= thresh
    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right = build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth)
    return TreeNode(feature=feat, threshold=thresh, left=left, right=right)
```
Note the greedy split search's complexity: $O(p \times n \log n)$ per node (feature loop × sorted-threshold scan) — directly why tree training on wide, high-cardinality data can be slow, motivating XGBoost/LightGBM's histogram-based approximate split-finding optimizations.

---

## 7. Library Implementation (Comparison)

| From scratch | Production library |
|---|---|
| `train_logistic_regression` | `sklearn.linear_model.LogisticRegression` — uses more efficient solvers (L-BFGS, Phase 3 Lesson 5's Newton-family methods), handles regularization, multi-class natively |
| `build_tree` | `sklearn.tree.DecisionTreeClassifier` — optimized C implementation, handles many more split-quality/pruning options |
| Manual sequential boosting | `xgboost`/`lightgbm` — histogram-based approximate splits, regularization, second-order (Newton-style) gradient boosting, GPU support, massively faster |

---

## 8. Visual Explanations

**Bias-variance tradeoff as model complexity increases:**
```
Error
  │  \                                    ╱
  │   \  Bias² (decreasing)             ╱  Variance (increasing)
  │    \___                           ╱
  │        \___                    ╱
  │            \____          ___╱
  │                 \╲______╱
  │              Total error (U-shaped) -- minimum = optimal complexity
  └────────────────────────────────────────  model complexity
       underfitting          optimal          overfitting
```

**Gradient boosting: sequential residual-fitting:**
```
F0 (constant) ──▶ residuals r1 ──▶ fit tree h1 ──▶ F1 = F0 + lr·h1
                                                       │
                                            residuals r2 (from F1)
                                                       │
                                            fit tree h2 ──▶ F2 = F1 + lr·h2
                                                       │
                                                     ... (repeat M times) ...
```

---

## 9. Practical Examples

**Simple:** train logistic regression on a 2-feature synthetic dataset and visualize the decision boundary.
**Medium:** compare a decision tree's decision boundary against logistic regression's on data with a genuine non-linear (XOR-like) pattern, showing the tree captures it and the linear model cannot.
**Real-world:** train XGBoost on your actuarial mortality dataset with proper train/validation/test splits, tune `max_depth`/`learning_rate`/`n_estimators`, and compare against a logistic regression baseline with manually engineered interaction terms (Phase 2 Lesson 6) — directly quantifying how much "automatic interaction discovery" is worth versus manual feature engineering on your specific data.

---

## 10. Real Industry Use Cases

- **XGBoost/LightGBM/CatBoost**: dominate tabular ML competitions (Kaggle) and production systems (credit scoring, fraud detection, ad ranking, actuarial pricing) — still, in 2026, generally outperforming deep learning on structured/tabular data with moderate dataset sizes.
- **Logistic regression**: remains heavily used in regulated industries (insurance, credit) specifically because of its interpretability (coefficients have direct, auditable meaning) — a real, non-technical reason to prefer a "worse" model.
- **SVMs**: less dominant than in the 2000s-2010s but still used in specific high-dimensional, moderate-data-size regimes (text classification with sparse features, certain bioinformatics applications).
- **Random Forests**: a common robust "default" baseline before reaching for gradient boosting, valued for being harder to overfit and requiring less hyperparameter tuning.

---

## 11. Common Mistakes

- Using a linear model on data with strong non-additive interactions without engineering those interactions explicitly (Section 5's example) — silently leaves substantial predictive performance on the table.
- Over-trusting a single train/test split's performance number without cross-validation (Lesson 3, this phase) — especially dangerous on small actuarial/clinical datasets where a single split's randomness can substantially swing reported metrics.
- Ignoring regularization on linear models with many correlated features — leads to unstable, high-variance coefficient estimates (directly connecting to Phase 3 Lesson 1's ill-conditioning discussion).
- Growing decision trees to full depth without pruning/depth limits — near-guaranteed severe overfitting (memorizing training data, Section 2's bias-variance tradeoff in its most visible form).

---

## 12. Best Practices (2026)

- Default to gradient-boosted trees (XGBoost/LightGBM/CatBoost) as your first serious model for tabular data, with logistic/linear regression as an interpretable baseline for comparison and regulatory contexts.
- Always tune regularization (L1/L2 for linear models; `max_depth`/`min_child_weight`/`subsample` for tree ensembles) via proper cross-validation, never by eyeballing training-set performance.
- Use SHAP values (an information-theoretically grounded feature-attribution method, widely adopted since ~2017) for interpreting tree-ensemble predictions — critical in regulated domains like insurance, where "why did the model predict this" must have a defensible answer.
- Consider CatBoost specifically when you have many high-cardinality categorical features — it handles them natively via ordered target statistics (a more leakage-resistant relative of Phase 2 Lesson 6's target encoding) without manual preprocessing.

---

## 13. Exercises

**Easy:** Train logistic regression on a 2D synthetic dataset and plot the resulting linear decision boundary.
**Medium:** Implement the from-scratch decision tree (Section 6) and verify its predictions match `sklearn.tree.DecisionTreeClassifier` on a small dataset with the same `max_depth`.
**Hard:** Implement a minimal gradient boosting regressor from scratch (Section 4's algorithm, using the from-scratch decision tree as the weak learner) and verify its training loss decreases monotonically across boosting rounds.
**Mathematical:** Derive the gradient of the squared-error loss with respect to $F(x)$ in gradient boosting, and confirm it equals the ordinary residual $y - F(x)$.
**Coding:** Implement k-fold cross-validation from scratch (reusing Phase 3 Lesson 4's statistical framework) to compare logistic regression vs. XGBoost on your actuarial dataset, reporting mean and confidence interval (via bootstrap, Phase 3 Lesson 4) of the AUC difference.

---

## 14. Mini Project

Build a **complete supervised learning model comparison** on your actuarial mortality dataset: train logistic regression (with and without manually engineered interactions), a decision tree, a random forest, and XGBoost; evaluate all four via proper cross-validation with confidence intervals (Phase 3 Lesson 4's bootstrap); compute SHAP values for the best-performing tree ensemble to interpret which features drive predictions; and write a recommendation balancing predictive performance against interpretability requirements for an actual insurance pricing use case.

---

## 15. Interview Preparation

- Explain the bias-variance tradeoff and how it relates to model complexity and regularization.
- Derive the gradient of logistic regression's loss function with respect to the weights.
- How does gradient boosting differ from random forests (both being tree ensembles)?
- System design: how would you choose between an interpretable linear model and a higher-performing but opaque gradient-boosted tree ensemble for a regulated insurance pricing application?

---

## 16. Summary

Supervised learning formalizes learning $f: X \to Y$ from labeled examples, governed throughout by the bias-variance tradeoff: linear models (high bias, low variance, interpretable) sit at one end, deep decision trees (low bias, high variance, prone to overfitting without regularization) at the other, with ensemble methods (Lesson 5, this phase) and gradient boosting specifically designed to get the best of both. Every algorithm here — logistic regression's clean sigmoid-cross-entropy gradient, decision trees' greedy entropy/Gini-based splitting, gradient boosting's sequential residual-fitting — is a direct, traceable application of Phase 3's calculus, probability, and information theory.

---

## 17. References

- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (the definitive theoretical reference)
- Chen & Guestrin — "XGBoost: A Scalable Tree Boosting System" (2016, the original XGBoost paper)
- Lundberg & Lee — "A Unified Approach to Interpreting Model Predictions" (2017, the SHAP paper)
- James, Witten, Hastie, Tibshirani — *An Introduction to Statistical Learning* (more accessible companion to ESL)
