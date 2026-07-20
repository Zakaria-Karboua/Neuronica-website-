# Phase 4 · Lesson 4 — Feature Selection

> Prerequisite: Supervised Learning, Model Evaluation (Lessons 1, 3), Phase 2 Feature Engineering, Phase 3 Information Theory

---

## 1. Introduction

### What is feature selection?
The process of choosing a subset of available features that maximizes model performance/interpretability while discarding redundant, irrelevant, or noise-inducing ones. Distinct from (but complementary to) feature *engineering* (Phase 2 Lesson 6, which *creates* new features) and dimensionality *reduction* (Phase 4 Lesson 2's PCA, which *transforms* features into new combined ones rather than selecting a subset of the originals).

### Why does it exist?
More features are not always better: irrelevant features add noise and variance (Lesson 1's bias-variance tradeoff) without adding signal; redundant/correlated features can destabilize linear model coefficients (Phase 3 Lesson 1's ill-conditioning); and in high dimensions, the curse of dimensionality (Phase 2 Lesson 6) makes distance-based and even tree-based methods progressively less data-efficient per additional feature.

### Historical background
Feature selection has classical statistical roots (stepwise regression, dating to the mid-20th century) but gained renewed prominence with the genomics/bioinformatics era (2000s), where datasets routinely had far more features (genes) than samples (patients) — a regime ($p \gg n$) where feature selection isn't optional, it's a mathematical necessity (ordinary least squares is literally underdetermined when $p > n$).

### Real-world motivation
Your actuarial datasets, after feature engineering (region encodings, temporal features, interaction terms), can easily balloon to dozens or hundreds of candidate features — many redundant or noisy. Disciplined feature selection is what keeps the resulting model both performant and interpretable/auditable, a real regulatory concern in insurance.

---

## 2. Theory

### Three families of feature selection methods
| Family | Mechanism | Examples |
|---|---|---|
| **Filter methods** | Score each feature independently of any model, using a statistical criterion | correlation, mutual information (Phase 3 Lesson 6), chi-squared test |
| **Wrapper methods** | Use a model's actual performance to evaluate feature subsets | forward selection, backward elimination, recursive feature elimination (RFE) |
| **Embedded methods** | Feature selection happens *as part of* model training itself | Lasso (L1 regularization, Phase 3 Lesson 1), tree-based feature importances |

### Filter methods, formalized
- **Correlation-based**: fast but linear-relationship-only (Phase 2 Lesson 5's Anscombe warning applies directly).
- **Mutual-information-based** (Phase 3 Lesson 6): captures non-linear dependence, at higher computational cost (requires density/histogram estimation).
- **Variance threshold**: removes near-constant features outright — a cheap, always-worthwhile first pass.

### Wrapper methods, formalized
**Recursive Feature Elimination (RFE)**: train a model, rank features by importance/coefficient magnitude, remove the weakest, repeat — a greedy backward-elimination search through the (exponentially large, $2^p$) space of feature subsets, trading optimality for tractability, exactly analogous to Phase 1 Lesson 4's greedy algorithms.

### Embedded methods, formalized
Lasso's L1 penalty drives some coefficients exactly to zero (Phase 3 Lesson 1's geometric explanation: the L1 ball's corners), performing feature selection as a direct byproduct of regularized training — a single, unified optimization rather than a separate selection step.

---

## 3. Mathematical Foundations

### Why Lasso induces sparsity (geometric proof sketch, extending Phase 3 Lesson 1)
Minimizing $\|y-X\beta\|^2$ subject to $\|\beta\|_1 \le t$ is equivalent (via Lagrangian duality) to the penalized form. The L1 constraint region is a cross-polytope (diamond in 2D) with corners exactly on the coordinate axes; the unconstrained least-squares solution's elliptical contours generically first touch this diamond *at a corner* (where one or more coordinates are exactly zero) rather than on a flat edge — geometrically guaranteeing sparse solutions, unlike Ridge's circular constraint region, which has no corners and thus (generically) never produces exact zeros.

### Multicollinearity and Variance Inflation Factor (VIF)
For feature $X_j$, regress it against all *other* features to get $R_j^2$:
$$
\text{VIF}_j = \frac{1}{1-R_j^2}
$$
$\text{VIF}_j$ large (commonly, $>10$ used as a rule of thumb) indicates $X_j$ is well-predicted by other features — its own coefficient estimate in a linear model becomes highly unstable (large variance) as a direct consequence of the near-singular $X^TX$ matrix (Phase 3 Lesson 1's conditioning problem, now given a concrete per-feature diagnostic).

### Mutual information vs. correlation for filter-based selection (worked comparison)
As established in Phase 3 Lesson 6, mutual information $I(X_j;Y)$ captures *any* statistical dependence, while Pearson correlation captures only linear dependence — meaning a filter-based feature selection pipeline built purely on correlation will systematically discard genuinely predictive non-linear features that a mutual-information-based (or tree-importance-based) filter would correctly retain.

### Stability selection (a modern, more rigorous refinement)
Running Lasso (or any selection method) repeatedly on bootstrap resamples (Phase 3 Lesson 4) of the data, and retaining only features selected in a high proportion of resamples, addresses a real weakness of single-run feature selection: with correlated features, small data perturbations can flip *which* of two correlated features gets selected, even though the underlying signal is stable — stability selection reports the *set* of features robustly important across resampling, not an artifact of one particular training run.

---

## 4. Algorithm — Recursive Feature Elimination (RFE)

```
GIVEN a dataset with p features and a target number of features k to keep:
1. Train a model (e.g., logistic regression or a tree ensemble) on ALL p features
2. Rank features by importance (coefficient magnitude, or tree-based importance score)
3. REMOVE the single least important feature
4. RETRAIN the model on the remaining p-1 features
5. REPEAT steps 2-4 until exactly k features remain
RETURN the final k-feature subset and the retrained model
```
Complexity: $O(p)$ model retrainings (one per removed feature) — computationally more expensive than filter methods (which score all features in one pass) but directly optimizes for the *actual downstream model's* performance rather than a proxy statistical criterion, at the cost of being tied to whichever model is used during the RFE process itself.

---

## 5. Python Implementation

```python
"""feature_selection_core.py"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFE, VarianceThreshold
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Flags multicollinear features BEFORE they destabilize a linear model's coefficients."""
    vif_data = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    return vif_data.sort_values("VIF", ascending=False)


def filter_by_mutual_information(X: np.ndarray, y: np.ndarray, feature_names: list[str], top_k: int = 10):
    mi_scores = mutual_info_classif(X, y, random_state=0)
    ranked = sorted(zip(feature_names, mi_scores), key=lambda t: -t[1])
    return ranked[:top_k]


def lasso_embedded_selection(X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    """Embedded method: Lasso's L1 penalty selects features AS PART OF training."""
    lasso = LassoCV(cv=5, random_state=0).fit(X, y)
    selected = [name for name, coef in zip(feature_names, lasso.coef_) if abs(coef) > 1e-6]
    return selected, lasso.coef_


def rfe_wrapper_selection(X: np.ndarray, y: np.ndarray, feature_names: list[str], n_features: int = 5):
    estimator = RandomForestClassifier(n_estimators=100, random_state=0)
    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X, y)
    return [name for name, keep in zip(feature_names, selector.support_) if keep]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    age = rng.normal(45, 15, n)
    age_duplicate = age + rng.normal(0, 0.5, n)   # near-perfect collinear duplicate (VIF should flag this)
    smoker = rng.binomial(1, 0.25, n)
    noise1 = rng.normal(0, 1, n)                   # pure noise, should be filtered out
    noise2 = rng.normal(0, 1, n)

    X_df = pd.DataFrame({
        "age": age, "age_duplicate": age_duplicate, "smoker": smoker, "noise1": noise1, "noise2": noise2
    })
    y = (0.05 * age + 1.5 * smoker + rng.normal(0, 1, n) > 4).astype(int)

    print(compute_vif(X_df))
    print(filter_by_mutual_information(X_df.values, y, X_df.columns.tolist()))
    selected, coefs = lasso_embedded_selection(X_df.values, y, X_df.columns.tolist())
    print("Lasso-selected features:", selected)
```

**Expected finding:** `age_duplicate` should show a very high VIF (flagging the collinearity with `age`), and pure noise features should score low on mutual information and often get zeroed out by Lasso — a concrete, runnable demonstration of every method's intended behavior.

---

## 6. Build From Scratch

**A minimal stability selection implementation (Section 3's refinement, from scratch):**
```python
import numpy as np
from sklearn.linear_model import Lasso

def stability_selection(X: np.ndarray, y: np.ndarray, feature_names: list[str],
                          n_bootstraps: int = 100, alpha: float = 0.1, threshold: float = 0.6):
    rng = np.random.default_rng(0)
    n, p = X.shape
    selection_counts = np.zeros(p)

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)          # bootstrap resample (Phase 3 Lesson 4)
        X_boot, y_boot = X[idx], y[idx]
        lasso = Lasso(alpha=alpha).fit(X_boot, y_boot)
        selection_counts += (np.abs(lasso.coef_) > 1e-6).astype(int)

    selection_freq = selection_counts / n_bootstraps
    stable_features = [name for name, freq in zip(feature_names, selection_freq) if freq >= threshold]
    return stable_features, selection_freq
```
Running this alongside a *single* Lasso fit on correlated features often reveals exactly the instability Section 3 describes: two genuinely-correlated-but-both-somewhat-informative features may each get selected in only ~50% of bootstrap runs individually (each "taking turns" being zeroed out), while stability selection's *frequency* view correctly reveals that the *pair* jointly carries stable signal, even if any single Lasso run arbitrarily favors one over the other.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `stability_selection` | `sklearn`'s `RandomizedLasso` (older) / manually wrapping `LassoCV` with bootstrap as shown — this genuinely is close to production practice, no major library gap here |
| `compute_vif` | `statsmodels.stats.outliers_influence.variance_inflation_factor` — used directly, not a simplified stand-in |
| Manual RFE loop | `sklearn.feature_selection.RFE`/`RFECV` (cross-validated variant, choosing $k$ automatically) |
| Manual mutual information filter | `sklearn.feature_selection.mutual_info_classif`/`SelectKBest` |

---

## 8. Visual Explanations

**Why Lasso (L1) produces sparse solutions, Ridge (L2) doesn't (2D geometric view, from Phase 3 Lesson 1, revisited):**
```
Lasso (L1 diamond constraint):        Ridge (L2 circular constraint):
        β2                                    β2
        │  ╱╲                                 │   ___
        │ ╱  ╲   <- elliptical loss           │  /   \
    ────┼╱────╲──── β1   contours often       ──┼──○────  β1  contours often
        ╲      ╱     touch a CORNER            │  \___/    touch a SMOOTH point
         ╲____╱      (β1 or β2 = 0 exactly)     │            (neither exactly 0)
```

**Filter vs. Wrapper vs. Embedded selection (workflow position):**
```
Filter:    [ALL features] -> score independently -> [select top-k] -> [train model]
Wrapper:   [ALL features] -> train/evaluate model repeatedly, removing features -> [final subset]
Embedded:  [ALL features] -> [train model WITH built-in selection, e.g. Lasso] -> [selection is a BYPRODUCT]
```

---

## 9. Practical Examples

**Simple:** compute Pearson correlation and mutual information between 5 synthetic features and a target, comparing rankings.
**Medium:** run RFE with a random forest on a dataset with known irrelevant noise features, verifying the noise features are correctly eliminated.
**Real-world:** apply VIF-based multicollinearity screening, followed by Lasso-embedded selection, to your actuarial feature set (post Phase 2 Lesson 6's engineering pass) — producing a final, defensible, minimal feature set suitable for a regulator-facing interpretable pricing model.

---

## 10. Real Industry Use Cases

- **Genomics/bioinformatics**: feature selection is not optional when $p$ (genes, often tens of thousands) vastly exceeds $n$ (patients, often hundreds) — a regime where naive modeling is mathematically impossible without selection or regularization.
- **Regulated insurance pricing models**: regulators often require justification for every feature used in a filed rating model — disciplined, documented feature selection (not just "the model picked it") is a genuine compliance requirement, not just good practice.
- **Credit scoring**: legal requirements (e.g., anti-discrimination regulations) sometimes mandate excluding certain features or their close proxies entirely, making feature selection partly a legal/ethical exercise, not purely a statistical one.
- **High-dimensional NLP/embedding features** (bridging to Phase 6): feature selection principles reappear, though the *techniques* differ substantially — dimensionality reduction (PCA, Lesson 2) and learned sparse representations are more common than classical filter/wrapper methods at that scale.

---

## 11. Common Mistakes

- Performing feature selection using the *entire* dataset (including what will become the test set) before splitting — a data leakage error structurally identical to Phase 2 Lesson 6's target-encoding leakage trap; feature selection must happen *inside* the cross-validation loop, using only training folds.
- Relying solely on correlation-based filtering and discarding genuinely predictive non-linear features (Phase 3 Lesson 6's mutual information lesson, directly relevant again here).
- Ignoring multicollinearity in a linear/logistic regression model, producing wildly unstable, uninterpretable coefficients even when the model's *predictions* look fine.
- Treating a single Lasso run's selected features as a stable, final answer without stability selection or cross-validation, when correlated-feature "coin flips" (Section 6) may make the specific selection somewhat arbitrary.

---

## 12. Best Practices (2026)

- Always perform feature selection *within* cross-validation folds, never on the full dataset before splitting — directly analogous to Phase 2 Lesson 6's leakage-safe target encoding discipline.
- Use VIF screening as a fast, standard pre-check before fitting any linear/logistic regression model with many correlated engineered features.
- Prefer tree-ensemble feature importances or Lasso-embedded selection as robust defaults over pure filter methods, reserving filter methods (correlation/mutual information) for fast initial triage during EDA (Phase 2 Lesson 5).
- Use stability selection (or cross-validated RFE, `RFECV`) rather than a single selection run whenever the selected feature *set* itself will be reported/justified to stakeholders or regulators.

---

## 13. Exercises

**Easy:** Compute VIF for a small set of features including one deliberately collinear pair, and confirm it correctly flags the collinearity.
**Medium:** Compare filter-based (mutual information) vs. embedded (Lasso) feature selection on a dataset with both linear and non-linear feature-target relationships, discussing which method retains which features and why.
**Hard:** Implement stability selection from scratch (Section 6) and empirically demonstrate the "coin-flip" instability of a single Lasso run on two correlated, jointly-informative features.
**Mathematical:** Prove geometrically (extending Phase 3 Lesson 1) why the L1 constraint region's corners, and not the L2 constraint region's smooth boundary, generically produce exact-zero coefficients.
**Coding:** Implement cross-validated RFE (`RFECV`-style) from scratch, automatically choosing the number of features that maximizes validation performance rather than fixing $k$ in advance.

---

## 14. Mini Project

Build a **complete, leakage-safe feature selection pipeline** for your actuarial dataset, performed correctly *within* a cross-validation loop: VIF-based multicollinearity screening, mutual-information-based filtering for a fast initial pass, stability-selection-refined Lasso for the final embedded selection, and a comparison of model performance (via Phase 4 Lesson 3's proper nested CV) using the full feature set versus the selected subset — quantifying whether feature selection actually improved out-of-sample performance, hurt it negligibly while improving interpretability, or both.

---

## 15. Interview Preparation

- Explain the difference between filter, wrapper, and embedded feature selection methods, with an example of each.
- Why does Lasso (L1) produce sparse solutions while Ridge (L2) does not?
- What is VIF, and how would you use it to diagnose multicollinearity in a linear model?
- Why must feature selection happen inside a cross-validation loop rather than before splitting the data?

---

## 16. Summary

Feature selection disciplines a feature set down to what's genuinely useful: filter methods (correlation, mutual information) offer fast, model-agnostic first-pass screening; wrapper methods (RFE) optimize directly for a specific model's performance at higher computational cost; and embedded methods (Lasso) fold selection directly into training via L1 regularization's sparsity-inducing geometry. VIF-based multicollinearity screening and stability selection (bootstrap-refined selection) address the two biggest practical failure modes — unstable linear coefficients and arbitrary single-run selection instability — and, as with feature engineering (Phase 2 Lesson 6), every selection step must happen strictly within cross-validation to avoid leakage.

---

## 17. References

- Guyon, I. & Elisseeff, A. — "An Introduction to Variable and Feature Selection" (2003, the standard survey)
- Tibshirani, R. — "Regression Shrinkage and Selection via the Lasso" (1996, the original Lasso paper)
- Meinshausen, N. & Bühlmann, P. — "Stability Selection" (2010)
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, Chapter 3 (Linear Methods, Lasso/Ridge geometry)
