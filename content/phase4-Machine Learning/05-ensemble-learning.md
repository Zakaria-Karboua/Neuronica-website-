# Phase 4 · Lesson 5 — Ensemble Learning

> Prerequisite: Supervised Learning, Model Evaluation (Lessons 1, 3)

---

## 1. Introduction

### What is ensemble learning?
Combining multiple models' predictions to produce a result more accurate and/or robust than any single model alone. Random Forests and XGBoost — already used extensively in Lesson 1 — are themselves ensembles; this lesson formalizes *why* combining models works, the mathematical conditions under which it helps, and the broader taxonomy (bagging, boosting, stacking) beyond the two examples already used.

### Why does it exist?
A single model's errors reflect its particular biases and its particular sensitivity to the training sample it happened to see. If multiple models' errors are not perfectly correlated, combining them can cancel out some of that noise — a direct, formalizable consequence of the bias-variance tradeoff (Lesson 1) and basic statistics (averaging reduces variance).

### Historical background
Bagging (Bootstrap Aggregating) was introduced by Breiman (1996), followed by Random Forests (Breiman, 2001) adding random feature subsampling to further decorrelate trees. Boosting has a distinct lineage: AdaBoost (Freund & Schapire, 1995) was the first practical boosting algorithm, later reinterpreted as a form of gradient descent in function space (Friedman, 2001), generalizing directly into modern Gradient Boosting Machines and XGBoost/LightGBM/CatBoost.

### Real-world motivation
XGBoost, which you've already used for your mortality model, IS an ensemble method (boosting). This lesson explains precisely why it works, when a different ensemble strategy (bagging, stacking) might serve you better, and how to combine heterogeneous models (e.g., XGBoost + logistic regression + a neural network) via stacking for further gains.

---

## 2. Theory

### Bagging (Bootstrap Aggregating)
Train $B$ independent models, each on a different bootstrap resample (Phase 3 Lesson 4) of the training data, and average their predictions (regression) or vote (classification). Reduces **variance** without changing bias — most effective for high-variance, low-bias base models (deep, unpruned decision trees) that overfit individually but whose errors are relatively uncorrelated across resamples.

### Random Forests (bagging + feature randomness)
Adds a second randomization layer beyond bagging: at each tree split, only a random subset of features is considered — deliberately **decorrelating** the trees further (two trees might otherwise make very similar splits near the top, since the strongest features dominate; forcing some splits to ignore the top feature spreads out which features drive which trees).

### Boosting
Train models **sequentially**, each new model focusing on the previous ensemble's mistakes (Lesson 1's gradient boosting algorithm, generalized here). Reduces **bias** primarily (weak, high-bias base learners like shallow "stumps" become a strong learner collectively) but can also reduce variance with proper regularization (learning rate, tree depth limits, subsampling).

### Stacking (stacked generalization)
Train several *heterogeneous* base models (e.g., logistic regression, random forest, XGBoost, a neural network), then train a **meta-model** on their out-of-fold predictions to learn the optimal way to combine them — potentially capturing that, e.g., the neural network is more reliable for one data region while XGBoost is more reliable for another.

---

## 3. Mathematical Foundations

### Why averaging reduces variance (the core mathematical justification for bagging)
If $B$ models each have prediction variance $\sigma^2$ and pairwise correlation $\rho$, the variance of their average is:
$$
\text{Var}\left(\frac{1}{B}\sum_{i=1}^B \hat f_i(x)\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$
As $B \to \infty$, this converges to $\rho\sigma^2$ — **not zero**. This is the precise mathematical reason Random Forests decorrelate trees via feature randomness (Section 2): reducing $\rho$ directly lowers the achievable variance floor, whereas simply adding more trees ($B$) alone eventually hits diminishing returns if $\rho$ stays high.

### Bias-variance decomposition of bagging (formalized)
Bagging leaves bias essentially unchanged (the average of many similarly-biased models is still similarly biased) while reducing variance per the formula above — this is *why* bagging pairs best with low-bias, high-variance base learners (deep trees): there's substantial variance to remove, and bias isn't being made worse.

### AdaBoost's exponential loss, derived
AdaBoost minimizes exponential loss $L = \sum_i e^{-y_i F(x_i)}$ ($y_i \in \{-1,+1\}$) in a stagewise-additive manner. At each round, samples are reweighted:
$$
w_i^{(t+1)} = w_i^{(t)} \cdot e^{\alpha_t \cdot \mathbb{1}[y_i \ne h_t(x_i)]}, \qquad \alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)
$$
where $\epsilon_t$ is the weighted error rate of the $t$-th weak learner. Misclassified samples get *up-weighted*, forcing the next weak learner to focus on the current ensemble's mistakes — Friedman (2001) later showed this is a special case of gradient descent in function space using exponential loss, unifying AdaBoost with the general gradient boosting framework from Lesson 1.

### Stacking's out-of-fold requirement (avoiding leakage, directly reusing Lesson 4's principle)
If base models' predictions used to train the meta-model come from data those base models were *trained on*, the meta-model sees artificially confident (overfit) predictions — a leakage pattern structurally identical to Phase 2 Lesson 6's target-encoding leakage. The correct procedure generates each base model's predictions via out-of-fold cross-validation (identical mechanism to Phase 2 Lesson 6's K-fold target encoding), ensuring the meta-model only ever sees honest, held-out-style predictions during its own training.

---

## 4. Algorithm — Stacking (fully specified, leakage-safe)

```
GIVEN base models M1, M2, ..., Mk and a meta-model MetaM, k_folds for out-of-fold generation:
1. FOR each base model M_j:
     generate OUT-OF-FOLD predictions across the entire training set:
       (train M_j on folds != i, predict on fold i, for every fold i -- Phase 2 Lesson 6's exact mechanism)
     -> this produces oof_preds_j, one prediction per training row, NONE overfit
2. STACK these out-of-fold predictions as a new feature matrix: Z = [oof_preds_1, ..., oof_preds_k]
3. TRAIN the meta-model on Z (predicting the true target y) -- learns how to best COMBINE base models
4. FOR PREDICTION on new data:
     retrain each M_j on the FULL training set (no folding needed at inference time)
     get each M_j's prediction on the new data -> feed into the trained MetaM -> final prediction
```

---

## 5. Python Implementation

```python
"""ensemble_learning_core.py"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def leakage_safe_stacking(X: np.ndarray, y: np.ndarray, base_models: dict, meta_model, k_folds: int = 5):
    n = len(y)
    oof_preds = np.zeros((n, len(base_models)))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    for j, (name, model) in enumerate(base_models.items()):
        for train_idx, val_idx in kf.split(X):
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx, j] = model_clone.predict_proba(X[val_idx])[:, 1]

    meta_model.fit(oof_preds, y)

    # Refit base models on FULL data for actual future predictions
    for model in base_models.values():
        model.fit(X, y)

    return base_models, meta_model


def stacked_predict(X_new: np.ndarray, base_models: dict, meta_model) -> np.ndarray:
    base_preds = np.column_stack([m.predict_proba(X_new)[:, 1] for m in base_models.values()])
    return meta_model.predict_proba(base_preds)[:, 1]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] * 0.8 + X[:, 1] ** 2 - X[:, 2] * X[:, 3] + rng.normal(0, 0.5, n) > 0.5).astype(int)
    X_train, X_test = X[:2400], X[2400:]
    y_train, y_test = y[:2400], y[2400:]

    base_models = {
        "logreg": LogisticRegression(),
        "rf": RandomForestClassifier(n_estimators=100, random_state=0),
        "xgb": xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss"),
    }
    meta = LogisticRegression()
    fitted_bases, fitted_meta = leakage_safe_stacking(X_train, y_train, base_models, meta)
    stacked_preds = stacked_predict(X_test, fitted_bases, fitted_meta)
    print("Stacked ensemble AUC:", roc_auc_score(y_test, stacked_preds))

    for name, model in fitted_bases.items():
        individual_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"{name} individual AUC:", individual_auc)

    # Bagging and boosting comparison, using sklearn's built-ins
    bagged = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0).fit(X_train, y_train)
    boosted = AdaBoostClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
    print("Bagged trees AUC:", roc_auc_score(y_test, bagged.predict_proba(X_test)[:, 1]))
    print("AdaBoost AUC:", roc_auc_score(y_test, boosted.predict_proba(X_test)[:, 1]))
```

---

## 6. Build From Scratch

**AdaBoost from scratch (directly implementing Section 3's reweighting formula):**
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def adaboost_from_scratch(X: np.ndarray, y: np.ndarray, n_estimators: int = 50):
    """y must be in {-1, +1} for the exponential-loss formulation."""
    n = len(y)
    weights = np.ones(n) / n
    models, alphas = [], []

    for _ in range(n_estimators):
        stump = DecisionTreeClassifier(max_depth=1)   # a "weak learner" -- barely better than random
        stump.fit(X, y, sample_weight=weights)
        preds = stump.predict(X)

        weighted_error = np.sum(weights * (preds != y)) / np.sum(weights)
        weighted_error = np.clip(weighted_error, 1e-10, 1 - 1e-10)   # avoid log(0)/div-by-0
        alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)   # EXACTLY Section 3's formula

        weights *= np.exp(-alpha * y * preds)   # up-weight misclassified, down-weight correct
        weights /= weights.sum()                 # renormalize to a valid distribution

        models.append(stump)
        alphas.append(alpha)

    return models, alphas

def adaboost_predict(X: np.ndarray, models: list, alphas: list) -> np.ndarray:
    agg = sum(a * m.predict(X) for a, m in zip(alphas, models))
    return np.sign(agg)
```
Each weak learner here is a "decision stump" (`max_depth=1`, barely better than a coin flip individually) — yet the boosted ensemble, via the exact reweighting mechanism derived in Section 3, becomes a strong classifier, the historically foundational demonstration that boosting "weak learners" into a "strong learner" is not just a slogan but a provable, mechanically explicit result.

---

## 7. Library/Tool Comparison

| From scratch | Production library |
|---|---|
| `adaboost_from_scratch` | `sklearn.ensemble.AdaBoostClassifier` — handles multi-class, more numerically robust weight normalization |
| `leakage_safe_stacking` | `sklearn.ensemble.StackingClassifier` (handles the out-of-fold mechanics internally, equivalent logic) |
| Manual bagging via bootstrap loop | `sklearn.ensemble.BaggingClassifier`/`RandomForestClassifier` — parallelized tree training, optimized C implementations |
| Manual gradient boosting (Lesson 1) | `xgboost`/`lightgbm`/`catboost` — histogram-based splits, regularization, second-order gradient information, GPU support |

---

## 8. Visual Explanations

**Bagging (parallel, variance reduction) vs. Boosting (sequential, bias reduction):**
```
BAGGING (parallel, independent):          BOOSTING (sequential, dependent):
  Data ──┬──▶ Model 1 (bootstrap A)         Data ──▶ Model 1 ──▶ residuals/reweighting
         ├──▶ Model 2 (bootstrap B)                              │
         └──▶ Model 3 (bootstrap C)                               ▼
                    │                                    Model 2 (focuses on M1's errors)
              AVERAGE/VOTE                                          │
                                                                     ▼
                                                          Model 3 (focuses on remaining errors)
                                                                     │
                                                            WEIGHTED SUM
```

**Effect of correlation ρ on ensemble variance (Section 3's formula, visualized):**
```
Variance
  │  ╲
  │   ╲___
  │       ╲______            <- floor = ρσ², NEVER reaches zero even as B→∞
  │              ‾‾‾‾‾‾‾‾‾‾‾‾
  └────────────────────────── number of models B
  (adding more CORRELATED models: diminishing returns, hits a floor)
  (REDUCING ρ, e.g. via Random Forest's feature randomness: LOWERS the floor itself)
```

---

## 9. Practical Examples

**Simple:** compare a single decision tree's variance (across different bootstrap training samples) against a bagged ensemble's variance on the same data.
**Medium:** implement AdaBoost from scratch (Section 6) and visualize how the sample weights evolve across boosting rounds, highlighting which points become "hard examples."
**Real-world:** build a stacked ensemble combining logistic regression (interpretable baseline), random forest, and XGBoost for your actuarial mortality prediction, using leakage-safe out-of-fold stacking, and quantify (via Phase 4 Lesson 3's nested CV + bootstrap CI) whether the stacked ensemble meaningfully outperforms the best individual model.

---

## 10. Real Industry Use Cases

- **XGBoost/LightGBM/CatBoost**: gradient boosting dominates tabular ML in production across finance, insurance, ad tech, and Kaggle competitions.
- **Random Forests**: remain a robust, low-maintenance default in many production systems where extensive hyperparameter tuning isn't feasible or where variance/robustness matters more than squeezing out the last percent of performance.
- **Stacking in Kaggle competitions**: virtually every top-placing solution in tabular Kaggle competitions uses some form of stacking/blending multiple diverse model types — a well-documented, empirically proven technique for squeezing out final performance gains.
- **Netflix Prize (2006-2009)**: the winning solution was itself a large stacked ensemble of over 100 individual models/techniques — an early, famous, large-scale demonstration of stacking's power.

---

## 11. Common Mistakes

- Stacking base models whose out-of-fold predictions were generated with leakage (models trained on data overlapping their own prediction set) — silently inflates the meta-model's apparent performance, an exact structural repeat of Phase 2 Lesson 6 and Phase 4 Lesson 4's leakage traps.
- Using bagging with already low-variance, high-bias base models (e.g., bagging shallow decision stumps) — provides little benefit, since there's little variance to reduce (bagging fixes variance problems, not bias problems).
- Boosting for too many rounds without adequate regularization (learning rate, tree depth, subsampling) — can eventually overfit, despite boosting's generally strong resistance to overfitting relative to naive expectations.
- Assuming an ensemble is automatically better than its best individual component — if base models are highly correlated (Section 3's $\rho$), the ensemble may provide negligible improvement over simply using the single best model.

---

## 12. Best Practices (2026)

- Use Random Forests as a robust, low-tuning-effort baseline; reach for gradient boosting (XGBoost/LightGBM/CatBoost) when squeezing out maximum tabular performance, tuned via proper nested cross-validation (Phase 4 Lesson 3).
- When stacking, deliberately include *diverse* base model types (linear, tree-based, potentially neural) rather than several near-identical models — diversity (low $\rho$) is what drives real stacking gains, not just model count.
- Always generate stacking's base-model predictions via proper out-of-fold cross-validation, never using in-sample predictions.
- Monitor for overfitting in boosting via a validation set and early stopping (halt adding trees once validation performance stops improving) — a standard, essential safeguard in every serious XGBoost/LightGBM training pipeline.

---

## 13. Exercises

**Easy:** Train a single decision tree and a bagged ensemble of 50 trees on the same dataset; compare their variance across 10 different train/test splits.
**Medium:** Implement AdaBoost from scratch (Section 6) on a small binary classification dataset and plot the training error as a function of the number of boosting rounds.
**Hard:** Implement leakage-safe stacking from scratch (Section 6) combining 3 diverse base models, and empirically demonstrate the performance inflation that occurs if you skip the out-of-fold requirement and use in-sample base predictions instead.
**Mathematical:** Derive the variance-of-average formula (Section 3) for $B$ models with pairwise correlation $\rho$, starting from the definition of variance of a sum of correlated random variables.
**Coding:** Implement Random Forest's feature-randomness mechanism from scratch (modify Phase 4 Lesson 1's from-scratch decision tree to consider only a random feature subset at each split) and empirically show it reduces inter-tree prediction correlation compared to plain bagging without feature randomness.

---

## 14. Mini Project

Build a **complete ensemble comparison and stacking pipeline** for your actuarial dataset: train and nested-cross-validate (Phase 4 Lesson 3) a single decision tree, a Random Forest, AdaBoost, and XGBoost individually; then build a leakage-safe stacked ensemble combining the most diverse subset of these; report bootstrap confidence intervals (Phase 3 Lesson 4) for every model's ROC-AUC; and write a final recommendation on whether the added complexity of stacking is justified by the actual performance gain versus simply using the best-tuned individual model.

---

## 15. Interview Preparation

- Explain the difference between bagging and boosting, and why one primarily reduces variance while the other primarily reduces bias.
- Derive (or explain conceptually) why Random Forest's feature-randomness mechanism improves over plain bagging of decision trees.
- What is stacking, and what leakage risk must be carefully avoided when implementing it?
- Why doesn't averaging an infinite number of correlated models drive ensemble variance to zero?

---

## 16. Summary

Ensemble learning formalizes a genuinely powerful, mathematically justified idea: combining models can reduce variance (bagging, via averaging correlated-but-not-identical errors, with Random Forest's feature randomness specifically lowering that correlation floor), reduce bias (boosting, via sequential error-correction, unifying AdaBoost and gradient boosting under one framework), or optimally blend heterogeneous strong models (stacking, when done leakage-safely via out-of-fold predictions). XGBoost — likely your primary tool already — is gradient boosting; this lesson gives you the theoretical grounding for when to reach for a different ensemble strategy instead, and how to combine multiple strategies for further gains.

---

## 17. References

- Breiman, L. — "Bagging Predictors" (1996) and "Random Forests" (2001)
- Freund, Y. & Schapire, R. — "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting" (1997, AdaBoost)
- Friedman, J. — "Greedy Function Approximation: A Gradient Boosting Machine" (2001)
- Wolpert, D. — "Stacked Generalization" (1992, the original stacking paper)
