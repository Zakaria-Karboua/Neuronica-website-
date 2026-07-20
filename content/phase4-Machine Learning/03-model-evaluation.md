# Phase 4 · Lesson 3 — Model Evaluation

> Prerequisite: Supervised Learning, Unsupervised Learning (Lessons 1–2), Phase 3 Statistics

---

## 1. Introduction

### What is model evaluation?
The systematic, rigorous measurement of how well a model performs — not just a single accuracy number, but the right metric for the problem, honest estimation via proper data splitting, and statistically sound comparison between candidate models. This is arguably the single most consequential lesson in this curriculum: a model evaluated wrong looks great in development and fails, sometimes catastrophically, in production.

### Why does it exist?
A model's performance on the data it was trained on is a systematically optimistic, unreliable estimate of its real-world performance (directly Lesson 1's bias-variance/overfitting concern). Model evaluation exists to produce an honest estimate of generalization performance *before* deployment, using metrics that actually reflect what matters for the business/clinical/actuarial decision being made.

### Historical background
Cross-validation traces to the 1930s-40s (early "hold-out" methods in statistics); k-fold cross-validation and the accuracy/precision/recall framework matured through 20th-century pattern recognition and information retrieval research. In 2026, evaluation methodology has expanded further to address the specific failure modes of deployed ML systems: calibration, fairness across subgroups, and robustness to distribution shift.

### Real-world motivation
An insurance model with 95% accuracy sounds impressive — until you learn the base rate of the event being predicted is 95% negative, making a model that predicts "no" every single time also score 95% accuracy while being completely useless. This lesson's job is making sure you never fall into that trap, or dozens of others like it.

---

## 2. Theory

### Train/validation/test splitting
- **Training set**: used to fit model parameters.
- **Validation set**: used to tune hyperparameters and select among model candidates.
- **Test set**: touched exactly once, at the very end, to report final honest performance — never used for any decision-making along the way (using it repeatedly for model selection turns it into a second validation set, silently reintroducing overfitting at the *model-selection* level, a subtle but real trap).

### Cross-validation
$k$-fold CV: partition data into $k$ folds, train on $k-1$, validate on the remaining fold, rotate, average results — gives a more stable performance estimate than a single split, especially valuable on smaller datasets (directly relevant to actuarial/clinical data, often limited in size). **Stratified** k-fold preserves class balance in each fold — essential for imbalanced classification (rare disease, rare fraud, rare mortality events).

### Classification metrics (and why accuracy alone routinely misleads)
| Metric | Formula | When it matters most |
|---|---|---|
| Accuracy | $(TP+TN)/n$ | only when classes are balanced |
| Precision | $TP/(TP+FP)$ | cost of false positives is high (e.g., flagging a legitimate claim as fraud) |
| Recall (Sensitivity) | $TP/(TP+FN)$ | cost of false negatives is high (e.g., missing a genuine high-risk patient) |
| F1 score | $2\cdot\frac{P \cdot R}{P+R}$ | balances precision/recall into one number |
| ROC-AUC | area under TPR vs. FPR curve | threshold-independent ranking quality |
| PR-AUC | area under Precision vs. Recall curve | preferred over ROC-AUC on strongly imbalanced data |

### Calibration — does a predicted probability mean what it says?
A model is **well-calibrated** if, among all instances predicted to have probability $p$ of the positive class, approximately $p$ fraction actually are positive. A model can have excellent ranking ability (high AUC) while being badly calibrated (systematically over- or under-confident) — a critical distinction for actuarial applications, where the *actual probability value* (not just the ranking) directly determines pricing/reserving.

### Regression metrics
$$
\text{MAE} = \frac{1}{n}\sum|y_i-\hat y_i|, \quad \text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i-\hat y_i)^2}, \quad R^2 = 1 - \frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}
$$
RMSE penalizes large errors more heavily than MAE (squared term) — appropriate when large errors are disproportionately costly (a common actuarial reality: one massively mispriced catastrophic claim matters more than being off by a small amount on many small claims).

---

## 3. Mathematical Foundations

### Why accuracy fails on imbalanced data (formalized)
If the positive class prevalence is $\pi$, a trivial "always predict negative" classifier achieves accuracy $1-\pi$. For rare-event prediction (e.g., $\pi = 0.02$ for a rare disease), this trivial classifier scores 98% accuracy while having **zero** recall — utterly useless, yet numerically appearing excellent. This directly motivates precision/recall/PR-AUC as the honest metrics for imbalanced problems.

### ROC-AUC as a probabilistic interpretation
$$
\text{AUC} = P(\hat y_{\text{pos}} > \hat y_{\text{neg}})
$$
i.e., AUC is the probability that a randomly chosen positive example receives a higher predicted score than a randomly chosen negative example — a genuinely elegant, threshold-independent ranking-quality interpretation, computable directly via the Mann-Whitney U statistic (a non-parametric hypothesis test, Phase 3 Lesson 4's family).

### Calibration measurement: the Brier score and reliability diagrams
$$
\text{Brier score} = \frac{1}{n}\sum_i (\hat p_i - y_i)^2
$$
This decomposes (Murphy, 1973) into **calibration** (how close predicted probabilities are to true frequencies) and **refinement/resolution** (how well the model discriminates) components — a model can improve its Brier score by improving either calibration or discrimination, and the decomposition tells you which is the actual problem when a model's raw score disappoints.

### Cross-validation's bias-variance tradeoff in $k$
Higher $k$ (more folds, e.g., leave-one-out): lower bias (each training fold uses almost all the data) but higher variance (folds are highly correlated with each other, since they share almost all their data) and higher computational cost. Lower $k$ (e.g., $k=5$): more bias (each fold trains on less data) but lower variance and cost. $k=5$ or $k=10$ is the standard practical compromise, empirically validated across countless studies rather than derived from a single clean formula.

---

## 4. Algorithm — Nested Cross-Validation (for honest hyperparameter tuning + evaluation)

```
GIVEN a dataset and a model with hyperparameters to tune:
OUTER LOOP (k_outer folds, e.g. 5): -- for HONEST performance estimation
    FOR each outer fold:
        hold out this fold as the outer TEST set
        INNER LOOP (k_inner folds, e.g. 5) on the remaining outer-training data:
            -- for HYPERPARAMETER TUNING, never touching the outer test fold
            FOR each hyperparameter candidate:
                run k_inner-fold CV, pick the best hyperparameter set
        RETRAIN on the FULL outer-training data using the best hyperparameters found
        EVALUATE on the outer TEST fold -> record this score
REPORT: mean and confidence interval (Phase 3 Lesson 4 bootstrap) of the k_outer outer-fold scores
```
**Why nested CV matters:** using the *same* CV loop both to tune hyperparameters AND to report final performance leaks information — the reported score is optimistically biased because the hyperparameters were specifically chosen to do well on the folds now being used to "evaluate" them. Nested CV keeps these two purposes cleanly separated — a subtle but real distinction that separates rigorous evaluation from an inflated performance claim.

---

## 5. Python Implementation

```python
"""model_evaluation_core.py — a rigorous evaluation toolkit"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, brier_score_loss,
    classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve
import xgboost as xgb


def evaluate_classifier(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": pr_auc,
        "brier_score": brier_score_loss(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def nested_cross_validation(X: np.ndarray, y: np.ndarray, param_grid: dict, k_outer: int = 5, k_inner: int = 5):
    outer_cv = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=0)
    outer_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=1)
        grid_search = GridSearchCV(
            xgb.XGBClassifier(eval_metric="logloss"), param_grid, cv=inner_cv, scoring="roc_auc"
        )
        grid_search.fit(X_train, y_train)   # hyperparameter tuning NEVER sees X_test

        best_model = grid_search.best_estimator_
        test_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        outer_scores.append(test_score)

    return np.array(outer_scores)


def calibration_report(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return prob_true, prob_pred   # compare these -- a well-calibrated model has prob_true ≈ prob_pred


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 5))
    y = rng.binomial(1, 0.05, n)   # deliberately IMBALANCED, like rare fraud/mortality events

    param_grid = {"max_depth": [2, 3], "n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
    scores = nested_cross_validation(X, y, param_grid, k_outer=3, k_inner=3)
    print(f"Nested CV ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
```

---

## 6. Build From Scratch

**Manual computation of precision/recall/ROC-AUC (to demystify the metric formulas):**
```python
import numpy as np

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

def precision_recall(y_true, y_pred):
    c = confusion_counts(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
    return precision, recall

def manual_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via the Mann-Whitney U interpretation: P(score_pos > score_neg)."""
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    # O(n_pos * n_neg) naive version -- illustrates the DEFINITION; use sklearn's O(n log n) version in practice
    wins = sum((p > n) for p in pos_scores for n in neg_scores)
    ties = sum((p == n) for p in pos_scores for n in neg_scores)
    return (wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores))
```
The naive $O(n_{\text{pos}} \times n_{\text{neg}})$ AUC computation directly demonstrates the Mann-Whitney U definition from Section 3, but real implementations (`sklearn.metrics.roc_auc_score`) use an $O(n\log n)$ rank-based computation — the same complexity-versus-clarity tradeoff seen throughout this curriculum (Phase 1 Lesson 4's sorting-based algorithmic improvements).

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `confusion_counts`/`precision_recall` | `sklearn.metrics.precision_score`/`recall_score`/`classification_report` |
| `manual_roc_auc` ($O(n^2)$) | `sklearn.metrics.roc_auc_score` ($O(n\log n)$, rank-based) |
| Manual nested CV loop | `sklearn.model_selection.GridSearchCV`/`RandomizedSearchCV` nested within an outer `cross_val_score` |
| Manual calibration binning | `sklearn.calibration.calibration_curve`, `CalibratedClassifierCV` (Platt scaling/isotonic regression for *fixing* poor calibration) |

---

## 8. Visual Explanations

**ROC curve vs. Precision-Recall curve (why PR-AUC matters more under imbalance):**
```
ROC space (TPR vs FPR):              PR space (Precision vs Recall):
1.0┤        ___----                  1.0┤‾‾‾‾--___
   │    ___/                             │          ‾‾--__
   │  _/                                 │                ‾‾--_
   │_/                                   │
0.0┼──────────────  FPR              0.0┼──────────────  Recall
   (looks "good" even for a mediocre    (reveals the TRUE difficulty
    classifier under heavy imbalance,     under imbalance -- precision
    since TN is huge and dominates FPR)   crashes as recall increases)
```

**Reliability diagram (calibration check):**
```
Observed frequency
  1.0┤                    ●  <- perfectly calibrated (on the diagonal)
     │                ●
     │            ●
     │        ●  ○  <- overconfident model: predicted 0.7, actually only ~0.4 positive
     │    ●
  0.0┼──────────────────── Predicted probability
     0.0                1.0
```

---

## 9. Practical Examples

**Simple:** compute precision, recall, and F1 for a binary classifier's predictions at a fixed threshold.
**Medium:** plot ROC and PR curves for the same model on an imbalanced dataset and discuss why they tell different stories.
**Real-world:** run nested cross-validation on your XGBoost mortality model, reporting a bootstrap confidence interval (Phase 3 Lesson 4) on ROC-AUC, and generate a calibration curve to check whether the model's predicted mortality probabilities can be trusted at face value for actuarial pricing — or whether post-hoc calibration (Platt scaling/isotonic regression) is needed first.

---

## 10. Real Industry Use Cases

- **Every production ML system with a business-critical decision threshold** (fraud detection, loan approval, medical diagnosis support): precision/recall tradeoffs are explicitly business decisions, not purely technical ones — a fraud team might deliberately accept lower precision for higher recall if missing fraud is far costlier than investigating false alarms.
- **Actuarial pricing models**: calibration is often *more* important than raw discrimination (AUC) — a pricing model's predicted probabilities directly become dollar amounts, so systematic miscalibration translates directly into systematic mispricing.
- **Clinical risk scores** (relevant to your cardiology work): both discrimination (can the model separate high/low risk patients) and calibration (does a "20% risk" prediction actually correspond to a 20% event rate) are independently, rigorously evaluated before clinical deployment, often via external validation cohorts.
- **Kaggle and ML research**: nested cross-validation (or a strict held-out test set used exactly once) is the gold standard for any credible claim of "our model achieves X performance."

---

## 11. Common Mistakes

- Reporting a single train/test split's metric without any confidence interval or cross-validation — presents a noisy point estimate as if it were a precise, stable fact.
- Using the test set repeatedly during model development/hyperparameter tuning, silently turning it into a second validation set and inflating the final reported number (exactly what nested CV, Section 4, is designed to prevent).
- Reporting only accuracy on an imbalanced dataset, hiding a model that may have near-zero recall on the class that actually matters.
- Confusing high AUC (good ranking) with good calibration (accurate probability values) — these are genuinely different properties, and a model can have one without the other.

---

## 12. Best Practices (2026)

- Always report a confidence interval (via bootstrap, Phase 3 Lesson 4, or cross-validation fold variance) alongside any point-estimate metric — a bare number without uncertainty context is an incomplete, easily-overinterpreted claim.
- Use nested cross-validation (or an equivalent strict train/validation/test separation) whenever hyperparameter tuning is involved, to avoid the reported-score-inflation trap.
- Choose your primary metric based on the actual business/clinical cost structure (precision vs. recall tradeoff), not by defaulting to accuracy or even AUC without thinking about what error costs actually matter.
- Explicitly check calibration (not just discrimination) for any model whose predicted probability will be used directly as a decision input (pricing, dosing, risk stratification) rather than just for ranking.

---

## 13. Exercises

**Easy:** Compute precision, recall, and F1 by hand for a small confusion matrix, and verify against `sklearn.metrics.classification_report`.
**Medium:** Generate ROC and PR curves for a classifier on a synthetically imbalanced dataset (5% positive rate) and explain the visual difference between them.
**Hard:** Implement nested cross-validation from scratch (without `GridSearchCV`) for a small hyperparameter grid, and empirically demonstrate the reported-score inflation that occurs when you skip the "nested" separation and instead tune hyperparameters using the same folds you report performance on.
**Mathematical:** Derive the Brier score decomposition (calibration + refinement components) and compute both components by hand for a small example dataset.
**Coding:** Implement `CalibratedClassifierCV`-style post-hoc calibration (Platt scaling: fit a logistic regression on top of the model's raw scores) from scratch, and verify it improves the calibration curve on a deliberately overconfident synthetic classifier.

---

## 14. Mini Project

Build a **complete rigorous evaluation report** for your actuarial mortality/risk model: implement nested cross-validation for honest hyperparameter tuning and performance estimation, report ROC-AUC and PR-AUC with bootstrap confidence intervals, generate and interpret a calibration curve (applying Platt scaling or isotonic regression if miscalibration is found), and write an explicit recommendation on the appropriate classification threshold given a stated business cost ratio between false positives and false negatives — treating this as a document that could genuinely support a real pricing/underwriting decision.

---

## 15. Interview Preparation

- Why can accuracy be a misleading metric, and what would you use instead for an imbalanced classification problem?
- Explain the difference between model discrimination (AUC) and calibration, and why both matter.
- What is nested cross-validation, and what problem does it solve that plain k-fold CV doesn't?
- System design: how would you decide on a classification threshold for a fraud-detection model given specific costs for false positives and false negatives?

---

## 16. Summary

Model evaluation is where a project's honesty is either preserved or quietly compromised: proper train/validation/test discipline (and nested cross-validation when tuning hyperparameters) prevents inflated performance claims, metric choice (precision/recall/PR-AUC over bare accuracy) must match the actual cost structure of the problem, and calibration — often under-emphasized relative to discrimination — is what determines whether a model's predicted probabilities can be trusted as literal numbers in pricing or clinical decisions. Every model built in this curriculum's remaining phases should be evaluated using exactly this lesson's rigor before any claim about its performance is taken seriously.

---

## 17. References

- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, Chapter 7 (Model Assessment and Selection)
- Murphy, A.H. — "A New Vector Partition of the Probability Score" (1973, the Brier score decomposition)
- Niculescu-Mizil & Caruana — "Predicting Good Probabilities With Supervised Learning" (2005, on model calibration)
- Saito & Rehmsmeier — "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets" (2015)
