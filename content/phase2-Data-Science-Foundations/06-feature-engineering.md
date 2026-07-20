# Phase 2 · Lesson 6 — Feature Engineering

> Prerequisite: NumPy, Pandas, Data Cleaning, Data Visualization, EDA (Lessons 1–5) — this lesson closes out Phase 2

---

## 1. Introduction

### What is feature engineering?
The process of transforming raw data into representations (features) that make the underlying signal easier for a model to learn — encoding categoricals, scaling numerics, constructing interaction/derived features, handling temporal structure, and reducing dimensionality where appropriate. Despite the rise of deep learning's automatic representation learning (Phase 5-6), feature engineering remains dominant and often decisive for tabular data — precisely the data type in your actuarial/mortality-modeling work.

### Why does it exist?
Most classical ML algorithms (linear models, tree ensembles like XGBoost) cannot automatically discover useful non-linear transformations, interactions, or domain-specific structure the way a deep network's hidden layers can — they rely on the *input representation itself* already exposing the signal reasonably well. A famous aphorism (attributed to various ML practitioners): "applied machine learning is basically feature engineering" — still largely true for tabular/structured data even in 2026.

### Historical background
Feature engineering has always been central to statistics and econometrics (transformations, interaction terms, polynomial features have centuries of precedent in regression modeling) but became a distinct, systematized ML discipline as competitive ML (Kaggle, from 2010 onward) repeatedly demonstrated that feature engineering, not algorithm choice, was the dominant driver of leaderboard performance on tabular problems.

### Real-world motivation
The gap between a mediocre and an excellent XGBoost mortality model is overwhelmingly a feature-engineering gap, not a hyperparameter-tuning gap — the same is true of your actuarial pricing and forecasting work.

---

## 2. Theory

### Categorical encoding strategies
| Method | When to use | Risk |
|---|---|---|
| One-hot encoding | low-cardinality categoricals (<~15 categories), linear models | dimensionality explosion for high-cardinality features |
| Ordinal encoding | genuinely ordered categories (e.g., risk tiers: low/med/high) | imposes false ordering if used on unordered categories |
| Target encoding (mean encoding) | high-cardinality categoricals, tree models | **leakage risk** if not done with proper cross-validation folds |
| Frequency encoding | high-cardinality, when frequency itself is informative | loses category identity information |
| Embeddings (learned) | very high-cardinality categoricals in deep learning contexts | requires enough data to learn meaningful embeddings (Phase 6 territory) |

### Numeric transformations
- **Scaling** (min-max, z-score, robust/MAD-based): required for distance-based or gradient-based models (k-NN, linear regression, neural nets); irrelevant for tree-based models (splits are scale-invariant).
- **Log/Box-Cox transforms**: reduce right-skew (common for financial/claims data, directly following Lesson 5's skewness diagnosis) and can linearize otherwise non-linear relationships with the target.
- **Binning/discretization**: converts continuous variables into categorical bins — sometimes improves tree-model interpretability or captures known non-linear thresholds (e.g., regulatory age bands in insurance).

### Interaction and derived features
Domain knowledge often suggests specific interactions a model would otherwise need enormous data to discover on its own — e.g., `claim_amount / policy_tenure` (severity rate) or `age × smoker_status` (a classic actuarial interaction with well-documented non-additive mortality effects).

### Temporal feature engineering
For any date/time column: extract cyclical components (day-of-week, month, is-holiday), lag features (value N periods ago), and rolling-window aggregates (trailing mean/count) — directly reusing pandas' `.groupby().rolling()`/`.shift()` machinery from Lesson 2, formalized further in Phase 4's Time Series lesson.

---

## 3. Mathematical Foundations

### Target encoding and its leakage mechanism (formalized)
Naive target encoding replaces category $c$ with:
$$
\hat{\mu}_c = \frac{1}{n_c}\sum_{i: x_i = c} y_i
$$
computed using the *entire* training set including row $i$ itself — this directly encodes $y_i$ into $\hat{\mu}_c$ for row $i$, a textbook leakage case (Lesson 5). The correct approach uses **out-of-fold encoding**: for each row, compute $\hat{\mu}_c$ using only *other folds'* data, or apply additive smoothing toward the global mean:
$$
\hat{\mu}_c^{\text{smoothed}} = \frac{n_c \bar{y}_c + m \bar{y}}{n_c + m}
$$
where $m$ is a smoothing strength parameter and $\bar{y}$ the global target mean — shrinking rare categories' estimates toward the global mean, reducing both leakage risk and high-variance overfitting on small categories.

### Box-Cox transformation (generalizing log transforms)
$$
y^{(\lambda)} = \begin{cases} \dfrac{y^\lambda - 1}{\lambda} & \lambda \ne 0 \\ \ln(y) & \lambda = 0 \end{cases}
$$
$\lambda$ is typically chosen via maximum likelihood to best normalize the distribution — the ordinary log transform ($\lambda = 0$) is simply one special case of this more general family.

### Curse of dimensionality (why one-hot encoding high-cardinality features hurts)
As feature dimensionality $p$ grows, the volume of the space grows exponentially ($\propto r^p$ for a hypercube of side $r$), meaning fixed-size training data becomes exponentially sparser in that space — distance-based methods (k-NN, clustering) degrade sharply, and even tree models face diminishing returns per additional sparse one-hot column. This is the formal justification for preferring target/frequency encoding over one-hot encoding once cardinality exceeds roughly a few dozen categories.

---

## 4. Algorithm — Leakage-Safe Target Encoding (K-Fold)

```
GIVEN training data with categorical column C and target Y, K folds:
1. Split training data into K folds
2. FOR each fold k:
     a. Compute mean(Y) per category, using ONLY data OUTSIDE fold k
     b. Apply that mapping to encode column C WITHIN fold k
3. For the TEST/production set (never part of any fold):
     compute the final mapping using the ENTIRE training set (all folds now allowed,
     since test data was never used to compute it)
4. Apply additive smoothing (formula in Section 3) throughout, especially for rare categories
```
This K-fold procedure is precisely what prevents target encoding from leaking each row's own target value into its own encoded feature — a foundational technique you'll need again in Phase 4's cross-validation lessons.

---

## 5. Python Implementation

```python
"""feature_engineering.py — leakage-safe, production-grade patterns"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def kfold_target_encode(
    df: pd.DataFrame, cat_col: str, target_col: str, n_splits: int = 5, smoothing: float = 10.0
) -> pd.Series:
    """Leakage-safe target encoding via out-of-fold computation + additive smoothing."""
    global_mean = df[target_col].mean()
    encoded = pd.Series(index=df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        stats = train_fold.groupby(cat_col)[target_col].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        encoded.iloc[val_idx] = df.iloc[val_idx][cat_col].map(smoothed).fillna(global_mean)

    return encoded


def engineer_actuarial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric transforms: log-transform right-skewed severity (Lesson 5 diagnosis -> action)
    df["log_claim_amount"] = np.log1p(df["claim_amount"])   # log1p handles zero claims safely

    # Domain-informed interaction feature
    df["severity_rate"] = df["claim_amount"] / df["policy_tenure_years"].clip(lower=0.1)

    # Temporal features
    df["claim_month"] = df["claim_date"].dt.month
    df["claim_month_sin"] = np.sin(2 * np.pi * df["claim_month"] / 12)   # cyclical encoding
    df["claim_month_cos"] = np.cos(2 * np.pi * df["claim_month"] / 12)   # avoids false Dec->Jan discontinuity

    # Rolling/lag features per policyholder
    df = df.sort_values(["policyholder_id", "claim_date"])
    df["trailing_3_claim_count"] = (
        df.groupby("policyholder_id")["claim_amount"]
        .transform(lambda s: s.rolling(3, min_periods=1).count())
    )

    # Leakage-safe target encoding for high-cardinality region/occupation codes
    df["region_encoded"] = kfold_target_encode(df, "region", "claim_amount")

    return df
```

**Notes:** `claim_month_sin`/`claim_month_cos` (cyclical encoding) is a specifically important pattern — naively encoding month as 1-12 tells a model December (12) and January (1) are maximally *far apart* numerically, when they're actually adjacent in the annual cycle; the sin/cos pair correctly represents this circular structure.

---

## 6. Build From Scratch

**A minimal, from-scratch smoothed target encoder (mirrors the leakage-safe logic explicitly):**
```python
def smoothed_target_mean(values: list[float], smoothing: float, global_mean: float) -> float:
    n = len(values)
    if n == 0:
        return global_mean
    local_mean = sum(values) / n
    return (n * local_mean + smoothing * global_mean) / (n + smoothing)

def manual_kfold_encode(rows: list[dict], cat_key: str, target_key: str, k: int = 5, smoothing: float = 10.0):
    import random
    random.seed(42)
    global_mean = sum(r[target_key] for r in rows) / len(rows)
    indices = list(range(len(rows)))
    random.shuffle(indices)
    folds = [indices[i::k] for i in range(k)]   # simple round-robin fold assignment

    encoded = [None] * len(rows)
    for fold_i in range(k):
        val_idx = set(folds[fold_i])
        train_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]
        category_values: dict = {}
        for r in train_rows:
            category_values.setdefault(r[cat_key], []).append(r[target_key])
        for i in val_idx:
            cat = rows[i][cat_key]
            encoded[i] = smoothed_target_mean(category_values.get(cat, []), smoothing, global_mean)
    return encoded
```
Building this by hand once makes the K-fold-out-of-fold logic (Section 4's algorithm) fully concrete and mechanically verifiable, before trusting `sklearn`/`category_encoders`' more polished, production-optimized versions.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `manual_kfold_encode` | `category_encoders.TargetEncoder` (handles many edge cases, integrates with sklearn Pipelines) |
| Manual sin/cos cyclical encoding | Same approach used directly in production — this one genuinely IS the standard technique, not a simplified stand-in |
| Manual rolling/lag features | `pandas.DataFrame.rolling`/`.shift()`, or dedicated feature-store rolling-aggregate engines (Feast) at production scale |
| Ad hoc feature scripts | **Feature stores** (Feast, Tecton) — versioned, reusable, leakage-audited feature definitions shared across training and serving, critical for avoiding train/serve skew |

---

## 8. Visual Explanations

**Cyclical encoding fixing the "December-January" discontinuity:**
```
Naive numeric month:           Cyclical (sin, cos) encoding:
  Dec=12 ────X──── Jan=1          Dec and Jan sit ADJACENT
  (numerically far apart,          on a circle (sin/cos pair),
   despite being adjacent            correctly close in encoded space
   in the actual calendar)
                    Jan
                 Dec  ● 1
                   ●     
              Nov ●   ● Feb
                (circular, not linear)
```

**K-fold out-of-fold target encoding (leakage-safe):**
```
Fold 1  Fold 2  Fold 3  Fold 4  Fold 5
  │       │       │       │       │
  ▼       (train on folds 2,3,4,5 -> encode fold 1)
[ENC]    │       │       │       │
         ▼       (train on folds 1,3,4,5 -> encode fold 2)
        [ENC]    │       │       │
                  ... (repeat for every fold) ...
Each row's encoding is computed WITHOUT ever seeing its own target value.
```

---

## 9. Practical Examples

**Simple:** one-hot encode a low-cardinality `gender` column and z-score scale an `age` column.
**Medium:** apply cyclical encoding to a `day_of_week` feature and compare model performance against naive ordinal encoding on a synthetic dataset with a genuinely cyclical target pattern.
**Real-world:** engineer a complete actuarial feature set for your mortality model — log-transformed claim severity, leakage-safe target-encoded occupation/region codes, trailing rolling claim counts, and a domain-informed `age × smoker_status` interaction term — directly extending your existing XGBoost project with more rigorous, leakage-audited feature construction.

---

## 10. Real Industry Use Cases

- **Kaggle competition history**: repeatedly demonstrates feature engineering (target encoding, interaction terms, domain-specific derived features) as the dominant differentiator on tabular leaderboards, often more than model architecture choice.
- **Insurance pricing models** (directly your domain): actuarial feature engineering routinely includes interaction terms with well-established non-additive effects (age × smoking status, region × vehicle type) informed by decades of actuarial literature, not purely data-driven discovery.
- **Feature stores at scale** (Uber's Michelangelo, Airbnb's Zipline, open-source Feast): exist specifically to prevent train/serve skew — ensuring the exact same feature computation logic (including leakage-safe encodings) runs identically at training time and real-time serving.
- **Fraud detection systems**: heavily rely on engineered velocity/frequency features (e.g., "number of transactions in the last hour") computed via exactly the rolling-window techniques in this lesson.

---

## 11. Common Mistakes

- Computing target encoding on the *full* dataset before any train/test split or cross-validation folding — a textbook leakage bug that inflates offline validation metrics while destroying real-world performance.
- One-hot encoding very high-cardinality categoricals (e.g., a `zip_code` with 40,000 unique values) — causes the curse of dimensionality (Section 3) and often degrades tree-model performance versus target/frequency encoding.
- Naively encoding cyclical variables (month, hour, day-of-week) as plain integers, silently telling the model false "distance" relationships across the year/day boundary.
- Engineering features using information only available *after* the prediction moment (e.g., "total claims this year" as a feature when predicting whether *this specific* claim will occur) — feature engineering's own version of Lesson 5's leakage problem, requiring careful "as-of" time-awareness in feature computation.

---

## 12. Best Practices (2026)

- Always compute target/leave-one-out encodings inside proper cross-validation folds (Section 4), never on the full dataset before splitting.
- Prefer feature-store-style versioned feature definitions (even a lightweight in-house convention) over ad hoc notebook feature scripts, specifically to guarantee training/serving consistency.
- Default to cyclical (sin/cos) encoding for any genuinely periodic feature (time of day, day of week, month) rather than naive ordinal encoding.
- Document the "as-of" time assumption for every engineered feature explicitly (what information was available at the moment this feature would be computed in production) — the single most effective habit for preventing feature-level data leakage.

---

## 13. Exercises

**Easy:** One-hot encode a categorical column and z-score scale two numeric columns using `sklearn.preprocessing`.
**Medium:** Implement cyclical (sin/cos) encoding for an `hour_of_day` feature and demonstrate, via a nearest-neighbor distance calculation, that 23:00 and 01:00 are now correctly close together.
**Hard:** Implement K-fold out-of-fold target encoding from scratch (Section 6), then empirically demonstrate the leakage that occurs when you skip the fold-splitting step (compare validation AUC/RMSE with vs. without proper folding on a synthetic dataset).
**Mathematical:** Derive how additive smoothing (Section 3's formula) behaves in the limit as $n_c \to 0$ (rare category) and as $n_c \to \infty$ (common category), confirming it correctly reduces to the global mean and the raw category mean respectively.
**Coding:** Build a small `FeatureEngineer` class (applying Phase 1's SOLID/Strategy principles) with swappable encoding strategies (one-hot, target, frequency) selectable by categorical column cardinality automatically.

---

## 14. Mini Project

Build a **complete, leakage-audited feature engineering pipeline** for your actuarial/mortality dataset: implement K-fold target encoding for high-cardinality categoricals, cyclical encoding for any temporal features, domain-informed interaction terms (with a written justification citing actuarial reasoning, not just data-driven discovery), and an explicit "as-of" documentation comment for every feature stating what information would genuinely be available at prediction time — then validate, using a held-out test set, that your leakage-safe target encoding doesn't show the inflated (too-good) performance that a naive full-dataset target encoding would.

---

## 15. Interview Preparation

- Explain target encoding and the specific mechanism by which it can leak target information if done incorrectly.
- Why might one-hot encoding hurt model performance on a very high-cardinality categorical feature?
- How would you encode a cyclical feature like hour-of-day, and why does naive ordinal encoding fail here?
- System design: how would you design a feature-engineering pipeline that guarantees identical feature computation between training and real-time production serving?

---

## 16. Summary

Feature engineering is where domain knowledge, statistical transformation, and leakage-aware engineering discipline combine to give classical ML models (especially tree ensembles like XGBoost) the representational leverage that deep learning gets automatically from hidden layers. K-fold out-of-fold target encoding, cyclical encoding for periodic variables, and rigorously "as-of"-documented derived features are the concrete techniques that separate a feature set that merely looks good in offline validation from one that reliably performs the same way in production — closing out Phase 2's foundation before Phase 3 formalizes the underlying mathematics and Phase 4 builds the models these features feed into.

---

## 17. References

- Micci-Barreca, D. — "A Preprocessing Scheme for High-Cardinality Categorical Attributes" (2001, the original target-encoding paper)
- Zheng, A. & Casari, A. — *Feature Engineering for Machine Learning* (O'Reilly)
- `category_encoders` library documentation
- Kaggle competition winner write-ups (Kaggle's official blog/forums) — real-world evidence of feature engineering's outsized impact on tabular ML performance
