# Phase 2 · Lesson 5 — Exploratory Data Analysis (EDA)

> Prerequisite: NumPy, Pandas, Data Cleaning, Data Visualization (Lessons 1–4)

---

## 1. Introduction

### What is EDA?
A systematic, open-minded investigation of a dataset's structure, distributions, relationships, and anomalies — performed *before* any modeling commitment — using summary statistics, visualization (Lesson 4), and hypothesis generation. EDA is where you form and test informal hypotheses about the data ("does claim severity vary meaningfully by region?") that will later guide formal feature engineering (Lesson 6) and model selection (Phase 4).

### Why does it exist?
John Tukey formalized EDA in his 1977 book *Exploratory Data Analysis*, explicitly contrasting it with confirmatory (hypothesis-testing) analysis: EDA's job is *detective work* — finding what questions are worth asking — while confirmatory analysis's job is *rigorously answering* a pre-specified question. Skipping EDA and jumping straight to modeling is how subtle data problems (leakage, encoding errors, outlier-driven correlations) end up silently baked into a shipped model.

### Historical background
Tukey's EDA philosophy predates and directly inspired much of modern data science practice, emphasizing robust statistics, visual-first investigation, and comfort with ambiguity over premature statistical formality — a philosophy still central to how top-tier data scientists approach a new dataset in 2026, even with LLM-assisted tooling now able to automate much of the *mechanical* exploration.

### Real-world motivation
Before building your XGBoost mortality model or your Brent oil forecasting pipeline, you almost certainly did — or should have done — exactly this: looked at distributions, checked for leakage-prone features, examined correlations, and formed hypotheses about what would actually predict the outcome.

---

## 2. Theory

### The EDA workflow (a repeatable structure, not a checklist to rush through)
1. **Understand the data-generating context** — what does each row represent, how was it collected, what biases might exist in collection itself (survivorship bias, selection bias)?
2. **Univariate analysis** — distribution, central tendency, spread, skewness, and outliers (Lesson 3) for each variable individually.
3. **Bivariate/multivariate analysis** — relationships between variables (correlation, cross-tabulation, conditional distributions).
4. **Target-variable-focused analysis** (if a target exists) — how does each feature relate to the outcome you'll eventually predict? This is where potential **data leakage** first becomes visible (a feature suspiciously perfectly correlated with the target often means it was computed *using* the target, or *after* the outcome occurred).
5. **Hypothesis formation** — write down concrete, testable hypotheses ("region affects claim severity," "age has a non-linear relationship with mortality risk") to carry into Phase 4's formal modeling.

### Skewness and kurtosis (quantifying distribution shape)
$$
\text{Skewness} = \frac{E[(X - \mu)^3]}{\sigma^3}, \qquad \text{Kurtosis} = \frac{E[(X - \mu)^4]}{\sigma^4}
$$
Skewness > 0: right-tailed (typical of claims/income data — a few large values pull the tail right). Kurtosis > 3 ("leptokurtic," excess kurtosis > 0): heavier tails than normal — a direct early warning sign that Phase 3's Extreme Value Theory tools may be more appropriate than naive Gaussian assumptions for that variable.

### Correlation vs. causation, and correlation's own limitations
Pearson correlation $\rho$ only captures *linear* relationships:
$$
\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}
$$
A near-zero Pearson correlation does **not** imply no relationship (Anscombe's Quartet, Lesson 4, is the canonical proof) — a genuinely strong non-linear (e.g., quadratic, threshold) relationship can have $\rho \approx 0$. Spearman rank correlation (based on ranks, not raw values) captures monotonic-but-nonlinear relationships and is more robust to outliers.

---

## 3. Mathematical Foundations

### Data leakage as a formal concept
A feature $X_j$ **leaks** target information if $X_j$ is computed using information that would not be available at prediction time — formally, if
$$
X_j = g(Y, \dots) \text{ for some function } g \text{, even indirectly}
$$
Detecting leakage during EDA typically shows up as a suspiciously *too-good* correlation or a suspiciously *too-good* early model performance (Phase 4's evaluation lessons formalize this further) — EDA's job is to raise the flag before you waste a training run on a leaked feature.

### Multivariate structure: covariance matrix
For $p$ features, the $p \times p$ covariance matrix $\Sigma$ where $\Sigma_{ij} = \text{Cov}(X_i, X_j)$ summarizes pairwise linear relationships in one object — directly the input to later dimensionality reduction (PCA, Phase 4 Unsupervised Learning) and a natural EDA artifact (visualized as the Lesson 4 correlation heatmap).

### The Central Limit Theorem as an EDA sanity-check tool
Aggregated quantities (means of samples) tend toward normality regardless of the underlying distribution's shape, as $n \to \infty$:
$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1)
$$
This is *why* group-level means in EDA (e.g., "average claim by region") can look deceptively smooth/normal even when the underlying per-claim distribution is wildly skewed — an important EDA trap: always inspect the *raw* per-observation distribution, not just aggregated summaries, before concluding a variable is "well-behaved."

---

## 4. Algorithm — A Systematic EDA Procedure

```
GIVEN a new raw dataset with a target variable:
1. df.shape, df.dtypes, df.head() -> basic shape/type sanity check
2. audit_missingness(df) (Lesson 3) -> know what's missing and why before anything else
3. FOR each numeric column:
     compute mean, median, std, skewness, kurtosis, IQR-based outlier count
     plot histogram/KDE (log scale if skewed)
4. FOR each categorical column:
     value_counts(), check cardinality (a 500-category column needs different handling than a 3-category one)
5. Compute correlation matrix (Pearson AND Spearman) -> visualize as heatmap
6. FOR each feature vs. target:
     visualize relationship (scatter/boxplot depending on types)
     FLAG any suspiciously perfect relationship as a POTENTIAL LEAKAGE candidate for investigation
7. Write down 3-5 concrete hypotheses to test formally in Phase 4
```

---

## 5. Python Implementation

```python
"""eda_core.py — a repeatable, systematic EDA routine"""
import pandas as pd
import numpy as np
from scipy import stats


def univariate_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        rows.append({
            "column": col,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "skewness": stats.skew(series),
            "excess_kurtosis": stats.kurtosis(series),   # scipy returns EXCESS kurtosis (normal=0)
            "iqr_outlier_pct": (
                (series < series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25))) |
                (series > series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25)))
            ).mean(),
        })
    return pd.DataFrame(rows)


def leakage_scan(df: pd.DataFrame, target: str, threshold: float = 0.95) -> list[str]:
    """Flags features SUSPICIOUSLY highly correlated with the target for manual investigation.
    A high score here is NOT proof of leakage -- it's a prompt to go look."""
    numeric_df = df.select_dtypes(include=[np.number])
    if target not in numeric_df.columns:
        return []
    corr_with_target = numeric_df.corr()[target].drop(target).abs()
    suspicious = corr_with_target[corr_with_target > threshold].index.tolist()
    return suspicious


def bivariate_correlation_report(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson = df[numeric_cols].corr(method="pearson")
    spearman = df[numeric_cols].corr(method="spearman")
    # Large divergence between the two hints at a NON-LINEAR (but monotonic) relationship
    return pearson, spearman


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "age": rng.normal(45, 15, n).clip(18, 95),
        "claim_amount": rng.lognormal(7, 1.2, n),
        "region_risk_score": rng.normal(0, 1, n),
    })
    df["target_mortality_flag"] = (df["age"] > 65).astype(int)
    df["computed_after_the_fact"] = df["target_mortality_flag"] * 0.99 + rng.normal(0, 0.01, n)  # fake leak

    print(univariate_summary(df, ["age", "claim_amount", "region_risk_score"]))
    print("Potential leakage candidates:", leakage_scan(df, "target_mortality_flag"))
    pearson, spearman = bivariate_correlation_report(df, ["age", "claim_amount", "region_risk_score"])
    print(pearson)
```

---

## 6. Build From Scratch

**A minimal skewness/kurtosis calculator (to demystify `scipy.stats.skew/kurtosis`):**
```python
def skewness(data: list[float]) -> float:
    n = len(data)
    mean = sum(data) / n
    m2 = sum((x - mean) ** 2 for x in data) / n
    m3 = sum((x - mean) ** 3 for x in data) / n
    return m3 / (m2 ** 1.5)

def excess_kurtosis(data: list[float]) -> float:
    n = len(data)
    mean = sum(data) / n
    m2 = sum((x - mean) ** 2 for x in data) / n
    m4 = sum((x - mean) ** 4 for x in data) / n
    return (m4 / (m2 ** 2)) - 3   # subtract 3 so normal distribution has excess kurtosis = 0
```
Running this against a right-skewed synthetic dataset (e.g., lognormal-simulated claims) versus a symmetric one (normal-simulated ages) makes the abstract "skewness > 0 means right tail" statement into a directly observed, hand-computed number.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `univariate_summary` loop | `df.describe()`, `ydata-profiling` (automated full EDA reports), `sweetviz` |
| Manual skewness/kurtosis | `scipy.stats.skew`/`kurtosis` (handles edge cases, bias correction options) |
| Manual leakage scan | Still largely a *human judgment* task even with tooling — automated correlation flags are a starting point, not a substitute for domain reasoning |
| Manual correlation heatmap | `seaborn.heatmap(df.corr())`, interactive versions in `plotly` |

---

## 8. Visual Explanations

**EDA workflow as a funnel (broad to targeted):**
```
   Shape/dtypes/missingness  (broad sanity check)
             │
   Univariate distributions  (per-variable understanding)
             │
   Bivariate relationships   (pairwise structure, correlation)
             │
   Target-focused analysis   (feature-target relationships, LEAKAGE CHECK)
             │
   Concrete hypotheses       (narrow, testable statements -> feed into Phase 4)
```

**Skewness sign convention:**
```
Left-skewed (negative)        Symmetric (zero)         Right-skewed (positive)
      ▄▄██                        ▄██▄                      ██▄▄
    ▄████████                   ████████                  ████████▄▄
  ▄███████████▄                ██████████▄               ▄███████████▄▄▄▄
  long tail LEFT                 balanced                  long tail RIGHT
                                                        (typical: claims, income)
```

---

## 9. Practical Examples

**Simple:** compute skewness/kurtosis for a claims-amount column and interpret the sign and magnitude.
**Medium:** produce a correlation heatmap for a policyholder feature table and identify the two most correlated (non-target) features — a precursor to Phase 4's feature-selection/multicollinearity concerns.
**Real-world:** an EDA pass on your XGBoost mortality-model dataset explicitly checking for leakage: does any feature (e.g., "days_in_hospital_before_death") only exist or take a specific value *because* the outcome already occurred — exactly the kind of subtle leakage that silently inflates offline model performance while being useless (or actively misleading) at real prediction time.

---

## 10. Real Industry Use Cases

- **Every serious ML project at any AI lab**: EDA is the mandatory first phase before model architecture selection — skipping it is a well-known anti-pattern associated with silently leaked or biased models reaching production.
- **Clinical research** (your cardiology work): EDA routinely uncovers issues like measurement-unit inconsistencies across hospital systems, or informative missingness (sicker patients have more missing vitals) before any modeling begins.
- **Automated EDA tools** (`ydata-profiling`, `sweetviz`, and increasingly LLM-driven EDA assistants in 2026): automate the *mechanical* parts of Section 4's checklist, but leakage detection and hypothesis formation remain fundamentally human/domain-expert judgment calls.
- **A/B test analysis teams**: EDA on experiment data routinely catches sample-ratio mismatches or randomization bugs *before* anyone draws a causal conclusion from the experiment.

---

## 11. Common Mistakes

- Jumping straight to modeling without EDA — the single most common root cause of silently leaked, biased, or simply misunderstood models in real projects.
- Trusting Pearson correlation alone and missing genuine non-linear relationships (Anscombe's Quartet risk, Lesson 4) — always visualize, don't just compute a correlation coefficient.
- Treating a high feature-target correlation found during EDA as automatically real predictive signal, rather than a leakage red flag requiring investigation.
- Only looking at aggregated/grouped summaries (which, per the CLT, look deceptively smooth) instead of raw per-observation distributions, missing real skewness/outlier structure.

---

## 12. Best Practices (2026)

- Use automated EDA tools (`ydata-profiling`, `sweetviz`) to generate a fast *first-pass* report, but always follow up with manual, hypothesis-driven investigation — automation accelerates the mechanical checklist, not the judgment calls.
- Explicitly write out your leakage-check reasoning for any feature with suspiciously strong target correlation, as a permanent artifact (not just a mental note) — this becomes essential documentation if the model is later audited.
- Compute both Pearson and Spearman correlations by default; a large divergence between them is itself a useful signal (non-linear monotonic relationship) worth visualizing.
- Timebox EDA appropriately — Tukey's "detective work" mindset is valuable but EDA should converge toward concrete hypotheses, not become an open-ended activity with no output.

---

## 13. Exercises

**Easy:** Compute skewness and kurtosis for 3 different synthetic distributions (normal, lognormal, uniform) and verify the signs/magnitudes match theoretical expectations.
**Medium:** Given a dataset with a target variable, compute Pearson and Spearman correlations for all features and identify any feature where they diverge substantially — visualize that relationship to understand why.
**Hard:** Design and run a synthetic leakage-injection experiment: create a dataset, deliberately inject one leaked feature (computed partly from the target), and confirm your `leakage_scan` function (Section 5) successfully flags it while not flagging genuinely predictive, non-leaked features.
**Mathematical:** Prove, using a two-point example, that Pearson correlation can be exactly zero for a variable pair with a perfect (but non-monotonic, e.g., quadratic/symmetric) relationship.
**Coding:** Extend `univariate_summary` (Section 5) to also report the Freedman-Diaconis recommended bin count (Lesson 4) for each numeric column, linking the two lessons directly.

---

## 14. Mini Project

Perform a **complete EDA report** on a dataset of your choice (claims data, your Brent oil dataset, or the DZD exchange-rate data): systematically work through Section 4's procedure end to end, producing univariate summaries with skewness/kurtosis interpretation, a Pearson/Spearman correlation heatmap with divergence analysis, an explicit leakage scan with written reasoning for every flagged feature, and a final list of 3-5 concrete, testable hypotheses to carry into Phase 4's modeling lessons — treat this as a real analytical deliverable, not a throwaway notebook.

---

## 15. Interview Preparation

- Walk through your standard EDA process on a new dataset you've never seen before.
- What is data leakage, and describe a real or plausible example you'd look for during EDA.
- Why might Pearson correlation mislead you about a real relationship in your data, and what would you check instead/additionally?
- System design/ML-specific: how would you build automated leakage-detection checks into a data pipeline's CI (tying back to Phase 1 Lesson 9's testing philosophy)?

---

## 16. Summary

EDA is disciplined detective work: understanding a dataset's shape, distributions, and relationships *before* committing to a model, explicitly hunting for data leakage and informative missingness, and converging on concrete, testable hypotheses rather than open-ended exploration. Skewness/kurtosis quantify distribution shape objectively; Pearson vs. Spearman divergence flags non-linear structure; and a disciplined leakage scan is the single highest-leverage habit for avoiding a model that looks great offline and fails silently in production — exactly the failure mode formal model evaluation (Phase 4) is designed to catch if EDA misses it first.

---

## 17. References

- Tukey, J.W. — *Exploratory Data Analysis* (1977, the foundational text)
- Anscombe, F.J. — "Graphs in Statistical Analysis" (1973)
- `ydata-profiling` and `sweetviz` official documentation
- Kaufman, S. et al. — "Leakage in Data Mining: Formulation, Detection, and Avoidance" (2012) — the key academic reference on data leakage
