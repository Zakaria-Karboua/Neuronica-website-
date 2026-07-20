# Phase 2 · Lesson 3 — Data Cleaning

> Prerequisite: NumPy, Pandas (Lessons 1–2)

---

## 1. Introduction

### What is data cleaning?
The systematic process of detecting and correcting (or appropriately removing) corrupt, inconsistent, duplicate, missing, or otherwise unusable data before it reaches modeling or analysis. This is not a preliminary chore — it is often the single largest time investment in any real-world data science project, and its quality is the ceiling on everything built afterward.

### Why does it exist?
"Garbage in, garbage out" is not a cliché in ML — it's a mathematical certainty. A model trained on systematically biased or corrupted data will faithfully learn and reproduce that bias/corruption, no matter how sophisticated the algorithm (XGBoost, transformers, anything). Data cleaning exists to make the "in" as trustworthy as possible before the "out" is ever evaluated.

### Historical background
Data cleaning as a formal discipline grew alongside data warehousing in the 1990s (ETL — Extract, Transform, Load) and matured further with the "data quality" movement in enterprise software. In 2026, data cleaning is increasingly LLM-assisted (using language models to detect anomalies, suggest standardizations, or flag inconsistent categorical labels) but the underlying statistical judgment about *what to do* with dirty data remains a human/domain-expert responsibility.

### Real-world motivation
Real datasets — insurance claims, clinical records, financial transactions — are never clean: inconsistent date formats, duplicate policyholder records under slightly different names, outlier claim amounts from data-entry typos, missing values that mean different things in different contexts. Your actuarial and cardiology work has almost certainly already run into every failure mode covered here.

---

## 2. Theory

### Categories of data quality problems
| Problem | Example | 
|---|---|
| **Missing data** | blank fields, sentinel values (`-999`, `"N/A"`) used inconsistently |
| **Duplicates** | exact row duplicates, or "fuzzy" duplicates (same person, different spelling) |
| **Inconsistent formatting** | dates as `01/02/2024` vs `2024-02-01`; categorical labels `"Male"` vs `"M"` vs `"male "` |
| **Outliers/invalid values** | negative ages, claim amounts of $0.01 or $10,000,000 from data entry errors |
| **Structural errors** | mismatched schemas across merged files, wrong dtypes (numbers stored as strings) |
| **Inconsistency across sources** | the same policyholder ID meaning different things in two different systems |

### Outlier detection concepts
- **Statistical outliers**: values far from the bulk of the distribution (IQR rule, z-score threshold).
- **Contextual outliers**: values normal in general but anomalous *given context* (e.g., a $50 claim is normal, but anomalous for a policyholder whose historical claims are all $5,000+).
- **Genuine extreme values vs. errors**: not all outliers are mistakes — a $2M claim might be a real catastrophic event, not a typo. This is precisely the domain Phase 3's Extreme Value Theory addresses formally; data cleaning must distinguish "this needs EVT-aware modeling" from "this is a data entry error."

---

## 3. Mathematical Foundations

### IQR (Interquartile Range) outlier rule
$$
\text{IQR} = Q_3 - Q_1, \quad \text{outlier if } x < Q_1 - 1.5 \cdot \text{IQR} \text{ or } x > Q_3 + 1.5 \cdot \text{IQR}
$$
This is a distribution-free (non-parametric) rule, robust to non-normal data — preferable to a naive $\pm 3\sigma$ rule when data is skewed (as claims/financial data typically is).

### Z-score outlier rule (assumes approximate normality)
$$
z = \frac{x - \mu}{\sigma}, \quad \text{flag if } |z| > k \text{ (commonly } k=3\text{)}
$$
**Caveat**: $\mu$ and $\sigma$ computed from data *containing* the outliers are themselves distorted by them (non-robust statistics) — a chicken-and-egg problem. **Robust alternative**: use the median and MAD (Median Absolute Deviation):
$$
\text{MAD} = \text{median}(|x_i - \text{median}(x)|), \quad z_{\text{robust}} = \frac{0.6745 (x - \text{median}(x))}{\text{MAD}}
$$
The constant $0.6745$ makes $z_{\text{robust}}$ comparable to a standard z-score under normality, while median/MAD remain far less sensitive to the very outliers you're trying to detect.

### Fuzzy matching for duplicate detection
Levenshtein (edit) distance between two strings $s_1, s_2$:
$$
\text{lev}(s_1, s_2) = \text{minimum number of single-character insertions, deletions, substitutions to transform } s_1 \to s_2
$$
Computed via dynamic programming (direct callback to Phase 1 Lesson 4's DP concept) in $O(|s_1| \cdot |s_2|)$ time, using the recurrence:
$$
D(i,j) = \min \begin{cases} D(i-1,j) + 1 \\ D(i,j-1) + 1 \\ D(i-1,j-1) + \mathbb{1}[s_1[i] \ne s_2[j]] \end{cases}
$$

---

## 4. Algorithm — Robust Outlier Detection Pipeline

```
GIVEN a numeric column potentially containing genuine extremes AND data errors:
1. Compute median and MAD (robust, unaffected by the outliers themselves)
2. Flag |robust z-score| > threshold as "candidate outlier"
3. FOR each candidate:
     a. Check domain plausibility (is a $10M claim physically impossible, or just rare?)
     b. Check for duplicate/data-entry signatures (e.g., exact digit transposition of a plausible value)
     c. Cross-reference other fields (does this record have other anomalies too?)
4. Classify each candidate as: DATA ERROR (correct or remove) vs GENUINE EXTREME (keep, flag for
   EVT-aware modeling downstream, Phase 3)
5. Document EVERY decision (never silently drop rows without a logged reason)
```
Step 5 is not optional in any serious pipeline — undocumented row-dropping is one of the most common sources of irreproducible analysis and hidden bias.

---

## 5. Python Implementation

```python
"""data_cleaning.py — a realistic, auditable cleaning pipeline"""
import pandas as pd
import numpy as np


def detect_outliers_robust(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Robust (median/MAD-based) outlier flagging — resistant to the outliers themselves."""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return pd.Series(False, index=series.index)   # avoid div-by-zero on degenerate data
    robust_z = 0.6745 * (series - median) / mad
    return robust_z.abs() > threshold


def standardize_categorical(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Normalizes inconsistent category labels ('M', 'male ', 'Male') to a canonical form."""
    cleaned = series.str.strip().str.lower()
    return cleaned.map(mapping).fillna(cleaned)   # unmapped values pass through, but visibly so


def audit_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Produces a transparent missing-data report — the FIRST step of any cleaning pipeline."""
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean().round(4),
        "dtype": df.dtypes,
    })
    return report.sort_values("missing_pct", ascending=False)


def clean_claims(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full pipeline returning BOTH the cleaned data AND a log of every decision made."""
    log = {}
    df = df.copy()

    # 1. Standardize categorical inconsistencies
    gender_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
    df["gender"] = standardize_categorical(df["gender"], gender_map)

    # 2. Flag (not silently drop) implausible values
    df["age_implausible"] = (df["age"] < 0) | (df["age"] > 120)
    log["age_implausible_count"] = int(df["age_implausible"].sum())

    # 3. Robust outlier detection on claim amounts
    df["amount_outlier"] = detect_outliers_robust(df["amount"])
    log["amount_outlier_count"] = int(df["amount_outlier"].sum())

    # 4. Exact duplicate detection (logged, not silently dropped)
    dup_mask = df.duplicated(subset=["policyholder_id", "claim_date", "amount"], keep="first")
    log["exact_duplicates_removed"] = int(dup_mask.sum())
    df = df[~dup_mask]

    return df, log


if __name__ == "__main__":
    raw = pd.DataFrame({
        "policyholder_id": [1, 2, 3, 3, 4],
        "claim_date": pd.to_datetime(["2024-01-01"] * 5),
        "amount": [1200, 900, 50000, 50000, -50],   # one genuine-looking outlier, one exact dup, one invalid
        "gender": ["M", "Female", "male ", "male ", "F"],
        "age": [45, 130, 60, 60, 30],                 # one implausible age
    })
    cleaned, cleaning_log = clean_claims(raw)
    print(audit_missingness(raw))
    print(cleaning_log)
```

---

## 6. Build From Scratch

**Levenshtein distance from scratch (for fuzzy duplicate detection):**
```python
def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    return dp[m][n]

def fuzzy_duplicate_pairs(names: list[str], max_distance: int = 2) -> list[tuple[str, str]]:
    """O(n^2) pairwise comparison — fine for thousands of names, needs blocking for millions (Section 12)."""
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if levenshtein(names[i].lower(), names[j].lower()) <= max_distance:
                pairs.append((names[i], names[j]))
    return pairs

print(fuzzy_duplicate_pairs(["John Smith", "Jon Smith", "Jane Doe"]))
# [('John Smith', 'Jon Smith')]  <- likely the same person, one-character typo
```
$O(m \cdot n)$ per pair (string lengths $m, n$) times $O(k^2)$ pairs for $k$ names — this quadratic blow-up is exactly why real record-linkage systems use **blocking** (only comparing records sharing some cheap key, e.g., same birth year) before applying expensive fuzzy matching, reducing effective comparisons dramatically.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `levenshtein` (pure Python DP) | `rapidfuzz`/`python-Levenshtein` (C-optimized, 10-100x faster) |
| Manual pairwise duplicate scan | `recordlinkage` / `dedupe` libraries (blocking + ML-based match scoring) |
| Manual robust z-score | `scipy.stats.median_abs_deviation`, `sklearn.preprocessing.RobustScaler` |
| Hand-logged cleaning decisions | Data quality frameworks (Great Expectations, `pandera`) with declarative, versioned validation rules |

---

## 8. Visual Explanations

**IQR outlier boundaries on a boxplot (conceptually, drawn textually):**
```
        Q1-1.5IQR   Q1    median    Q3   Q3+1.5IQR
             |       [======|======]        |
   outliers  o                                    o o   outliers
   (below)                                        (above, e.g. genuine catastrophic claims)
```

**Blocking before fuzzy matching (reduces O(n²) to tractable scale):**
```
WITHOUT blocking: compare ALL pairs -> O(n^2)
     n=1,000,000 -> ~5×10^11 comparisons (infeasible)

WITH blocking (e.g., group by birth_year first):
     buckets of ~1,000 records each -> O(n^2 / k) within each bucket
     -> feasible: only compare within same bucket, not across all million
```

---

## 9. Practical Examples

**Simple:** standardize a `region` column with inconsistent capitalization/whitespace.
**Medium:** detect and log exact-duplicate claims (same policyholder, date, amount) before any modeling.
**Real-world:** a robust outlier audit on a claims dataset distinguishing (a) a clear data-entry error (claim amount `999999999` next to a plausible `9999`, likely a stray extra digit) from (b) a genuine catastrophic claim requiring EVT-aware treatment downstream rather than removal — writing the *reasoning*, not just the code, for each classification.

---

## 10. Real Industry Use Cases

- **Insurance/actuarial data pipelines** (directly your domain): data cleaning decisions (which claims are errors vs. genuine tail events) materially affect reserve calculations and pricing — regulatory scrutiny often requires documented, auditable cleaning logic, not silent `dropna()` calls.
- **Clinical data systems** (relevant to your cardiology internship work): inconsistent units (mmHg vs kPa for blood pressure), inconsistent date formats across hospital systems, and genuine vs. erroneous extreme vital signs are constant, high-stakes cleaning problems.
- **Customer/entity resolution** (banking, e-commerce): fuzzy matching + blocking is the core technique behind deduplicating customer records across merged systems (e.g., after a company acquisition).
- **Great Expectations / Pandera**: production data-quality frameworks used at scale to declaratively encode "what does clean data look like" as versioned, testable rules (bridging directly to Phase 1 Lesson 9's testing philosophy, applied to data itself).

---

## 11. Common Mistakes

- Silently dropping rows/outliers without logging *why* — makes analysis irreproducible and hides potential bias.
- Using a non-robust z-score threshold on skewed financial/claims data, where the mean/std are themselves distorted by the very outliers being searched for.
- Treating all missing values the same regardless of mechanism (MCAR/MAR/MNAR, from Lesson 2) — sometimes "missing" itself is informative (e.g., a skipped lab test because a patient was too sick) and should become a feature, not be imputed away.
- Performing $O(n^2)$ fuzzy matching on large datasets without blocking, making the pipeline computationally infeasible at real scale.

---

## 12. Best Practices (2026)

- Use declarative data-validation frameworks (`pandera`, Great Expectations) to encode cleaning rules as testable, versioned schemas rather than ad hoc scripts — increasingly standard in production ML/data pipelines.
- Always separate "flag" from "act": first flag every anomaly with a boolean column, review/decide, *then* act (drop/impute/correct) — never combine detection and correction into one opaque step.
- Use blocking + `rapidfuzz` (not naive Levenshtein loops) for any fuzzy-matching task beyond toy scale.
- Keep an explicit, versioned data-cleaning changelog alongside your code — especially critical in regulated domains like insurance/healthcare where cleaning decisions may need to be justified to auditors/regulators years later.

---

## 13. Exercises

**Easy:** Write a function standardizing inconsistent date strings (`"01/02/2024"`, `"2024-02-01"`, `"Feb 1, 2024"`) into a single `datetime` format.
**Medium:** Implement robust (median/MAD) vs. non-robust (mean/std) outlier detection on a synthetic skewed dataset and show they disagree on which points are flagged.
**Hard:** Implement a blocking + fuzzy-matching duplicate-detection pipeline that scales to 100,000 records in reasonable time (blocking key: first letter of surname + birth year).
**Mathematical:** Prove that MAD-based robust z-scores remain bounded even as an increasing fraction of a dataset becomes extreme outliers, while mean/std-based z-scores do not (breakdown point argument).
**Coding:** Build a `pandera` schema encoding at least 5 validation rules for a claims dataset (non-negative amounts, valid date ranges, allowed categorical values, etc.) and run it as an automated check.

---

## 14. Mini Project

Build a **full auditable cleaning pipeline** for a realistic messy claims dataset (synthesize one with intentional issues: inconsistent categoricals, exact and fuzzy duplicates, implausible values, genuine extreme values, and MAR-style missingness): produce a written data-quality audit report (missingness by mechanism, outliers classified as error vs. genuine, duplicates found via blocking+fuzzy matching), a fully logged cleaning script (every dropped/modified row traceable to a specific rule), and a `pandera` schema validating the final cleaned output.

---

## 15. Interview Preparation

- How do you decide whether an outlier is a data error or a genuine extreme value?
- Why is MAD-based outlier detection often preferred over a simple z-score on financial/claims data?
- How would you detect duplicate customer records across two merged databases with inconsistent name formatting?
- System design: how would you build a data-cleaning pipeline that's auditable enough to satisfy a regulatory review of an insurance pricing model?

---

## 16. Summary

Data cleaning is where domain judgment meets statistical rigor: robust statistics (median/MAD) protect outlier detection from being corrupted by the very outliers under investigation; missingness mechanism reasoning (MCAR/MAR/MNAR) determines whether imputation helps or silently biases a model; and blocking-based fuzzy matching makes duplicate detection computationally tractable at real scale. Every cleaning decision should be logged and auditable — especially in regulated domains like insurance and healthcare, where "why was this row removed" must have a defensible answer.

---

## 17. References

- Rubin, D.B. — foundational missing-data mechanism theory (revisited from Lesson 2)
- Levenshtein, V. — original 1966 paper on edit distance
- Great Expectations / Pandera official documentation
- Christen, P. — *Data Matching* (the standard reference on record linkage/deduplication at scale)
