# Phase 2 · Lesson 2 — Pandas

> Prerequisite: NumPy (Lesson 1)

---

## 1. Introduction

### What is pandas?
Pandas is Python's primary library for labeled, heterogeneous tabular data manipulation — the `DataFrame` (2D, labeled rows/columns, mixed column types) and `Series` (1D labeled array) structures. Created by Wes McKinney in 2008 at AQR Capital Management to bring R's `data.frame` ergonomics into Python's ecosystem for quantitative finance work — a direct ancestor of the actuarial/financial data work you already do.

### Why does it exist?
NumPy arrays are fast but homogeneous and unlabeled — no natural way to say "this column is `claim_amount` (float), this one is `region` (string), joined by an `index`." Pandas adds a labeled, heterogeneous, SQL-like layer on top of (historically) NumPy arrays, letting you slice by label, group by key, join across tables, and handle missing data natively — precisely the operations real-world tabular data (claims, transactions, time series) demands.

### Historical background
Pandas 1.0 (2020) stabilized a mature API; pandas 2.0 (2023) introduced an optional Apache Arrow backend (`dtype_backend="pyarrow"`), improving string handling, nullable-type support, and interoperability with the broader 2026 data ecosystem (Polars, DuckDB, Arrow-native tools) — a significant architectural shift worth knowing about even if NumPy-backed pandas remains the default in many codebases.

### Real-world motivation
Every dataset you've worked with (claims files, exchange-rate time series, cardiology internship data) almost certainly passed through pandas at some point for cleaning, joining, and feature engineering before modeling.

---

## 2. Theory

### Core structures
- **`Series`**: a 1D labeled array (an index + values), effectively a NumPy array with a label attached to each position.
- **`DataFrame`**: a dict-like collection of `Series` sharing a common index — conceptually a table, but internally (in classic pandas) organized as column-oriented "blocks" of same-dtype data for efficiency.
- **`Index`**: the labels themselves — can be integers, strings, dates (`DatetimeIndex`), or hierarchical (`MultiIndex`).

### Key operations, conceptually
- **Selection**: `.loc[]` (label-based), `.iloc[]` (position-based) — a very common source of confusion; know which one you're using.
- **GroupBy**: split-apply-combine — partition rows by a key, apply a function to each partition, combine results — directly the same mathematical partitioning idea as SQL's `GROUP BY` (Phase 1 Lesson 7).
- **Merge/Join**: relational joins (`inner`, `left`, `right`, `outer`) — identical semantics to SQL joins, same underlying algorithms (hash join, sort-merge join).
- **Missing data**: pandas represents missingness via `NaN` (float) or, more precisely in modern pandas, dedicated nullable dtypes (`Int64`, `boolean`) supporting `pd.NA` — critical for correctly distinguishing "0" from "missing."

---

## 3. Mathematical Foundations

### GroupBy as a partition + function application
Given a dataset $D$ and a key function $k$, GroupBy partitions $D$ into equivalence classes:
$$
D = \bigsqcup_{v \in \text{range}(k)} \{ d \in D : k(d) = v \}
$$
then applies an aggregate $f$ to each class. This is set-theoretically identical to the `GROUP BY` operator from Phase 1's SQL lesson — pandas' GroupBy and SQL's GROUP BY are the same mathematical operation with different syntax.

### Complexity of core operations
| Operation | Complexity |
|---|---|
| `.loc[label]` (single label, sorted index) | $O(\log n)$ (binary search) or $O(1)$ (hash index) |
| `.groupby(key).agg(fn)` | $O(n)$ to partition (hashing) + cost of `fn` per group |
| `merge` (join) | $O(n + m)$ hash join (default for equality joins), or $O(n \log n)$ sort-merge — same theory as Phase 1 SQL Lesson 4 |
| `sort_values` | $O(n \log n)$ |

### Missing-data statistics
When dropping/imputing missing values, you are implicitly assuming a missingness mechanism:
- **MCAR** (Missing Completely At Random): missingness independent of any variable — dropping rows introduces no bias.
- **MAR** (Missing At Random): missingness depends on *observed* variables — can be corrected via conditional imputation.
- **MNAR** (Missing Not At Random): missingness depends on the *unobserved* value itself (e.g., very sick patients failing to report vitals) — no amount of clever imputation fully fixes this without external information.

This isn't just a technical footnote — it directly affects whether `df.dropna()` silently introduces bias into a mortality model, a real and serious risk in clinical/actuarial data.

---

## 4. Algorithm — How `.groupby().agg()` Executes

```
GIVEN df.groupby("region")["claim_amount"].mean():
1. HASH each row's "region" value -> bucket rows into groups (O(n))
2. FOR each group:
     apply the aggregate function (mean) to that group's claim_amount values
3. COMBINE results into a new Series/DataFrame indexed by the group keys
Total: O(n) to partition + O(n) total work across all aggregate computations = O(n)
```
This mirrors SQL's hash-based `GROUP BY` execution (Phase 1 Lesson 7) almost exactly — recognizing this means you can predict pandas' GroupBy performance characteristics using the same mental model you already have for SQL.

---

## 5. Python Implementation

```python
"""pandas_core.py — realistic patterns for actuarial/claims data work"""
import pandas as pd
import numpy as np

claims = pd.DataFrame({
    "policyholder_id": [1, 1, 2, 3, 3, 3],
    "claim_date": pd.to_datetime(
        ["2024-01-10", "2024-06-15", "2024-03-01", "2023-11-20", "2024-02-14", "2024-05-30"]
    ),
    "amount": [1200.0, 800.0, 3400.0, np.nan, 500.0, 950.0],
    "region": ["North", "North", "South", "East", "East", "East"],
})

policyholders = pd.DataFrame({
    "policyholder_id": [1, 2, 3, 4],
    "birth_date": pd.to_datetime(["1970-05-01", "1955-08-20", "1990-12-01", "1985-01-01"]),
})

# --- Handling missing data explicitly (never silently) ---
missing_pct = claims["amount"].isna().mean()
print(f"Missing claim amounts: {missing_pct:.1%}")
claims["amount"] = claims["amount"].fillna(claims.groupby("region")["amount"].transform("median"))
# NOTE: median imputed PER REGION (conditional imputation), not a single global median —
# a meaningfully better MAR-aware default than blanket imputation.

# --- Merge (join), mirroring SQL semantics from Phase 1 ---
merged = claims.merge(policyholders, on="policyholder_id", how="left")

# --- Feature engineering: age at claim time, vectorized ---
merged["age_at_claim"] = (merged["claim_date"] - merged["birth_date"]).dt.days // 365

# --- GroupBy + multiple aggregations at once ---
summary = merged.groupby("region").agg(
    claim_count=("amount", "count"),
    avg_amount=("amount", "mean"),
    total_amount=("amount", "sum"),
).reset_index()

# --- Window-like operation: trailing claims per policyholder (mirrors SQL window functions) ---
merged = merged.sort_values(["policyholder_id", "claim_date"])
merged["cumulative_claims"] = merged.groupby("policyholder_id")["amount"].cumsum()

print(summary)
print(merged[["policyholder_id", "claim_date", "amount", "cumulative_claims"]])
```

**Notes:** `.groupby(...).transform("median")` returns a Series aligned to the *original* index (not collapsed) — the correct tool for imputation-by-group, versus `.agg()` which collapses to one row per group.

---

## 6. Build From Scratch

**A minimal GroupBy implementation (mirrors pandas' split-apply-combine):**
```python
def simple_groupby_mean(records: list[dict], key: str, value: str) -> dict[str, float]:
    """Reimplements the essence of df.groupby(key)[value].mean()."""
    groups: dict = {}
    for row in records:
        groups.setdefault(row[key], []).append(row[value])
    return {k: sum(v) / len(v) for k, v in groups.items()}

# Mirrors: claims_records -> pd.DataFrame(...).groupby("region")["amount"].mean()
records = [
    {"region": "North", "amount": 1200}, {"region": "North", "amount": 800},
    {"region": "South", "amount": 3400},
]
print(simple_groupby_mean(records, "region", "amount"))
```
This is exactly the hash-based partition-then-aggregate algorithm from Section 4, made explicit — and it's the *same* hash-table mechanism from Phase 1 Lesson 4 (DSA) and Lesson 7 (SQL), now appearing for the third time in three different contexts — a good sign you've internalized a genuinely transferable idea rather than three unrelated facts.

---

## 7. Library Implementation (Comparison)

| From scratch | Real pandas |
|---|---|
| `simple_groupby_mean` (pure Python dict) | `.groupby().mean()` — implemented in Cython/C, handles multi-key grouping, many aggregate functions, NaN-awareness |
| Manual missing-value dict scan | `.isna()`, `.fillna()`, dedicated nullable dtypes (`pd.NA`) |
| No indexing | full label/positional indexing (`.loc`/`.iloc`), hierarchical `MultiIndex` |
| No join optimization | `.merge()` dispatches to optimized hash/sort-merge joins, same theory as SQL |

---

## 8. Visual Explanations

**Split-apply-combine:**
```
   df.groupby("region")["amount"].mean()

SPLIT:                      APPLY (mean):        COMBINE:
North: [1200, 800]     ->    1000.0        \
South: [3400]          ->    3400.0         >--> Series indexed by region
East:  [500, 950]      ->     725.0        /
```

**`.loc` vs `.iloc`:**
```
        col_a  col_b
idx_x     1      10
idx_y     2      20
idx_z     3      30

df.loc["idx_y"]   -> row by LABEL   -> [2, 20]
df.iloc[1]        -> row by POSITION -> [2, 20]  (same result here, but NOT if index isn't 0..n sorted integers!)
```

---

## 9. Practical Examples

**Simple:** load a CSV, check `.info()` and `.describe()`, identify columns with missing values.
**Medium:** merge a claims table with a policyholders table and compute claim frequency per policyholder-year.
**Real-world:** build a full policyholder-level feature table (age, tenure, region, trailing-12-month claim count/amount, missing-value flags) directly analogous to the SQL feature-engineering mini-project from Phase 1, but now in pandas — useful for comparing which tool fits which stage of a real pipeline (often: SQL for large-scale extraction from a warehouse, pandas for the final flexible feature-engineering pass).

---

## 10. Real Industry Use Cases

- **Every tabular ML pipeline** (fraud detection, credit scoring, actuarial pricing) uses pandas as the final feature-engineering layer before modeling, even when raw extraction happens in SQL/Spark.
- **Quant finance** (McKinney's original use case): time-series resampling, rolling-window statistics (`df.rolling()`), and portfolio analytics.
- **Polars/DuckDB (2026 ecosystem)**: increasingly popular as faster, more memory-efficient alternatives/complements to pandas for larger-than-RAM tabular data, but expose highly pandas-influenced APIs — understanding pandas transfers directly.
- **A/B testing and BI reporting** (nearly every tech company): pandas is the standard tool for post-SQL-extraction statistical analysis and visualization prep (Lesson 4).

---

## 11. Common Mistakes

- **`SettingWithCopyWarning`**: modifying a DataFrame slice without realizing it may be a view vs. copy (mirrors NumPy's view/copy distinction from Lesson 1) — always use `.loc[row_indexer, col_indexer] = value` for explicit, unambiguous assignment.
- Blanket `df.dropna()` without checking the missingness mechanism (Section 3) — can silently bias a mortality/risk model if missingness correlates with the outcome (MNAR).
- Using `.apply()` with a Python-level row loop when a vectorized operation exists — this reintroduces exactly the performance problem NumPy vectorization (Lesson 1) was designed to solve; `.apply(lambda row: row.a + row.b, axis=1)` should almost always just be `df.a + df.b`.
- Ignoring dtypes after merges/concatenations — an `int` column silently becoming `float` after a merge introduces `NaN`s (which are inherently float in classic pandas), sometimes masking real data-quality issues.

---

## 12. Best Practices (2026)

- Use vectorized pandas/NumPy operations first; reach for `.apply()` only when no vectorized equivalent exists, and treat it as a performance smell to revisit.
- Explicitly document/handle missingness mechanism assumptions (MCAR/MAR/MNAR) in feature-engineering code comments — not just "drop NaNs" silently.
- Consider `pandas.options.mode.copy_on_write = True` (the direction pandas 2.x+ defaults are moving) to eliminate the entire class of view/copy ambiguity bugs.
- For genuinely large (tens of millions+ rows) or performance-critical tabular workloads, evaluate Polars or DuckDB as 2026-appropriate alternatives/complements — but pandas remains the right default for typical feature-engineering scale and ecosystem compatibility (scikit-learn, most notebooks).

---

## 13. Exercises

**Easy:** Load a claims CSV, report the percentage of missing values per column, and the dtype of each column.
**Medium:** Compute, per policyholder, the number of days since their most recent claim as of a fixed reference date.
**Hard:** Implement a rolling 6-month claim-count feature per policyholder using `.groupby().rolling()`, correctly handling policyholders with irregular claim timing (non-uniform date gaps).
**Mathematical:** Given a dataset with missingness mechanism MAR (missingness depends on `region`), prove via a small simulated example that per-region imputation reduces bias compared to blanket-median imputation.
**Coding:** Extend Section 6's `simple_groupby_mean` to support multiple aggregate functions (mean, count, sum) computed in a single pass over the data.

---

## 14. Mini Project

Build a full **claims feature-engineering notebook**: load raw claims + policyholders CSVs, systematically audit and document missingness (mechanism assumption per column, not just percentage), perform appropriate group-aware imputation, engineer policyholder-level features (age, tenure, trailing-12-month claim frequency/severity, region, a data-quality flag column marking imputed values), and output a clean, model-ready feature table — explicitly comparing where you'd have used SQL (Phase 1) vs. pandas for each step and why.

---

## 15. Interview Preparation

- Explain the difference between `.loc` and `.iloc`, and a scenario where confusing them causes a silent bug.
- What is `SettingWithCopyWarning`, and how do you avoid it?
- Explain MCAR/MAR/MNAR and how each should influence your missing-data handling strategy.
- System design: how would you decide between pandas, Polars, and SQL/a data warehouse for a given tabular data task at a specific scale?

---

## 16. Summary

Pandas brings SQL-like relational operations (joins, group-by, filtering) and NumPy-like vectorized numerical operations together into one labeled, heterogeneous-column tabular structure — the default workhorse for feature engineering on any dataset that fits comfortably in memory. The split-apply-combine mental model, honest handling of missing-data mechanisms, and disciplined avoidance of row-wise Python loops are what separate feature-engineering code that scales and stays correct from code that silently breaks or crawls on real production data volumes.

---

## 17. References

- McKinney, W. — *Python for Data Analysis* (3rd ed., written by pandas' creator)
- Official pandas documentation, especially the "Merge, join, concatenate" and "GroupBy" user guides
- Rubin, D.B. — foundational papers on missing-data mechanisms (MCAR/MAR/MNAR)
- Polars documentation (for contrast/comparison as a 2026-relevant alternative)
