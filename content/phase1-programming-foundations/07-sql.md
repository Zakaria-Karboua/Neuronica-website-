# Phase 1 · Lesson 7 — SQL

> Prerequisite: Lessons 1–6

---

## 1. Introduction

### What is SQL?
Structured Query Language — a declarative language for defining, querying, and manipulating data in relational database management systems (RDBMS: PostgreSQL, MySQL, SQLite, and cloud warehouses like Snowflake/BigQuery which use SQL dialects). "Declarative" means you state *what* result you want; the query planner decides *how* to compute it efficiently — a fundamentally different paradigm from the imperative Python you've written so far.

### Why does it exist?
Edgar F. Codd's 1970 paper "A Relational Model of Data for Large Shared Data Banks" proposed representing data as tables (relations) governed by set theory and predicate logic, letting users ask questions without knowing the physical storage layout — a landmark abstraction still underlying nearly all structured data storage 55+ years later.

### Historical background
SQL standardized in 1986 (ANSI SQL); every major RDBMS extends it slightly differently but the relational algebra core is universal. In 2026, SQL remains the primary interface for structured feature stores, data warehouses feeding ML training pipelines, and increasingly, natural-language-to-SQL is itself a common LLM application (Phase 7 territory) — meaning you must know real SQL to evaluate whether an LLM's generated SQL is even correct.

### Real-world motivation
Feature engineering for any tabular ML task (your XGBoost mortality model, actuarial pricing) almost always starts with SQL against a warehouse or production database. Bad SQL either silently gives wrong numbers (subtle join bugs) or takes hours instead of seconds on real data volumes.

---

## 2. Theory

### Relational model core concepts
- **Table (relation)**: a set of rows (tuples), each conforming to a schema (named, typed columns).
- **Primary key**: uniquely identifies each row.
- **Foreign key**: a column referencing another table's primary key, enforcing referential integrity.
- **Normalization**: organizing tables to eliminate redundancy (1NF, 2NF, 3NF) — vs. **denormalization**, deliberately duplicating data for query speed, common in analytics/feature-store tables.

### Query execution conceptual order (NOT the order you write it in!)
```
FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT
```
This is the single most important mental model in SQL: you *write* `SELECT` first, but the engine *evaluates* `FROM`/`JOIN`/`WHERE` first. This explains why you can't reference a `SELECT` alias inside a `WHERE` clause (it doesn't exist yet at that evaluation stage) but *can* in `ORDER BY` (evaluated after `SELECT`).

### Join types
| Join | Result |
|---|---|
| `INNER JOIN` | only rows matching in both tables |
| `LEFT JOIN` | all left rows + matches from right (NULL if none) |
| `RIGHT JOIN` | all right rows + matches from left |
| `FULL OUTER JOIN` | all rows from both, matched where possible |
| `CROSS JOIN` | Cartesian product — every combination |

---

## 3. Mathematical Foundations

### Relational algebra (the formal theory SQL implements)
SQL operations map directly onto relational algebra operators, which are themselves operations on **sets**:
- Selection $\sigma_{\text{condition}}(R)$ → SQL `WHERE`
- Projection $\pi_{\text{columns}}(R)$ → SQL `SELECT`
- Join $R \bowtie S$ → SQL `JOIN`
- Union $R \cup S$ → SQL `UNION`
- Set difference $R - S$ → SQL `EXCEPT`

A join's formal definition:
$$
R \bowtie_{\theta} S = \{ t_r \cup t_s : t_r \in R, t_s \in S, \theta(t_r, t_s) \text{ holds} \}
$$
where $\theta$ is the join predicate (e.g., `R.id = S.id`). Naively this is $O(|R| \times |S|)$ (nested loop join); real query planners use hash joins ($O(|R| + |S|)$ expected, building a hash table on the smaller table — directly reusing Lesson 4's hash-table complexity) or sort-merge joins ($O(n \log n)$) depending on data size, indexes, and distribution statistics.

### Query planning as combinatorial optimization
For a query joining $k$ tables, there are up to $k!$ possible join orders (more, considering different join algorithms per step). Query optimizers use cost-based estimation (row-count statistics, histograms) plus dynamic programming (again reusing Lesson 4!) to avoid brute-force enumeration, typically solving optimal join order for small-to-moderate $k$ and using heuristics beyond that.

### Aggregation and window functions
`GROUP BY` implements the mathematical notion of partitioning a set into equivalence classes by a key, then applying an aggregate function $f: \text{class} \to \mathbb{R}$ (e.g., `AVG`, `SUM`, `COUNT`) to each partition. Window functions (`OVER (PARTITION BY ... ORDER BY ...)`) generalize this by computing aggregates *without collapsing rows* — essential for rolling statistics (e.g., trailing 12-month claims average per policyholder).

---

## 4. Algorithm — How a Query Planner Chooses a Join Strategy (conceptual)

```
GIVEN: query joining claims (10M rows) and policyholders (100K rows) on policyholder_id
IF policyholder_id is indexed on policyholders:
    -> Nested Loop Join: for each claims row, index-lookup policyholders -> O(n log m) or O(n) with hash index
ELSE IF no index but enough memory:
    -> Hash Join: build hash table on smaller table (policyholders) -> O(n + m)
ELSE (data too large for memory, or already sorted):
    -> Sort-Merge Join: sort both on key, merge in one pass -> O(n log n + m log m)
Query planner picks based on: table sizes, available indexes, memory, and data statistics (collected via ANALYZE)
```
This is why `EXPLAIN ANALYZE` is the single most useful SQL debugging command — it shows you which strategy the planner actually chose and where time is spent.

---

## 5. Practical Implementation

```sql
-- Schema for an actuarial-style example
CREATE TABLE policyholders (
    policyholder_id  SERIAL PRIMARY KEY,
    birth_date       DATE NOT NULL,
    gender           CHAR(1),
    region           VARCHAR(50)
);

CREATE TABLE claims (
    claim_id         SERIAL PRIMARY KEY,
    policyholder_id  INT REFERENCES policyholders(policyholder_id),
    claim_date       DATE NOT NULL,
    amount           NUMERIC(12, 2) NOT NULL,
    claim_type       VARCHAR(50)
);

CREATE INDEX idx_claims_policyholder ON claims(policyholder_id);  -- speeds up the join below

-- Average claim amount and claim count per age band, last 3 years, region-filtered
WITH aged_policyholders AS (
    SELECT
        policyholder_id,
        region,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)) AS age,
        WIDTH_BUCKET(EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)), 0, 100, 10) AS age_band
    FROM policyholders
)
SELECT
    ap.age_band,
    ap.region,
    COUNT(c.claim_id)                       AS claim_count,
    ROUND(AVG(c.amount), 2)                 AS avg_claim_amount,
    ROUND(SUM(c.amount), 2)                 AS total_claims
FROM aged_policyholders ap
JOIN claims c ON c.policyholder_id = ap.policyholder_id
WHERE c.claim_date >= CURRENT_DATE - INTERVAL '3 years'
GROUP BY ap.age_band, ap.region
HAVING COUNT(c.claim_id) > 30          -- suppress statistically noisy small groups
ORDER BY ap.age_band, ap.region;

-- Window function: each policyholder's trailing 12-month claim total
SELECT
    policyholder_id,
    claim_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY policyholder_id
        ORDER BY claim_date
        RANGE BETWEEN INTERVAL '12 months' PRECEDING AND CURRENT ROW
    ) AS trailing_12mo_total
FROM claims
ORDER BY policyholder_id, claim_date;
```

**Notes:** the `WITH aged_policyholders AS (...)` is a **CTE (Common Table Expression)** — a named, readable subquery, preferred in 2026 style over deeply nested subqueries. The window function computes a *running* total per policyholder without collapsing rows, unlike `GROUP BY`.

---

## 6. Build From Scratch

**A minimal in-memory relational join engine in Python (to demystify what a JOIN actually does):**
```python
def hash_join(left: list[dict], right: list[dict], left_key: str, right_key: str) -> list[dict]:
    """Mirrors a real hash-join query plan: O(len(left) + len(right))."""
    index: dict = {}
    for row in right:
        index.setdefault(row[right_key], []).append(row)

    result = []
    for l_row in left:
        for r_row in index.get(l_row[left_key], []):
            merged = {**l_row, **{f"right_{k}": v for k, v in r_row.items()}}
            result.append(merged)
    return result

def group_by_aggregate(rows: list[dict], key: str, agg_col: str) -> dict:
    """Mirrors GROUP BY + AVG()."""
    groups: dict = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row[agg_col])
    return {k: sum(v) / len(v) for k, v in groups.items()}
```
This is precisely a hash join and group-by aggregate, implemented explicitly — and it's the same $O(n+m)$ hash-table mechanism from Lesson 4, applied to relational data.

---

## 7. Library/Tool Comparison

| From scratch | Production |
|---|---|
| `hash_join` (Python dict) | PostgreSQL/MySQL query planner's hash join operator (C, disk-spill-aware for data larger than RAM) |
| `group_by_aggregate` | SQL `GROUP BY` + aggregate functions, or `pandas.groupby` (Phase 2) |
| Manual row filtering | SQL `WHERE` pushed down to storage engine with index usage — vastly faster on large tables |
| No transactions | Real RDBMS: ACID transactions (Atomicity, Consistency, Isolation, Durability) |

---

## 8. Visual Explanations

```
INNER JOIN                    LEFT JOIN
 A     B                       A     B
┌───┐ ┌───┐                  ┌───┐ ┌───┐
│ ● │ │ ● │  -> matched only  │ ● │ │ ● │  -> ALL of A, matched B where possible
│ ○ │ │ ○ │     (● rows)      │ ○ │ │ ○ │     (● matched, ○ from A gets NULL for B)
└───┘ └───┘                  └───┘ └───┘
```

**Logical execution order vs written order:**
```
WRITTEN:                          EXECUTED:
SELECT col                        1. FROM/JOIN  (build the working row set)
FROM t1 JOIN t2                   2. WHERE      (filter rows)
WHERE ...                         3. GROUP BY   (partition)
GROUP BY ...                      4. HAVING     (filter groups)
HAVING ...                        5. SELECT     (project columns/aliases)
ORDER BY ...                      6. ORDER BY    (sort final result)
```

---

## 9. Practical Examples

**Simple:** `SELECT * FROM claims WHERE amount > 10000 ORDER BY claim_date DESC LIMIT 10;`
**Medium:** find policyholders with more than 5 claims in the last year using `GROUP BY` + `HAVING`.
**Real-world:** a feature-engineering query producing a per-policyholder feature row (age, region, claim_count_1y, avg_claim_amount_1y, days_since_last_claim) ready to feed directly into an XGBoost training pipeline (Phase 4) — exactly the kind of query you'd write before your mortality/pricing model even sees the data.

---

## 10. Real Industry Use Cases

- **Feature stores** (Feast, Tecton): often backed by SQL warehouses; feature definitions are frequently just versioned SQL queries.
- **Snowflake / BigQuery / Redshift**: petabyte-scale SQL warehouses feeding both BI dashboards and ML training data extraction at every major tech company.
- **Text-to-SQL LLM applications**: a major 2026 enterprise AI use case — LLMs translate natural language questions into SQL against a company's warehouse; evaluating these systems requires genuine SQL fluency to catch silently wrong joins/aggregations.
- **A/B test analysis** (Netflix, Meta, Airbnb): typically SQL-first, computing metric aggregates and confidence intervals directly in the warehouse before any Python touches the data.

---

## 11. Common Mistakes

- **Silent row multiplication from an unintended one-to-many join** — joining a "one row per policyholder" table to a "many rows per policyholder" claims table *before* aggregating inflates counts/sums invisibly. Always aggregate the many-side first (CTE) or verify row counts before/after joins.
- Using `COUNT(*)` when you meant `COUNT(DISTINCT column)` — silently counts duplicates.
- Forgetting `NULL` handling: `NULL = NULL` evaluates to `NULL` (not `TRUE`) in SQL's three-valued logic — use `IS NULL`/`IS NOT NULL`, and remember aggregate functions (`AVG`, `SUM`) silently skip `NULL`s, which can bias results if nulls aren't actually missing-at-random.
- Not using `EXPLAIN ANALYZE` before assuming a slow query is "just how it is" — often a missing index is the entire problem.

---

## 12. Best Practices (2026)

- Write CTEs (`WITH ... AS (...)`) instead of deeply nested subqueries — dramatically more readable and maintainable, and modern optimizers handle CTEs efficiently (materialization behavior varies by engine — check).
- Always run `EXPLAIN ANALYZE` on any query touching a table beyond a few hundred thousand rows before shipping it into a pipeline.
- Index foreign keys used in joins and columns used in frequent `WHERE`/`ORDER BY` clauses — but don't over-index (each index slows down writes).
- For ML feature pipelines, prefer **incremental/idempotent** queries (e.g., `INSERT ... ON CONFLICT DO UPDATE`, partitioned by date) over full-table recomputation where table size makes that costly.
- When using LLM-generated SQL (text-to-SQL tools), always validate against `EXPLAIN` and spot-check row counts — never trust generated SQL blindly on production data.

---

## 13. Exercises

**Easy:** Write a query returning the 5 most recent claims per region.
**Medium:** Write a query computing each policyholder's claim frequency (claims per year of policy tenure).
**Hard:** Write a query using window functions to flag policyholders whose claim amount in the most recent quarter is more than 2 standard deviations above their own historical average (an anomaly-detection feature, precursor to Phase 4's outlier detection).
**Mathematical:** Given tables of size $n$ and $m$ with no index, derive why a hash join ($O(n+m)$) beats a nested loop join ($O(n \times m)$) as $n, m$ grow, and identify at what relative sizes a sort-merge join might still be preferred (when data is already sorted, or under strict memory limits).
**Coding:** Extend the from-scratch `hash_join` function (Section 6) to support a left join (preserving unmatched left rows with `None` for right-side fields).

---

## 14. Mini Project

Build an **actuarial feature-engineering pipeline in SQL**: given raw `policyholders` and `claims` tables, write a single well-structured query (using CTEs and window functions) that outputs one feature row per policyholder — age, region, tenure, claim frequency, average claim severity, trailing-12-month claim total, and a binary "high-risk" flag — validated with `EXPLAIN ANALYZE` for reasonable performance, ready to be exported and fed into the XGBoost pipeline you'll build in Phase 4.

---

## 15. Interview Preparation

- Explain the logical order of SQL clause execution vs. the order they're written in.
- What's the difference between `WHERE` and `HAVING`, and why can't you filter on an aggregate in `WHERE`?
- Explain when a hash join is preferred over a nested loop join or sort-merge join.
- System design: how would you design a schema + indexing strategy for a claims database expected to grow to 500M rows, optimized for both fast writes (new claims) and fast monthly aggregate reporting?

---

## 16. Summary

SQL is declarative set theory made practical: you describe the relational algebra you want (selection, projection, join, aggregation), and a cost-based query planner — using the same hashing and dynamic-programming ideas from Lesson 4 — decides the fastest physical execution plan. Mastering the *logical execution order*, join mechanics, and `EXPLAIN ANALYZE` is what separates SQL that happens to return correct-looking numbers from SQL you can actually trust at production scale — a non-negotiable skill for any tabular ML/actuarial feature pipeline.

---

## 17. References

- Codd, E.F. — "A Relational Model of Data for Large Shared Data Banks" (1970)
- *SQL Performance Explained* — Markus Winand (also at use-the-index-luke.com)
- PostgreSQL official documentation (query planning, indexing)
- Mode Analytics SQL Tutorial (practical, widely used for analytics SQL)
