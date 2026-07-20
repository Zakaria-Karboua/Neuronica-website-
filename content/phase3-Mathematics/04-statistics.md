# Phase 3 · Lesson 4 — Statistics

> Prerequisite: Probability (Lesson 3)

---

## 1. Introduction

### What is statistics?
The discipline of drawing reliable conclusions from data — estimation, hypothesis testing, confidence intervals — under explicitly quantified uncertainty. Where probability theory reasons *forward* (given a known distribution, what data would we expect), statistics reasons *backward* (given observed data, what can we infer about the underlying distribution/process).

### Why does it exist?
Any single dataset is a finite, noisy sample from a broader (often infinite, hypothetical) population or process. Statistics provides the formal machinery for distinguishing genuine signal from sampling noise — critically important given how easy it is for humans (and models) to see patterns in pure randomness.

### Historical background
Modern inferential statistics largely crystallized in the early 20th century: Fisher's work on experimental design, maximum likelihood, and significance testing (1920s); Neyman-Pearson's hypothesis testing framework (1930s); the ongoing, still-unresolved tension between these frameworks and Bayesian alternatives. In 2026, statistics remains the rigorous backbone underneath ML evaluation, A/B testing, and any claim of "this model/change is actually better," even as deep learning's engineering culture sometimes underemphasizes it.

### Real-world motivation
Every model comparison in Phase 4 ("is XGBoost actually better than logistic regression on this data, or is the difference just noise?") is a statistical question requiring exactly this lesson's tools, not just "which number is bigger."

---

## 2. Theory

### Point estimation and estimator properties
An **estimator** $\hat\theta$ is a function of data used to estimate a population parameter $\theta$. Desirable properties:
- **Unbiasedness**: $E[\hat\theta] = \theta$ (no systematic over/under-estimation).
- **Consistency**: $\hat\theta \to \theta$ as $n \to \infty$.
- **Efficiency**: minimum variance among unbiased estimators (Cramér-Rao lower bound formalizes the theoretical floor).

### Confidence intervals
A $(1-\alpha)$ confidence interval is constructed so that, *over repeated sampling*, it contains the true parameter $(1-\alpha)$ of the time — a subtle, frequently misinterpreted statement (it is **not** "there's a 95% probability the true value is in this specific interval," which is a Bayesian-flavored statement the frequentist framework doesn't license).

### Hypothesis testing framework
$$
H_0 \text{ (null hypothesis)}, \quad H_1 \text{ (alternative)}
$$
A test statistic is computed from data; a **p-value** is the probability of observing a test statistic at least as extreme as the one observed, *assuming $H_0$ is true*. Reject $H_0$ if $p < \alpha$ (significance level, commonly 0.05) — **not** proof $H_1$ is true, only that the observed data would be unusual under $H_0$.

### Type I / Type II errors and statistical power
| | $H_0$ True | $H_0$ False |
|---|---|---|
| **Reject $H_0$** | Type I error (false positive), rate $\alpha$ | Correct (true positive) |
| **Fail to reject** | Correct (true negative) | Type II error (false negative), rate $\beta$ |

**Power** $= 1 - \beta$ — the probability of correctly detecting a real effect. Power depends on effect size, sample size, variance, and $\alpha$ — critically important when *designing* an experiment (how much data do I need to reliably detect a meaningful difference), not just when analyzing one after the fact.

---

## 3. Mathematical Foundations

### The t-test, derived conceptually
Comparing two group means $\bar{X}_1, \bar{X}_2$ with pooled standard error $SE$:
$$
t = \frac{\bar{X}_1 - \bar{X}_2}{SE}, \qquad SE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$
Under $H_0$ (equal means), $t$ follows a Student's t-distribution with degrees of freedom determined by the Welch-Satterthwaite equation (for unequal variances) — the t-distribution's heavier tails than the normal (especially at small $n$) correctly account for the extra uncertainty from estimating variance from a finite sample rather than knowing it exactly.

### Multiple testing correction
Running $m$ independent hypothesis tests at $\alpha = 0.05$ each gives probability of *at least one* false positive:
$$
P(\text{at least one Type I error}) = 1 - (1-\alpha)^m
$$
For $m = 20$ tests, this is already $\approx 64\%$ — a direct mathematical proof of why "testing many features/hypotheses and reporting whichever came back significant" is statistically invalid without correction. **Bonferroni correction**: use $\alpha/m$ per test; **Benjamini-Hochberg (FDR control)**: a less conservative, more commonly used modern alternative controlling the *expected proportion* of false discoveries rather than the probability of any single one.

### Bootstrap resampling (a modern, distribution-free alternative)
Given a sample of size $n$, generate $B$ bootstrap samples by resampling *with replacement* from the original data, compute the statistic of interest on each, and use the resulting empirical distribution to construct confidence intervals — valid even when no clean analytic formula exists (e.g., confidence interval for a median, or for a complex ML model's evaluation metric), at the cost of $O(B \times n)$ computation instead of a closed-form calculation.

### Statistical power formalized
For a two-sample t-test with effect size $d$ (Cohen's d, standardized mean difference), required sample size per group for power $1-\beta$ at significance $\alpha$:
$$
n \approx \frac{2(z_{\alpha/2} + z_\beta)^2}{d^2}
$$
This is the formal justification behind "power analysis" — computing, *before* running an experiment, how much data is actually needed to detect a meaningful effect, preventing both wasted resources (overpowered studies) and inconclusive null results (underpowered studies).

---

## 4. Algorithm — Bootstrap Confidence Interval (step by step)

```
GIVEN a sample of n observations and a statistic of interest (e.g., the median):
1. FOR b = 1 to B (e.g., B = 10,000):
     a. Draw a bootstrap sample of size n, WITH REPLACEMENT, from the original data
     b. Compute the statistic (e.g., median) on this bootstrap sample -> store it
2. Sort all B computed statistics
3. 95% confidence interval = [2.5th percentile, 97.5th percentile] of the bootstrap distribution
   (this is the "percentile method" -- simplest; more sophisticated corrections exist, e.g. BCa)
```
Complexity: $O(B \times n)$ — trivially parallelizable across bootstrap replicates, a good fit for the multiprocessing concepts from Phase 1 Lesson 2.

---

## 5. Python Implementation

```python
"""statistics_core.py — hypothesis testing, multiple comparisons, bootstrap"""
import numpy as np
from scipy import stats


def welch_t_test(group_a: np.ndarray, group_b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test -- does NOT assume equal variances, the safer default over Student's t-test."""
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    return t_stat, p_value


def benjamini_hochberg(p_values: np.ndarray, fdr: float = 0.05) -> np.ndarray:
    """Returns a boolean array of which hypotheses are significant after FDR correction."""
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = (np.arange(1, m + 1) / m) * fdr
    below_threshold = sorted_p <= thresholds
    if not below_threshold.any():
        return np.zeros(m, dtype=bool)
    max_significant_rank = np.max(np.where(below_threshold))
    significant = np.zeros(m, dtype=bool)
    significant[order[:max_significant_rank + 1]] = True
    return significant


def bootstrap_ci(data: np.ndarray, statistic_fn=np.median, n_bootstrap: int = 10_000, ci: float = 0.95):
    rng = np.random.default_rng(0)
    n = len(data)
    boot_stats = np.array([
        statistic_fn(rng.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)
    ])
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return lower, upper


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    model_a_scores = rng.normal(0.82, 0.03, 30)   # e.g., 30-fold CV accuracy for model A
    model_b_scores = rng.normal(0.85, 0.03, 30)   # model B, genuinely a bit better

    t_stat, p_val = welch_t_test(model_a_scores, model_b_scores)
    print(f"t={t_stat:.3f}, p={p_val:.4f}")

    # Simulating 20 feature-target correlation tests, only some truly non-null
    p_values = np.concatenate([rng.uniform(0, 0.01, 3), rng.uniform(0, 1, 17)])
    print("Significant after FDR correction:", benjamini_hochberg(p_values))

    claim_amounts = rng.lognormal(7, 1.2, 200)
    print("Bootstrap 95% CI for median claim:", bootstrap_ci(claim_amounts))
```

---

## 6. Build From Scratch

**A minimal permutation test (distribution-free hypothesis testing, from scratch):**
```python
import numpy as np

def permutation_test(group_a: np.ndarray, group_b: np.ndarray, n_permutations: int = 10_000) -> float:
    """Tests H0: no difference between groups, WITHOUT assuming normality (unlike the t-test)."""
    observed_diff = np.mean(group_a) - np.mean(group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    rng = np.random.default_rng(0)

    count_extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(combined)
        perm_a, perm_b = shuffled[:n_a], shuffled[n_a:]
        perm_diff = np.mean(perm_a) - np.mean(perm_b)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    return count_extreme / n_permutations   # empirical p-value
```
**The logic, made explicit:** under $H_0$, group labels are meaningless — any random reshuffling of which observations belong to "A" vs. "B" is equally likely. If the *actually observed* group difference is rarely matched or exceeded by random reshuffles, that's direct evidence against $H_0$ — no distributional assumption (normality) required, unlike the t-test.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `permutation_test` | `scipy.stats.permutation_test` (vectorized, faster, more options) |
| `bootstrap_ci` | `scipy.stats.bootstrap` (supports BCa correction, more accurate than the plain percentile method) |
| `benjamini_hochberg` | `statsmodels.stats.multitest.multipletests(method="fdr_bh")` |
| Manual Welch's t-test | `scipy.stats.ttest_ind(equal_var=False)` — identical formula, well-tested edge cases |

---

## 8. Visual Explanations

**Type I vs. Type II error tradeoff (overlapping distributions under H0 and H1):**
```
   H0 distribution        H1 distribution
        ___                    ___
       /   \                  /   \
      /     \________________/     \
     /       │    overlap    │      \
    /        │  (ambiguous   │       \
             │    region)    │
             ▲ decision threshold
Moving threshold LEFT: fewer Type II errors, MORE Type I errors (and vice versa) -- a genuine tradeoff, not free.
```

**Bootstrap resampling (conceptual):**
```
Original sample:  [5, 8, 3, 9, 2]
Bootstrap 1:      [8, 8, 2, 5, 3]   (resampled WITH replacement)
Bootstrap 2:      [3, 9, 9, 2, 5]
Bootstrap 3:      [5, 5, 8, 3, 2]
...  (repeat B times)  ...
-> compute statistic on EACH bootstrap sample -> build empirical distribution -> read off percentiles
```

---

## 9. Practical Examples

**Simple:** run a two-sample t-test comparing claim amounts between two regions.
**Medium:** apply Benjamini-Hochberg correction to a set of 15 feature-target correlation tests from an EDA pass (Phase 2 Lesson 5), and compare which features remain "significant" versus a naive uncorrected threshold.
**Real-world:** run a power analysis to determine the minimum sample size needed to detect a genuinely meaningful difference (e.g., 2 percentage points) in claim-approval rates between two underwriting policies before actually collecting/analyzing that data — a real, common actuarial/business decision this lesson directly equips you to make rigorously.

---

## 10. Real Industry Use Cases

- **A/B testing at every major tech company** (Netflix, Airbnb, Booking.com): hypothesis testing, power analysis, and multiple-testing correction are the statistical backbone of every product experiment, and are frequently the subject of dedicated internal "experimentation platform" teams.
- **Model comparison in ML research and production**: rigorously determining whether Model B's improvement over Model A is statistically significant (not just numerically larger on one test set) using paired t-tests or bootstrap confidence intervals on the performance metric.
- **Actuarial rate filings**: regulatory approval for insurance pricing changes often requires statistically justified confidence intervals and significance testing on claims experience data, not just point estimates.
- **Clinical trials** (relevant to your cardiology work): the entire clinical trial methodology (power analysis for trial size, pre-registered hypotheses, multiple-testing correction across endpoints) is applied statistics at its highest-stakes.

---

## 11. Common Mistakes

- **P-hacking**: testing many hypotheses/features and reporting only the significant ones without multiple-testing correction — Section 3's math shows this inflates false-positive rates dramatically and predictably.
- Misinterpreting a p-value as "the probability the null hypothesis is true" (it is not — it's the probability of the observed data, or more extreme, *given* the null is true).
- Misinterpreting a 95% confidence interval as "95% probability the true value is in this specific interval" (a Bayesian-flavored misreading of a frequentist construct).
- Running an underpowered study/experiment and treating a non-significant result as "proof of no effect," when it may simply reflect insufficient sample size (a Type II error risk, not a settled negative finding).

---

## 12. Best Practices (2026)

- Always pre-register hypotheses and correction methods (Bonferroni/FDR) *before* looking at results — post-hoc "just this one test" reasoning is a direct route to p-hacking.
- Prefer bootstrap-based confidence intervals over normal-approximation formulas whenever the underlying distribution is skewed/heavy-tailed (e.g., claims/financial data) or when no clean analytic formula exists for the statistic of interest.
- Run power analyses *before* collecting data for any planned comparison, not just after getting an inconclusive result.
- Report effect sizes and confidence intervals alongside (not instead of) p-values — a p-value alone says nothing about the *magnitude* of a difference, only whether it's distinguishable from pure chance at the observed sample size.

---

## 13. Exercises

**Easy:** Run a t-test comparing two synthetic groups and interpret the resulting p-value correctly in one sentence.
**Medium:** Simulate 100 independent null hypothesis tests (all truly null) and empirically show the proportion of "significant" results at $\alpha=0.05$ matches the multiple-testing math from Section 3.
**Hard:** Implement a full power analysis: given an assumed effect size and variance, compute (via simulation, not just the formula) the sample size needed for 80% power at $\alpha = 0.05$, and verify against the closed-form formula in Section 3.
**Mathematical:** Derive why the Bonferroni correction ($\alpha/m$ per test) controls the family-wise error rate at $\alpha$ using a union bound argument.
**Coding:** Implement the BCa (bias-corrected and accelerated) bootstrap confidence interval and compare it against the plain percentile method (Section 5) on a skewed synthetic dataset.

---

## 14. Mini Project

Design and analyze a **complete statistical experiment**: simulate two underwriting policies' claim-approval rates with a genuinely small true difference, run a power analysis to determine adequate sample size *before* simulating the "experiment" data, then analyze the resulting data with an appropriate hypothesis test (accounting for multiple comparisons if testing several sub-metrics), a bootstrap confidence interval on the effect size, and a written recommendation — explicitly distinguishing "no significant difference detected" from "confidently no meaningful difference exists," a distinction with real business/regulatory consequences.

---

## 15. Interview Preparation

- Explain the difference between a Type I and Type II error, and the tradeoff between them.
- What does a p-value actually mean, and what's the most common misinterpretation?
- Why is multiple-testing correction necessary, and what's the difference between Bonferroni and FDR (Benjamini-Hochberg) correction?
- Design question: how would you determine the required sample size for an A/B test before running it?

---

## 16. Summary

Statistics formalizes the discipline of separating genuine signal from sampling noise: confidence intervals and hypothesis tests quantify uncertainty rigorously (with p-values meaning something quite specific and easily misinterpreted), multiple-testing correction (Bonferroni/FDR) prevents false discoveries from accumulating across many tests, and bootstrap resampling provides a distribution-free alternative when analytic formulas don't apply or normality assumptions are questionable — exactly the case for the skewed claims/financial data central to your domain. Every serious model comparison or experimental conclusion in Phase 4 onward rests on these tools being applied correctly.

---

## 17. References

- Wasserman, L. — *All of Statistics*
- Efron, B. & Tibshirani, R. — *An Introduction to the Bootstrap* (the definitive bootstrap reference)
- Benjamini, Y. & Hochberg, Y. — "Controlling the False Discovery Rate" (1995, the original FDR paper)
- Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments* (the modern, industry-grounded A/B testing reference)
