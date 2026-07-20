# Phase 2 · Lesson 4 — Data Visualization

> Prerequisite: NumPy, Pandas, Data Cleaning (Lessons 1–3)

---

## 1. Introduction

### What is data visualization?
The systematic encoding of data into visual form (position, length, color, shape) to exploit the human visual system's pattern-recognition capacity — far faster and more reliable for detecting structure, trends, and anomalies than reading raw numbers. In Python: `matplotlib` (the low-level foundation), `seaborn` (statistical-graphics layer on matplotlib), and `plotly` (interactive, web-native).

### Why does it exist?
Anscombe's Quartet (1973) is the canonical demonstration: four datasets with nearly identical mean, variance, correlation, and regression line — yet wildly different underlying structure (linear, curved, one outlier-driven, one with a vertical cluster) — visible instantly on a scatter plot, invisible in summary statistics alone. Visualization exists because summary statistics can lie by omission; plots reveal what numbers hide.

### Historical background
Modern statistical graphics theory traces to William Playfair (inventor of the bar/line/pie chart, 1780s-90s) and was formalized by Jacques Bertin's *Semiology of Graphics* (1967) and later Edward Tufte's work on graphical integrity and "data-ink ratio." Leland Wilkinson's *The Grammar of Graphics* (1999) — the theoretical basis for `ggplot2` and Python's `plotnine` — treats a chart as a composable mapping from data to visual encodings, a genuinely mathematical framework, not just "make it pretty."

### Real-world motivation
Every EDA (Lesson 5) you do, every model diagnostic (residual plots, calibration curves, ROC curves in Phase 4), and every stakeholder-facing report (your Brent oil forecasting work, DZD exchange-rate analysis) lives or dies on whether the visualization actually communicates the right thing without misleading.

---

## 2. Theory

### The Grammar of Graphics (conceptual, informs all modern plotting libraries)
Any statistical graphic is built from:
- **Data**: the dataset being visualized.
- **Aesthetic mappings**: which data columns map to which visual channels (x-position, y-position, color, size, shape).
- **Geometric objects (geoms)**: how data is rendered (points, lines, bars, areas).
- **Statistical transformations**: aggregations applied before rendering (binning for histograms, smoothing for trend lines).
- **Scales**: how data values map to the visual range (linear, log, categorical).
- **Facets**: small multiples — splitting one plot into a grid by a categorical variable.

### Choosing the right chart type (a decision framework, not a style preference)
| Question | Chart type |
|---|---|
| Distribution of one numeric variable? | Histogram, KDE, boxplot |
| Relationship between two numeric variables? | Scatter plot |
| Comparison across categories? | Bar chart (never pie chart for >3 categories — see Section 11) |
| Trend over time? | Line chart |
| Distribution across many groups? | Boxplot/violin plot, faceted histograms |
| Correlation structure among many variables? | Heatmap |
| Part-to-whole with very few categories? | Stacked bar (generally superior to pie even here) |

### Perceptual accuracy ranking (Cleveland & McGill, 1984 — empirically measured, not opinion)
Humans judge visual encodings with different accuracy, ranked from most to least accurate:
$$
\text{position (common scale)} > \text{length} > \text{angle/slope} > \text{area} > \text{color/shading}
$$
This is *why* bar charts (length/position) reliably outperform pie charts (angle) for comparison tasks, and why color should encode categories, not precise magnitude comparisons, whenever position-based alternatives exist.

---

## 3. Mathematical Foundations

### Histograms and bin-width selection
A histogram estimates a probability density by counting observations in bins of width $h$. Bin width dramatically affects the *interpretation* of the same data — too narrow shows noise as structure, too wide hides real structure. Freedman-Diaconis rule (robust to outliers, a good practical default):
$$
h = 2 \cdot \frac{\text{IQR}(x)}{n^{1/3}}
$$
using IQR (robust, from Lesson 3) rather than standard deviation makes bin-width selection resistant to the same outlier-distortion problem covered there.

### Kernel Density Estimation (KDE) — a smoother alternative to histograms
$$
\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)
$$
where $K$ is a kernel function (commonly Gaussian) and $h$ is the bandwidth (analogous to histogram bin width — the same underlying bias-variance tradeoff: small $h$ = noisy/overfit, large $h$ = oversmoothed).

### Color perception and colorblind-safe design
Roughly 8% of men and 0.5% of women have some form of color vision deficiency (most commonly red-green). Perceptually uniform, colorblind-safe colormaps (`viridis`, `cividis`) are designed so that (a) perceived brightness changes linearly with data value (unlike rainbow/`jet` colormaps, which introduce false visual boundaries) and (b) they remain distinguishable under common colorblindness simulations — a genuine correctness issue, not merely aesthetic.

---

## 4. Algorithm — Building a Statistical Graphic (conceptual pipeline)

```
GIVEN a dataset and an analytical question:
1. Identify variable types (numeric continuous / discrete / categorical / temporal)
2. Choose aesthetic mappings matching the QUESTION (not habit): 
     "how does X relate to Y" -> position (scatter/line), NOT color/size as primary encoding
3. Choose the appropriate statistical transform if needed (binning, smoothing, aggregation)
4. Select a perceptually accurate encoding (Cleveland-McGill ranking) for the MOST IMPORTANT comparison
5. Apply colorblind-safe, appropriately-scaled color only for SECONDARY encoding (category, not precise value)
6. Remove non-data ink (Tufte's principle): unnecessary gridlines, 3D effects, redundant legends
7. Add clear labels, units, and a title stating the FINDING, not just the variable names
```

---

## 5. Python Implementation

```python
"""dataviz_core.py — grammar-of-graphics-informed, publication-quality plotting"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", palette="viridis")   # colorblind-safe default

claims = pd.DataFrame({
    "region": np.random.default_rng(0).choice(["North", "South", "East", "West"], size=1000),
    "age": np.random.default_rng(1).normal(45, 15, size=1000).clip(18, 90),
    "amount": np.random.default_rng(2).lognormal(mean=7, sigma=1, size=1000),   # skewed, like real claims
})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Distribution of a skewed numeric variable — histogram + KDE, log scale where appropriate
sns.histplot(claims["amount"], kde=True, bins="fd", ax=axes[0, 0])   # "fd" = Freedman-Diaconis rule
axes[0, 0].set_xscale("log")
axes[0, 0].set_title("Claim Amount Distribution (log scale, FD bin width)")

# 2. Relationship between two numeric variables — scatter with regression trend
sns.regplot(data=claims, x="age", y="amount", scatter_kws={"alpha": 0.3}, ax=axes[0, 1])
axes[0, 1].set_title("Claim Amount vs. Age")

# 3. Comparison across categories — bar chart (position/length, NOT pie chart)
region_means = claims.groupby("region")["amount"].mean().sort_values()
sns.barplot(x=region_means.index, y=region_means.values, ax=axes[1, 0])
axes[1, 0].set_title("Average Claim Amount by Region")
axes[1, 0].set_ylabel("Mean Amount")

# 4. Distribution across groups — boxplot (shows spread + outliers, not just the mean)
sns.boxplot(data=claims, x="region", y="amount", ax=axes[1, 1])
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("Claim Amount Spread by Region (log scale)")

plt.tight_layout()
plt.savefig("claims_dashboard.png", dpi=150)
```

**Notes:** using `bins="fd"` directly applies the Freedman-Diaconis rule from Section 3 rather than an arbitrary default bin count; the log scale on the skewed `amount` variable prevents a few large claims from compressing the entire rest of the distribution into an unreadable spike near zero — a very common real mistake with financial/claims data.

---

## 6. Build From Scratch

**A minimal histogram binning + rendering routine (to demystify what `plt.hist` does):**
```python
import math

def freedman_diaconis_bins(data: list[float]) -> int:
    sorted_data = sorted(data)
    n = len(sorted_data)
    q1 = sorted_data[n // 4]
    q3 = sorted_data[3 * n // 4]
    iqr = q3 - q1
    if iqr == 0:
        return int(math.sqrt(n))    # fallback for degenerate data
    bin_width = 2 * iqr / (n ** (1 / 3))
    data_range = sorted_data[-1] - sorted_data[0]
    return max(1, int(math.ceil(data_range / bin_width)))

def compute_histogram(data: list[float], n_bins: int) -> list[int]:
    lo, hi = min(data), max(data)
    bin_width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for x in data:
        idx = min(int((x - lo) / bin_width), n_bins - 1)   # clamp the max value into the last bin
        counts[idx] += 1
    return counts
```
This is exactly what `numpy.histogram`/`matplotlib.pyplot.hist` do internally: choose a bin count (often via a rule like Freedman-Diaconis), then a single linear-time pass bucketing each value — $O(n)$ total.

---

## 7. Library/Tool Comparison

| From scratch | Production library |
|---|---|
| `compute_histogram` | `np.histogram` / `plt.hist` — handles edge cases, weighted histograms, density normalization |
| Manual `plt` subplot wiring | `seaborn` — statistical-graphics layer with sensible defaults (confidence bands, colorblind-safe palettes) |
| Static PNG output | `plotly`/`bokeh` — interactive, zoomable, hoverable charts, essential for exploratory dashboards |
| Manual color choice | `viridis`/`cividis` perceptually-uniform, colorblind-safe colormaps built into matplotlib/seaborn |

---

## 8. Visual Explanations

**Cleveland-McGill perceptual accuracy ranking (as an ASCII ranking):**
```
MOST accurate                                                  LEAST accurate
Position  >  Length  >  Angle/Slope  >  Area  >  Color/Shading  >  Volume
   │                                                                  │
(bar/scatter charts)                                          (3D pie charts —
                                                                 avoid almost always)
```

**Bin width tradeoff (bias-variance, same concept as KDE bandwidth):**
```
Too few bins (h large):        Too many bins (h small):        Just right (FD rule):
█                                █ █ █ █ █ █ █ █ █ █ █           ██████
██████                          █ ██ █ █ ██ █ █ █ █             ██████████
████████████                    (noisy, overfit to samples)     ████████████████
(oversmoothed, hides structure)                                  (reveals real shape)
```

---

## 9. Practical Examples

**Simple:** a histogram of claim amounts with an appropriately chosen bin width and log scale.
**Medium:** a faceted set of boxplots comparing claim distributions across region AND claim type simultaneously (small multiples).
**Real-world:** a time-series line chart of your Brent crude oil forecasting residuals overlaid with confidence bands, using log scale where appropriate and clear annotation of any structural break (e.g., 2020 oil price crash) — directly reusing your existing forecasting project as the visualization subject.

---

## 10. Real Industry Use Cases

- **Financial/actuarial reporting** (directly your domain): regulators and stakeholders expect claim/loss distributions visualized with correct scale choices (log scale for skewed severity data) — a genuinely common source of misleading regulatory filings when done poorly.
- **ML model monitoring dashboards** (Phase 8): visualizing prediction drift, feature distributions over time, and calibration curves is how production ML issues are caught before they cause real damage.
- **Scientific/clinical reporting** (relevant to your cardiology work): Kaplan-Meier survival curves (a specific, standardized chart type you'll formalize in Phase 3) are a direct, high-stakes application of these visualization principles.
- **Netflix/Airbnb/Uber data science teams**: heavy internal investment in dashboarding tools (often Plotly/Bokeh-based) precisely because fast, accurate visual communication of experiment results drives real business decisions.

---

## 11. Common Mistakes

- **Pie charts for more than 2-3 categories** — angle judgments (Cleveland-McGill) are measurably worse than length/position; a simple sorted bar chart almost always communicates the same information more accurately.
- **Truncated/non-zero y-axes on bar charts** — exaggerates differences and is a classic (sometimes intentional) way to mislead; less of a concern for line charts showing *trend* rather than *magnitude comparison*, but still requires disclosure.
- **Rainbow/`jet` colormaps for continuous data** — introduce perceptually false boundaries (sharp visual transitions where the underlying data changes smoothly), misleading viewers about where "real" structure exists; use `viridis`/`cividis` instead.
- **Ignoring overplotting** — thousands of scatter points fully overlapping hides density entirely; use alpha transparency, hexbin plots, or 2D KDE instead.
- **Showing a mean/bar chart for skewed data without also showing spread** — a bar chart of "average claim amount" alone hides that the distribution is massively right-skewed (a few huge claims) — always pair with a boxplot/violin or explicitly note skewness.

---

## 12. Best Practices (2026)

- Default to `viridis`/`cividis` (or other perceptually uniform, colorblind-safe palettes) rather than matplotlib's old default colormap or manually chosen colors.
- Use `seaborn`'s statistical defaults (automatic confidence intervals on regression/line plots) rather than manually computing and drawing your own, unless you have a specific reason to customize.
- For interactive exploration (not final publication), prefer `plotly`/`bokeh` — hover tooltips and zoom dramatically speed up EDA (Lesson 5).
- Title every chart with the **finding**, not just the variable names (e.g., "Claim severity is 3x higher in the South region" rather than "Claim Amount by Region") — a small habit with outsized communication impact, especially for non-technical stakeholders.

---

## 13. Exercises

**Easy:** Recreate Anscombe's Quartet, computing summary statistics (mean, variance, correlation) for all four datasets and confirming they're nearly identical, then plot all four to show how different they actually look.
**Medium:** Take a skewed financial dataset (or simulate one with a lognormal distribution) and compare a linear-scale vs. log-scale histogram, explaining which is more informative and why.
**Hard:** Implement your own Freedman-Diaconis bin-width selector (Section 6) and validate it against `numpy.histogram_bin_edges(data, bins="fd")` on several real and synthetic datasets.
**Mathematical:** Derive why the Freedman-Diaconis rule's $n^{1/3}$ scaling makes bin width shrink more slowly than a naive $\sqrt{n}$-bin rule as sample size grows, and discuss the implied bias-variance tradeoff.
**Coding:** Build a small function that automatically recommends a chart type given a dataframe's column dtypes and the analytical question type (distribution / relationship / comparison / trend), implementing the Section 2 decision framework as code.

---

## 14. Mini Project

Build a **complete EDA visualization report** for a claims (or your DZD exchange-rate) dataset: distribution plots for every numeric variable (correctly scaled, FD-binned), a correlation heatmap using a perceptually uniform colormap, faceted category comparisons using bar/boxplots (never pie charts), and a written 1-paragraph "finding" caption under each chart — then deliberately create one "bad" version of each chart (wrong chart type, misleading scale, rainbow colormap) side by side with your "good" version, explicitly demonstrating the Cleveland-McGill and Tufte principles you applied.

---

## 15. Interview Preparation

- Why are pie charts generally discouraged, and what does the Cleveland-McGill research actually show?
- When would you use a log scale on an axis, and what's a concrete data-quality risk of not doing so?
- How do you choose between a histogram and a KDE plot, and what's the bandwidth/bin-width tradeoff they share?
- Design question: you need to communicate a claims-severity distribution to non-technical stakeholders who will make a pricing decision based on it — walk through your exact chart choices and why.

---

## 16. Summary

Data visualization is applied perceptual science, not aesthetic preference: the Grammar of Graphics gives a compositional framework for building any chart from data + aesthetic mappings + geometric objects + statistical transforms + scales, while Cleveland-McGill's empirically measured perceptual-accuracy ranking and Tufte's data-ink principles tell you which encodings actually communicate truthfully and which mislead. Every visualization choice — bin width, color map, axis scale, chart type — is a decision with a right and wrong answer given the data and the question, not a matter of taste.

---

## 17. References

- Wilkinson, L. — *The Grammar of Graphics*
- Cleveland, W.S. & McGill, R. — "Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods" (1984)
- Tufte, E. — *The Visual Display of Quantitative Information*
- Anscombe, F.J. — "Graphs in Statistical Analysis" (1973, the original Quartet paper)
- Matplotlib, Seaborn, and Plotly official documentation
