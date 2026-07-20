# Phase 4 · Lesson 2 — Unsupervised Learning

> Prerequisite: Supervised Learning (Lesson 1), Phase 3 Linear Algebra

---

## 1. Introduction

### What is unsupervised learning?
Finding structure in data *without* labeled outcomes — clustering similar points together, reducing dimensionality while preserving meaningful structure, or estimating the underlying probability distribution that generated the data. No "correct answer" exists to check against, making evaluation fundamentally harder and more judgment-dependent than supervised learning.

### Why does it exist?
Labeled data is expensive and often unavailable at the scale needed — but raw, unlabeled data is abundant. Unsupervised learning extracts value from that abundance: discovering customer segments without pre-defined categories, compressing high-dimensional data for visualization or downstream modeling, or detecting anomalies without ever having seen a labeled "anomaly" example.

### Historical background
K-means clustering traces to Lloyd's 1957 algorithm (published 1982); PCA to Pearson (1901) and Hotelling (1933) — both far predating "machine learning" as a field, again originating in classical statistics. Modern unsupervised learning has been reinvigorated by deep learning: autoencoders, contrastive learning (SimCLR, 2020), and the embedding spaces underlying Phase 6-7's entire LLM/RAG stack are all, fundamentally, unsupervised or self-supervised representation learning.

### Real-world motivation
Segmenting policyholders into risk clusters without pre-defined categories, detecting anomalous claims patterns without labeled fraud examples, and reducing a wide correlated feature set (Phase 3 Lesson 1's PCA) down to a manageable, interpretable set of latent risk factors are all direct applications you may already have encountered in actuarial work.

---

## 2. Theory

### Clustering
Partitioning data into groups such that within-group similarity is high and between-group similarity is low, with no ground-truth group labels to check against.
- **K-means**: partition into $k$ clusters minimizing within-cluster variance; requires choosing $k$ in advance; assumes roughly spherical, similarly-sized clusters.
- **Hierarchical clustering**: builds a tree (dendrogram) of nested clusters, agglomerative (bottom-up merging) or divisive (top-down splitting); doesn't require pre-specifying $k$.
- **DBSCAN** (density-based): finds clusters of arbitrary shape based on point density, naturally identifies outliers as "noise" points not belonging to any cluster — often more appropriate than k-means for genuinely irregular real-world cluster shapes.

### Dimensionality reduction
- **PCA** (Phase 3 Lesson 1, formally derived there): linear projection onto directions of maximum variance.
- **t-SNE/UMAP**: non-linear techniques preserving *local* neighborhood structure for visualization, at the cost of not preserving global distances meaningfully (a common misinterpretation trap — cluster *sizes* and *inter-cluster distances* in a t-SNE plot are often not meaningful, only relative local grouping is).

### Density estimation and anomaly detection
Estimating $p(x)$, the underlying data distribution, without labels — points in low-density regions are natural anomaly candidates. Gaussian Mixture Models (GMMs) model $p(x)$ as a weighted sum of Gaussian components, fit via the Expectation-Maximization (EM) algorithm.

---

## 3. Mathematical Foundations

### K-means objective and Lloyd's algorithm
$$
\min_{\{C_k\}, \{\mu_k\}} \sum_{k=1}^{K}\sum_{x_i \in C_k} \|x_i - \mu_k\|^2
$$
This is NP-hard to solve exactly (combinatorial partition search), but Lloyd's algorithm finds a good local optimum via alternating minimization:
1. **Assignment step**: fix centroids, assign each point to its nearest centroid (minimizes the objective over cluster assignments).
2. **Update step**: fix assignments, recompute each centroid as the mean of its assigned points (minimizes the objective over centroid positions, since the mean minimizes sum-of-squared-distances — a direct calculus result: $\frac{d}{d\mu}\sum(x_i-\mu)^2 = 0 \Rightarrow \mu = \bar x$).

Each step provably never increases the objective, guaranteeing convergence (to a local, not necessarily global, optimum) — directly analogous to the coordinate-descent family of optimization methods (Phase 3 Lesson 5).

### The Expectation-Maximization (EM) algorithm for GMMs
GMM: $p(x) = \sum_{k=1}^{K}\pi_k \mathcal{N}(x;\mu_k,\Sigma_k)$. Since we don't observe which component generated each point (a **latent variable** $z_i$), direct MLE (Phase 3 Lesson 3) has no closed form. EM alternates:
- **E-step**: compute the posterior probability (**responsibility**) $\gamma_{ik} = P(z_i=k|x_i)$ that component $k$ generated point $i$, given current parameter estimates (a direct Bayes' theorem application, Phase 3 Lesson 3).
- **M-step**: update $\pi_k, \mu_k, \Sigma_k$ using these soft (probability-weighted) assignments as if they were known, maximizing the *expected* complete-data log-likelihood.

Each EM iteration provably increases (or holds steady) a lower bound on the true log-likelihood — a beautiful, general algorithm (also used far beyond GMMs, e.g., in hidden Markov models) for MLE with latent/missing variables.

### Silhouette score (evaluating clustering without ground truth)
For point $i$: $a(i)$ = mean distance to other points in its own cluster; $b(i)$ = mean distance to points in the nearest *other* cluster.
$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]
$$
$s(i)$ near $+1$: well-clustered; near 0: on a cluster boundary; negative: likely misassigned. Averaging across all points gives an overall clustering-quality score usable to compare different $k$ values — one of the few genuinely quantitative evaluation tools available without labels.

---

## 4. Algorithm — K-Means (Lloyd's Algorithm, fully specified)

```
INITIALIZE: choose k initial centroids (e.g., via k-means++ for smarter initialization)
REPEAT until convergence (assignments stop changing, or max iterations reached):
    ASSIGNMENT STEP:
      FOR each point x_i:
          assign x_i to the cluster k minimizing ||x_i - mu_k||^2
    UPDATE STEP:
      FOR each cluster k:
          mu_k = mean of all points currently assigned to cluster k
RETURN final centroids and assignments
```
**k-means++ initialization** (a crucial practical improvement over naive random initialization): choose the first centroid uniformly at random, then each subsequent centroid with probability proportional to its squared distance from the nearest already-chosen centroid — spreading initial centroids apart, dramatically reducing the chance of poor local optima that plain random initialization is prone to.

---

## 5. Python Implementation

```python
"""unsupervised_learning_core.py"""
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)
# Simulate 3 policyholder risk segments with different claim/age profiles
cluster_a = rng.normal([30, 2000], [5, 500], size=(300, 2))    # young, low claims
cluster_b = rng.normal([55, 8000], [8, 1500], size=(300, 2))   # middle-age, high claims
cluster_c = rng.normal([70, 4000], [6, 800], size=(300, 2))    # older, moderate claims
X = np.vstack([cluster_a, cluster_b, cluster_c])

# --- Choosing k via silhouette score (no ground truth available) ---
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k={k}: silhouette={score:.3f}")

# --- Final model at the best k ---
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("Centroids (age, claim_amount):\n", kmeans.cluster_centers_)

# --- DBSCAN: handles non-spherical clusters + outlier detection natively ---
dbscan = DBSCAN(eps=3.0, min_samples=10).fit(X / X.std(axis=0))   # scale first -- DBSCAN is distance-based
n_outliers = np.sum(dbscan.labels_ == -1)
print(f"DBSCAN found {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)} clusters, "
      f"{n_outliers} outlier points")

# --- Gaussian Mixture Model: soft (probabilistic) cluster assignment ---
gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
responsibilities = gmm.predict_proba(X)   # P(cluster | point) for EVERY cluster, not just the argmax
print("Example soft assignment (point 0):", responsibilities[0].round(3))

# --- PCA for dimensionality reduction (directly Phase 3 Lesson 1's math, applied) ---
pca = PCA(n_components=1).fit(X)
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

---

## 6. Build From Scratch

**K-means from scratch (directly implementing Section 4's algorithm):**
```python
import numpy as np

def kmeans_from_scratch(X: np.ndarray, k: int, n_iters: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    # k-means++ initialization
    centroids = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        dists = np.min([np.sum((X - c)**2, axis=1) for c in centroids], axis=0)
        probs = dists / dists.sum()
        centroids.append(X[rng.choice(len(X), p=probs)])
    centroids = np.array(centroids)

    for _ in range(n_iters):
        distances = np.array([np.sum((X - c)**2, axis=1) for c in centroids])   # (k, n)
        assignments = np.argmin(distances, axis=0)
        new_centroids = np.array([
            X[assignments == j].mean(axis=0) if np.any(assignments == j) else centroids[j]
            for j in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, assignments
```

**A minimal EM algorithm for a 1D two-component Gaussian mixture (to make Section 3's E/M steps fully concrete):**
```python
from scipy.stats import norm

def em_gmm_1d(x: np.ndarray, n_iters: int = 50):
    n = len(x)
    mu1, mu2 = np.percentile(x, 25), np.percentile(x, 75)   # simple init
    sigma1 = sigma2 = x.std()
    pi1 = pi2 = 0.5

    for _ in range(n_iters):
        # E-STEP: compute responsibilities (posterior P(component | x_i)) via Bayes' theorem
        w1 = pi1 * norm.pdf(x, mu1, sigma1)
        w2 = pi2 * norm.pdf(x, mu2, sigma2)
        total = w1 + w2
        r1, r2 = w1 / total, w2 / total   # responsibilities

        # M-STEP: update parameters using responsibility-weighted statistics
        pi1, pi2 = r1.mean(), r2.mean()
        mu1 = (r1 * x).sum() / r1.sum()
        mu2 = (r2 * x).sum() / r2.sum()
        sigma1 = np.sqrt((r1 * (x - mu1)**2).sum() / r1.sum())
        sigma2 = np.sqrt((r2 * (x - mu2)**2).sum() / r2.sum())

    return {"mu1": mu1, "mu2": mu2, "sigma1": sigma1, "sigma2": sigma2, "pi1": pi1, "pi2": pi2}
```

---

## 7. Library Implementation (Comparison)

| From scratch | Production library |
|---|---|
| `kmeans_from_scratch` | `sklearn.cluster.KMeans` — vectorized, multiple restarts (`n_init`), optimized centroid initialization |
| `em_gmm_1d` | `sklearn.mixture.GaussianMixture` — full covariance support, multi-dimensional, regularization for numerical stability |
| Manual PCA (Phase 3 Lesson 1) | `sklearn.decomposition.PCA` — uses efficient SVD-based solvers, handles `n_components` as a variance-ratio threshold |
| No density-based clustering from scratch shown | `sklearn.cluster.DBSCAN`/`HDBSCAN` — production-grade spatial indexing (k-d trees) for efficient neighbor queries |

---

## 8. Visual Explanations

**K-means: alternating assignment/update steps converging:**
```
Iteration 0 (random init):      Iteration 1 (after assign+update):     Converged:
  ●  ○    ×                        ●● ○○   ×                            ●●● ○○○    ×××
    ○  ●                          ○○  ●●                                ○○○  ●●●
  (× = centroids, poorly placed)  (centroids shift toward cluster means) (stable clusters)
```

**K-means (spherical assumption) vs. DBSCAN (arbitrary shape) on non-spherical data:**
```
Two crescent moons:              K-means (WRONG, forces spherical split):     DBSCAN (correct):
  ))))                                 ●●●|○○○                                 ))))
 ((((                                  ●●●|○○○                                ((((
  (splits down the middle,        (splits down the middle,               (correctly follows
   ignoring true shape)            ignoring true crescent shape)          the crescent density)
```

---

## 9. Practical Examples

**Simple:** cluster a 2D synthetic dataset with k-means and visualize the resulting clusters and centroids.
**Medium:** use the silhouette score to choose the optimal $k$ for a policyholder segmentation dataset.
**Real-world:** perform PCA-based dimensionality reduction on a wide, correlated actuarial risk-factor dataset (interest rate, mortality, lapse, market shocks) down to 2-3 principal risk factors, then cluster policyholders/scenarios in that reduced space to identify a small number of representative "risk regimes" for stress testing — directly extending your Phase 3 Lesson 1 PCA mini-project.

---

## 10. Real Industry Use Cases

- **Customer/policyholder segmentation**: insurance, retail, and marketing teams routinely use k-means/hierarchical clustering to discover natural customer segments without predefined categories.
- **Anomaly/fraud detection**: density-based methods (DBSCAN, isolation forests) and GMM-based density estimation flag unusual claims/transactions without needing labeled fraud examples — valuable precisely because labeled fraud data is scarce and biased toward *caught* fraud only.
- **Embeddings and representation learning** (Phase 6-7 preview): modern deep-learning-based unsupervised/self-supervised learning (contrastive learning, autoencoders) produces the embedding spaces underlying every RAG/vector-search system.
- **Genomics/bioinformatics** (occasionally overlapping with clinical work): hierarchical clustering is a standard tool for discovering gene expression patterns or patient subgroups without predefined labels.

---

## 11. Common Mistakes

- Applying k-means to non-spherical or very different-sized clusters and trusting the (systematically wrong) result without checking cluster shape assumptions.
- Not scaling features before distance-based clustering (k-means, DBSCAN) — a feature measured in the thousands (claim amount) will completely dominate a feature measured in single digits (age) unless standardized first.
- Over-interpreting t-SNE/UMAP plot *distances* between visually separated clusters as meaningful — these methods explicitly do not preserve global distance structure, only local neighborhoods.
- Choosing $k$ arbitrarily without any quantitative justification (silhouette score, elbow method) — a very common but easily avoidable weakness in unsupervised analysis writeups.

---

## 12. Best Practices (2026)

- Always scale/standardize features before any distance-based clustering algorithm.
- Use silhouette score (or the elbow method on within-cluster sum of squares) to justify $k$ selection quantitatively, not just visually.
- Prefer DBSCAN/HDBSCAN over k-means when cluster shapes are unknown or likely non-spherical, and when outlier detection is itself a goal.
- For dimensionality reduction feeding into a downstream supervised model, prefer PCA (linear, interpretable, invertible) over t-SNE/UMAP (non-linear, visualization-oriented, generally not meant to feed back into further modeling).

---

## 13. Exercises

**Easy:** Cluster a 2D synthetic dataset with 3 well-separated Gaussian blobs using k-means and verify the recovered centroids are close to the true generating means.
**Medium:** Compute silhouette scores for $k=2$ through $k=6$ on a dataset with a known true $k=4$ structure, and confirm the silhouette score peaks near the correct value.
**Hard:** Implement the from-scratch EM algorithm (Section 6) for a 1D two-component mixture and verify convergence by tracking the log-likelihood increasing monotonically across iterations.
**Mathematical:** Prove that the M-step's centroid update (cluster mean) is exactly the minimizer of within-cluster sum-of-squared distances, using basic calculus (Phase 3 Lesson 2).
**Coding:** Implement k-means++ initialization from scratch and empirically compare its convergence quality (final within-cluster sum of squares) against naive random initialization across multiple random seeds.

---

## 14. Mini Project

Build a **policyholder segmentation and risk-regime analysis tool**: simulate (or use real, if available) multi-dimensional policyholder data (age, region, claim history, tenure), apply PCA to reduce to the top principal components explaining 90%+ of variance, cluster the reduced representation using both k-means (with silhouette-justified $k$) and DBSCAN, compare which method produces more actionable/interpretable segments, and profile each discovered segment's characteristics (average age, claim frequency/severity) as a business-facing summary table.

---

## 15. Interview Preparation

- Explain the k-means algorithm and why it can converge to different results depending on initialization.
- What is the EM algorithm, and what problem does it solve that direct MLE cannot?
- How would you choose the number of clusters $k$ without any ground truth labels?
- When would you prefer DBSCAN over k-means, and what are DBSCAN's own limitations?

---

## 16. Summary

Unsupervised learning finds structure without labels: k-means and its EM-algorithm generalization (Gaussian Mixture Models) partition data via alternating optimization, silhouette scores provide a rare quantitative evaluation tool in the absence of ground truth, and PCA (Phase 3 Lesson 1's eigendecomposition, now applied practically) reduces dimensionality while preserving maximum variance. These techniques — clustering for segmentation, density estimation for anomaly detection, dimensionality reduction for tractability — are foundational both for classical tabular analysis and for the embedding-space reasoning that underlies Phase 6-7's entire LLM/RAG stack.

---

## 17. References

- Bishop, C. — *Pattern Recognition and Machine Learning*, Chapter 9 (EM algorithm, GMMs)
- Lloyd, S. — "Least Squares Quantization in PCM" (1982, the original k-means paper)
- Arthur & Vassilvitskii — "k-means++: The Advantages of Careful Seeding" (2007)
- Ester et al. — "A Density-Based Algorithm for Discovering Clusters" (1996, the original DBSCAN paper)
- van der Maaten & Hinton — "Visualizing Data using t-SNE" (2008)
