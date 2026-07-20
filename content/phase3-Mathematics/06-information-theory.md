# Phase 3 · Lesson 6 — Information Theory

> Prerequisite: Probability, Optimization (Lessons 3, 5) — this lesson closes out Phase 3

---

## 1. Introduction

### What is information theory?
The mathematical study of quantifying information, uncertainty, and the fundamental limits of data compression and communication — founded by Claude Shannon's 1948 paper "A Mathematical Theory of Communication." In ML, its core objects (entropy, cross-entropy, KL divergence, mutual information) are not exotic — they are the literal loss functions used to train nearly every classifier and language model you will build.

### Why does it exist?
Shannon was solving a concrete engineering problem at Bell Labs: how much can a message be compressed, and how reliably can it be transmitted over a noisy channel? The resulting mathematics turned out to be a universal language for quantifying uncertainty and information content — equally applicable to describing a probability distribution's "spread" (entropy) or measuring how different two distributions are (KL divergence, cross-entropy), which is exactly what a classification loss function needs to do.

### Historical background
Shannon's 1948 paper is one of the most consequential single papers in applied mathematics — it simultaneously founded digital communication theory and provided the mathematical vocabulary later adopted wholesale by machine learning (cross-entropy loss, the objective function behind virtually every classifier and every LLM's training objective, is a direct import from this theory).

### Real-world motivation
When you train a classifier with "cross-entropy loss," you are literally minimizing an information-theoretic quantity measuring how well your predicted distribution matches the true one. When an LLM is trained via "next-token prediction," the training objective is, precisely, cross-entropy between the model's predicted next-token distribution and the actual next token — Phase 6 will make this completely explicit.

---

## 2. Theory

### Entropy — quantifying uncertainty
For a discrete random variable $X$ with distribution $p$:
$$
H(X) = -\sum_x p(x)\log p(x)
$$
Entropy is maximized when $p$ is uniform (maximum uncertainty — you have no idea which outcome will occur) and is zero when $p$ is a point mass (no uncertainty — you know exactly what will happen). Units depend on the log base: base 2 gives **bits**, natural log gives **nats** (the ML-conventional choice).

### Cross-entropy — comparing a true distribution to a predicted one
$$
H(p, q) = -\sum_x p(x)\log q(x)
$$
where $p$ is the true distribution and $q$ is your model's predicted distribution. **This is literally the standard classification loss function** — for a single labeled example (a one-hot true distribution $p$), cross-entropy simplifies to $-\log q(y_{\text{true}})$, the negative log-probability the model assigned to the correct class.

### KL Divergence — the "distance" between distributions
$$
D_{KL}(p \| q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p)
$$
KL divergence is always $\ge 0$ (Gibbs' inequality), equal to zero **only** when $p = q$ exactly. It is **not symmetric** ($D_{KL}(p\|q) \ne D_{KL}(q\|p)$ in general), so it's not a true mathematical "distance" (metric) despite behaving somewhat like one — a subtlety with real consequences in variational inference and generative model training (Phase 6-7), where the choice of direction ($D_{KL}(p\|q)$ vs. $D_{KL}(q\|p)$) produces qualitatively different behavior (mode-covering vs. mode-seeking).

### Mutual Information — quantifying shared information between variables
$$
I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)} = H(X) - H(X|Y)
$$
$I(X;Y)$ measures how much knowing $Y$ reduces uncertainty about $X$ (and vice versa — it's symmetric) — zero exactly when $X \perp Y$ (independent, Phase 3 Lesson 3). Used directly for feature selection (Phase 4): features with high mutual information with the target carry genuinely useful, potentially non-linear predictive signal that a simple correlation coefficient (Phase 2 Lesson 5) might completely miss.

---

## 3. Mathematical Foundations

### Why cross-entropy loss equals negative log-likelihood (the crucial unification)
For classification with true label $y$ and predicted probability $q(y)$ for the correct class, the cross-entropy loss for one example is:
$$
L = -\log q(y)
$$
This is *exactly* the negative log-likelihood term from Phase 3 Lesson 3's Maximum Likelihood Estimation framework. **Training a classifier with cross-entropy loss and training it via maximum likelihood estimation under a categorical model are the same optimization problem** — not a coincidence, not an analogy, but literally identical mathematics viewed from two historical traditions (information theory vs. statistics) that converge on the same objective function.

### KL divergence decomposition (why minimizing cross-entropy also minimizes KL divergence)
$$
D_{KL}(p\|q) = H(p,q) - H(p)
$$
Since $H(p)$ (the true distribution's own entropy) doesn't depend on your model's parameters, minimizing cross-entropy $H(p,q)$ with respect to model parameters is *equivalent* to minimizing $D_{KL}(p\|q)$ — training a classifier is, precisely, minimizing the KL divergence between the true and predicted distributions.

### Gibbs' inequality (proving $D_{KL} \ge 0$)
Using the fact that $\log$ is concave (Jensen's inequality):
$$
-D_{KL}(p\|q) = \sum_x p(x)\log\frac{q(x)}{p(x)} \le \log\left(\sum_x p(x)\frac{q(x)}{p(x)}\right) = \log\left(\sum_x q(x)\right) = \log(1) = 0
$$
so $D_{KL}(p\|q) \ge 0$, with equality iff $p=q$ almost everywhere (Jensen's equality condition, since $q(x)/p(x)$ must be constant). This inequality is the theoretical bedrock guaranteeing that minimizing cross-entropy loss is a *sound* objective — it has a well-defined global minimum exactly when predictions match reality.

### Perplexity — the ML-practitioner-friendly reframing of cross-entropy
$$
\text{Perplexity} = 2^{H(p,q)} \quad \text{(or } e^{H(p,q)}\text{ using nats)}
$$
A language model's perplexity (Phase 6) is literally an exponentiated cross-entropy — interpretable as "the model is, on average, as uncertain as if choosing uniformly among this many options" — a genuinely intuitive way to communicate an information-theoretic quantity to non-specialists.

---

## 4. Algorithm — Computing Entropy/Cross-Entropy/KL Divergence Numerically

```
GIVEN discrete distributions p, q (arrays summing to 1):

ENTROPY(p):
    H = 0
    FOR each x where p(x) > 0:      # 0*log(0) is defined as 0 by convention (limit), must handle explicitly
        H -= p(x) * log(p(x))
    RETURN H

CROSS_ENTROPY(p, q):
    H = 0
    FOR each x where p(x) > 0:
        H -= p(x) * log(q(x) + epsilon)   # epsilon avoids log(0) if q assigns zero probability to a true event
    RETURN H

KL_DIVERGENCE(p, q):
    RETURN CROSS_ENTROPY(p, q) - ENTROPY(p)
```
The `epsilon` safeguard in cross-entropy is not a minor implementation detail: if the true distribution has non-zero probability somewhere the model assigns exactly zero, cross-entropy is *mathematically infinite* — a real failure mode in practice (a model overconfidently assigning near-zero probability to the actual correct answer), which is exactly why label smoothing (Phase 6) and careful softmax numerical implementation exist.

---

## 5. Python Implementation

```python
"""information_theory_core.py"""
import numpy as np
from scipy.stats import entropy as scipy_entropy


def entropy(p: np.ndarray, base: float = np.e) -> float:
    p = p[p > 0]                      # 0 log 0 := 0, by convention/limit
    return -np.sum(p * np.log(p)) / np.log(base)


def cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    q_safe = np.clip(q, eps, 1.0)
    return -np.sum(p * np.log(q_safe))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    return cross_entropy(p, q, eps) - entropy(p, base=np.e)


def mutual_information(joint: np.ndarray) -> float:
    """joint: 2D array, joint[i,j] = P(X=i, Y=j)."""
    p_x = joint.sum(axis=1, keepdims=True)
    p_y = joint.sum(axis=0, keepdims=True)
    independent = p_x @ p_y
    nonzero = joint > 0
    return np.sum(joint[nonzero] * np.log(joint[nonzero] / independent[nonzero]))


# --- Classification example: cross-entropy loss on one prediction ---
true_label_onehot = np.array([0, 0, 1, 0])           # true class = index 2
predicted_probs = np.array([0.05, 0.10, 0.80, 0.05])  # softmax output, well-calibrated
loss = cross_entropy(true_label_onehot, predicted_probs)
print(f"Cross-entropy loss: {loss:.4f}")             # equals -log(0.80)

# --- Comparing a confident-wrong vs confident-right prediction ---
overconfident_wrong = np.array([0.01, 0.01, 0.01, 0.97])   # very confident, WRONG class
print("Overconfident wrong loss:", cross_entropy(true_label_onehot, overconfident_wrong))  # huge loss

# --- Mutual information for feature selection (a genuinely non-linear-relationship-aware alternative to correlation) ---
rng = np.random.default_rng(0)
x = rng.integers(0, 3, 5000)
y = (x % 2)                     # deterministic but NON-LINEAR relationship
joint_counts = np.zeros((3, 2))
for xi, yi in zip(x, y):
    joint_counts[xi, yi] += 1
joint_probs = joint_counts / joint_counts.sum()
print("Mutual information:", mutual_information(joint_probs))
print("Pearson correlation:", np.corrcoef(x, y)[0, 1])   # likely near ZERO despite a perfect deterministic relationship!
```

**Notes:** the final example is a direct, concrete instance of Phase 2 Lesson 5's Anscombe's-Quartet-style warning — mutual information correctly detects the deterministic (but non-linear/non-monotonic) relationship between $x$ and $y$ that Pearson correlation completely misses.

---

## 6. Build From Scratch

Section 5 already implements these information-theoretic quantities directly from their definitions (no simplification needed — the formulas *are* the implementation). The natural "build from scratch" extension is **Huffman coding**, the concrete algorithm that operationalizes entropy as an actual compression bound:

```python
import heapq
from collections import Counter

def huffman_code_lengths(frequencies: dict[str, int]) -> dict[str, int]:
    """Constructs a Huffman tree and returns each symbol's optimal code LENGTH (not the code itself)."""
    heap = [[freq, [[symbol, 0]]] for symbol, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1]:
            pair[1] += 1
        for pair in hi[1]:
            pair[1] += 1
        heapq.heappush(heap, [lo[0] + hi[0], lo[1] + hi[1]])
    return {symbol: length for symbol, length in heap[0][1]}

text = "aaaaabbbcc"
freqs = Counter(text)
lengths = huffman_code_lengths(freqs)
total_bits = sum(freqs[s] * lengths[s] for s in freqs)
theoretical_min_bits = len(text) * entropy(np.array(list(freqs.values())) / len(text), base=2)
print(f"Huffman total bits: {total_bits}, Shannon entropy lower bound: {theoretical_min_bits:.2f}")
```
This directly demonstrates **Shannon's source coding theorem**: entropy is the theoretical *lower bound* on average bits-per-symbol for any lossless encoding, and Huffman coding (built using exactly the min-heap structure from Phase 1 Lesson 4) achieves a result provably close to that bound — entropy stops being an abstract formula and becomes a literal, verifiable compression limit.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `entropy`/`cross_entropy`/`kl_divergence` | `scipy.stats.entropy` (supports KL divergence directly via the `qk` parameter), `torch.nn.CrossEntropyLoss` (numerically stable, fused with softmax, GPU-accelerated) |
| `huffman_code_lengths` | Real compression libraries (`zlib`, `gzip`) use more sophisticated adaptive/arithmetic coding, but Huffman remains foundational and is used directly in formats like DEFLATE |
| `mutual_information` | `sklearn.feature_selection.mutual_info_classif`/`mutual_info_regression` — handles continuous variables via k-NN-based estimation |

---

## 8. Visual Explanations

**Entropy as a function of a Bernoulli probability (maximum uncertainty at p=0.5):**
```
H(p)
 1.0 ┤          ___
     │       ╱      ╲
 0.5 ┤     ╱          ╲
     │   ╱              ╲
 0.0 ┼─╱──────────────────╲─
     0    0.25   0.5  0.75   1.0    p
   (certain)  (max uncertainty)  (certain)
```

**Cross-entropy loss as prediction confidence changes (true class probability on x-axis):**
```
Loss = -log(q_true)
 High │╲
      │ ╲
      │  ╲___
 Low  │      ╲______________
      └──────────────────────  q_true (predicted prob of correct class)
      0                      1.0
  (wrong+confident = HUGE loss;  right+confident = loss near 0)
```

---

## 9. Practical Examples

**Simple:** compute the entropy of a fair coin vs. a heavily biased coin, confirming the fair coin has maximum entropy.
**Medium:** compute cross-entropy loss for a batch of classifier predictions and verify it matches `sklearn.metrics.log_loss`.
**Real-world:** use mutual information (rather than Pearson correlation, per Phase 2 Lesson 5's leakage/relationship-detection lesson) as a feature-selection criterion on your actuarial dataset, specifically to catch non-linear feature-target relationships that correlation-based screening would miss.

---

## 10. Real Industry Use Cases

- **Every classifier and LLM training loop**: cross-entropy loss (equivalently, negative log-likelihood / KL divergence minimization) is the near-universal training objective (Phase 4-6).
- **Decision tree splitting criteria** (Phase 4): information gain (a direct application of entropy reduction) is one of the two standard criteria (alongside Gini impurity) for choosing tree splits.
- **Variational autoencoders and diffusion models** (Phase 7-adjacent generative modeling): KL divergence appears explicitly in the training objective (the "regularization" term balancing reconstruction fidelity against a prior distribution).
- **Language model evaluation**: perplexity (Section 3) remains a standard, widely reported LLM evaluation metric, directly derived from cross-entropy.

---

## 11. Common Mistakes

- Forgetting the `epsilon`/clipping safeguard when computing cross-entropy numerically, causing `log(0) = -inf` crashes when a model confidently assigns exactly zero probability to the true class.
- Treating KL divergence as symmetric — $D_{KL}(p\|q) \ne D_{KL}(q\|p)$ in general, and confusing the two produces materially different (mode-covering vs. mode-seeking) behavior in generative modeling contexts.
- Relying solely on Pearson correlation for feature selection and missing genuinely predictive non-linear relationships that mutual information would catch (Section 5's deterministic-but-uncorrelated example).
- Confusing entropy (a property of one distribution) with cross-entropy/KL divergence (properties comparing two distributions) — a common conceptual mix-up when first learning this material.

---

## 12. Best Practices (2026)

- Use framework-provided, numerically stable implementations (`torch.nn.CrossEntropyLoss`, which fuses softmax + cross-entropy for stability) rather than hand-computing softmax then cross-entropy separately in production training code.
- Consider mutual-information-based feature selection alongside (not instead of) correlation-based screening during EDA (Phase 2 Lesson 5) — they catch different relationship types.
- Understand perplexity as a standard LLM evaluation reporting convention (Phase 6) — being able to convert between cross-entropy (nats/bits) and perplexity fluently is expected practitioner knowledge.
- When implementing custom loss functions involving KL divergence (e.g., knowledge distillation, Phase 6), be deliberate about direction ($D_{KL}(p\|q)$ vs $D_{KL}(q\|p)$) since it changes the optimization's qualitative behavior.

---

## 13. Exercises

**Easy:** Compute the entropy of a 6-sided fair die's outcome distribution and compare it to a loaded die's.
**Medium:** Implement cross-entropy loss from scratch for a multi-class classification batch and verify it matches `torch.nn.functional.cross_entropy`.
**Hard:** Implement Huffman coding (Section 6) for a real text file, compute the theoretical entropy-based compression lower bound, and compare the actual achieved compression ratio.
**Mathematical:** Prove that $D_{KL}(p\|q) \ne D_{KL}(q\|p)$ in general using a concrete small example with 3 outcomes.
**Coding:** Implement mutual-information-based feature selection and compare its selected features against Pearson-correlation-based selection on a synthetic dataset containing both linear and non-linear (but zero-correlation) feature-target relationships.

---

## 14. Mini Project

Build a **feature-selection tool comparing correlation vs. mutual information** on your actuarial claims dataset: compute both Pearson correlation and mutual information between every feature and the target, rank features by each method, identify any features where the two methods disagree substantially, visualize (Phase 2 Lesson 4) the specific relationship for each disagreement case, and write a short recommendation on which features a correlation-only screening process would have wrongly discarded.

---

## 15. Interview Preparation

- Explain entropy intuitively, and why a uniform distribution has maximum entropy.
- Derive why minimizing cross-entropy loss is equivalent to maximum likelihood estimation.
- What's the difference between KL divergence and cross-entropy, and why is KL divergence not symmetric?
- Why might mutual information be preferred over correlation for feature selection in some cases?

---

## 16. Summary

Information theory supplies the exact mathematical vocabulary of ML's most common training objective: cross-entropy loss is negative log-likelihood is KL-divergence-minimization, three historically distinct framings of the identical optimization problem. Entropy quantifies uncertainty, KL divergence quantifies distributional mismatch (asymmetrically), and mutual information generalizes correlation to catch non-linear dependence — closing out Phase 3's mathematical foundation with the exact concepts Phase 4's model evaluation, Phase 5's neural network loss functions, and Phase 6's LLM training objectives will all build directly upon.

---

## 17. References

- Shannon, C.E. — "A Mathematical Theory of Communication" (1948, the founding paper)
- Cover, T. & Thomas, J. — *Elements of Information Theory* (the standard modern textbook)
- MacKay, D. — *Information Theory, Inference, and Learning Algorithms* (free online, ML-oriented framing)
- Goodfellow, Bengio, Courville — *Deep Learning*, Chapter 3 (Information Theory section, directly connecting to neural network loss functions)
