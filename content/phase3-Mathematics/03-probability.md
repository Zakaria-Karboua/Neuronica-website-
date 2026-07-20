# Phase 3 · Lesson 3 — Probability

> Prerequisite: Linear Algebra, Calculus (Lessons 1–2). Given your survival analysis/EVT coursework, this lesson emphasizes the ML-modeling framing of concepts you've likely already proven rigorously.

---

## 1. Introduction

### What is probability theory?
The mathematical formalization of uncertainty — a rigorous framework (Kolmogorov's axioms, 1933) for reasoning about random events, distributions, and how information updates belief. Every ML model that outputs anything resembling "confidence" (classification probabilities, generative model sampling, uncertainty quantification) is built directly on this formalism.

### Why does it exist?
Real-world data is never perfectly deterministic — measurement noise, inherent randomness (mortality timing, claim occurrence), and incomplete information all require a principled way to reason under uncertainty rather than pretending the world is fully predictable. Probability theory provides that principled language.

### Historical background
Formal probability traces to Pascal and Fermat's correspondence on gambling problems (1654); rigorous measure-theoretic foundations came from Kolmogorov (1933), unifying discrete and continuous probability under one axiomatic framework. Bayesian inference (Bayes, 1763; Laplace's later development) provides the alternative/complementary framework for updating beliefs given evidence — both traditions are alive and essential in modern ML (frequentist statistics, Phase 4's evaluation metrics vs. Bayesian deep learning, probabilistic generative models in Phase 7).

### Real-world motivation
Your XGBoost mortality model outputs a probability; a well-calibrated model's probability should genuinely mean what it says (Phase 4's calibration lesson). LLMs (Phase 6) are literally probability distributions over the next token, sampled at generation time. None of this is metaphorical — it's the same probability theory covered here.

---

## 2. Theory

### Foundational definitions
- **Sample space** $\Omega$: the set of all possible outcomes.
- **Random variable** $X: \Omega \to \mathbb{R}$: a function mapping outcomes to numbers.
- **Probability distribution**: describes how probability mass/density is spread over $X$'s possible values — a probability mass function (PMF) for discrete $X$, probability density function (PDF) for continuous $X$.
- **Cumulative distribution function (CDF)**: $F(x) = P(X \le x)$.

### Key distributions and where they arise in ML
| Distribution | Typical ML use |
|---|---|
| Bernoulli/Binomial | binary classification outcomes |
| Categorical/Multinomial | multi-class classification, next-token prediction (Phase 6) |
| Gaussian (Normal) | noise models, weight initialization, many statistical tests (Phase 3 Lesson 4) |
| Poisson | claim-count modeling (directly your actuarial domain) |
| Exponential | time-to-event/survival modeling, waiting times |
| Beta | modeling probabilities themselves (Bayesian priors for Bernoulli parameters) |

### Conditional probability and Bayes' theorem
$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \qquad P(A|B) = \frac{P(B|A)P(A)}{P(B)} \quad \text{(Bayes' theorem)}
$$
Bayes' theorem is the mathematical engine of updating belief given new evidence — foundational to Bayesian statistics, spam filters (Naive Bayes), and conceptually to how you should interpret any diagnostic test's predictive value (a classic, often misunderstood application: a positive test result's true meaning depends heavily on the base rate $P(A)$, not just the test's sensitivity).

### Independence and conditional independence
$X$ and $Y$ are independent if $P(X, Y) = P(X)P(Y)$ — knowing one gives no information about the other. **Conditional independence** ($X \perp Y \mid Z$) is subtler and central to graphical models and Naive Bayes' core (often false-but-useful) assumption.

---

## 3. Mathematical Foundations

### Expectation and variance
$$
E[X] = \sum_x x \cdot P(X=x) \quad \text{(discrete)}, \qquad E[X] = \int x f(x)\,dx \quad \text{(continuous)}
$$
$$
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$$
Linearity of expectation, $E[aX + bY] = aE[X] + bE[Y]$, holds **regardless of independence** — a frequently underappreciated fact that simplifies enormous amounts of probabilistic reasoning (e.g., expected total claims across policyholders, even if claims are correlated).

### Law of Large Numbers and Central Limit Theorem (formalized)
$$
\bar{X}_n \xrightarrow{p} \mu \quad \text{(LLN: sample mean converges to true mean)}
$$
$$
\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2) \quad \text{(CLT: properly scaled, converges to Gaussian)}
$$
The CLT is *why* Gaussian assumptions are so pervasive in statistics (Phase 3 Lesson 4) despite most real data being non-Gaussian — aggregated quantities (means, sums) tend toward normality regardless of the underlying distribution, under fairly general conditions (finite variance).

### Maximum Likelihood Estimation (MLE) — the unifying estimation principle
Given i.i.d. data $x_1, \dots, x_n$ from a distribution with unknown parameter $\theta$, the likelihood is:
$$
L(\theta) = \prod_{i=1}^n f(x_i; \theta), \qquad \hat{\theta}_{MLE} = \arg\max_\theta L(\theta) = \arg\max_\theta \sum_i \log f(x_i; \theta)
$$
Taking the log (log-likelihood) converts a product into a sum — both for numerical stability (products of many small probabilities underflow to zero) and analytical tractability (derivatives of sums are easier than derivatives of products). **This is the exact framework that derives most ML loss functions**: minimizing cross-entropy loss for classification is *equivalent* to maximum likelihood estimation under a categorical/Bernoulli model — not a coincidence, but the same mathematical object viewed two ways.

### Bayesian inference formalized
$$
\underbrace{P(\theta | D)}_{\text{posterior}} \propto \underbrace{P(D|\theta)}_{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}
$$
Unlike MLE (a single point estimate), Bayesian inference maintains a full distribution over $\theta$, naturally quantifying uncertainty — directly relevant to actuarial reserving (credibility theory is fundamentally Bayesian: blending a company's own limited claims experience with broader industry priors, weighted by data volume).

---

## 4. Algorithm — Maximum Likelihood Estimation (worked example: Bernoulli)

```
GIVEN n coin flips, k of which are heads, ESTIMATE p (probability of heads):

Likelihood: L(p) = p^k * (1-p)^(n-k)
Log-likelihood: ℓ(p) = k*log(p) + (n-k)*log(1-p)

TAKE DERIVATIVE, SET TO ZERO:
  dℓ/dp = k/p - (n-k)/(1-p) = 0
  k(1-p) = (n-k)p
  k - kp = np - kp
  k = np
  p_hat = k/n     <- the intuitive answer, NOW DERIVED rigorously via MLE
```
This same procedure — write the log-likelihood, differentiate, set to zero, solve — is exactly how logistic regression's coefficients (Phase 4), Gaussian mixture model parameters, and countless other ML model parameters are formally derived (often requiring numerical optimization, Lesson 5, when no closed form exists).

---

## 5. Python Implementation

```python
"""probability_core.py — distributions, Bayes' theorem, and MLE in practice"""
import numpy as np
from scipy import stats


# --- Bayes' theorem applied: diagnostic test interpretation (a classic, high-stakes example) ---
def posterior_given_positive_test(prevalence: float, sensitivity: float, specificity: float) -> float:
    """P(disease | positive test) -- NOT the same as sensitivity, a very common misunderstanding."""
    p_disease = prevalence
    p_no_disease = 1 - prevalence
    p_positive_given_disease = sensitivity
    p_positive_given_no_disease = 1 - specificity   # false positive rate

    p_positive = (p_positive_given_disease * p_disease) + (p_positive_given_no_disease * p_no_disease)
    return (p_positive_given_disease * p_disease) / p_positive


# Even a 99%-sensitive, 95%-specific test can have a LOW positive predictive value if disease is rare
print(posterior_given_positive_test(prevalence=0.001, sensitivity=0.99, specificity=0.95))
# ~0.019 -- only ~2% of positive results are true positives, when the base rate is very low!


# --- MLE fitting via scipy (Poisson, directly relevant to actuarial claim-count modeling) ---
claim_counts = np.array([0, 1, 0, 2, 1, 3, 0, 1, 1, 2])
lambda_mle = claim_counts.mean()    # for Poisson, the MLE of lambda IS simply the sample mean (derivable via Section 4's method)
print(f"MLE lambda: {lambda_mle}")

fitted_poisson = stats.poisson(mu=lambda_mle)
print("P(exactly 2 claims):", fitted_poisson.pmf(2))


# --- Simulating the Central Limit Theorem empirically ---
rng = np.random.default_rng(0)
raw_samples = rng.exponential(scale=2.0, size=(10_000, 30))   # 10,000 experiments, each averaging 30 exponential draws
sample_means = raw_samples.mean(axis=1)
print(f"Sample means distribution: mean={sample_means.mean():.3f}, std={sample_means.std():.3f}")
# Despite exponential being heavily right-skewed, sample_means will look approximately NORMAL (CLT in action)
```

---

## 6. Build From Scratch

**A minimal Bayesian updater (Beta-Bernoulli conjugate model, foundational to credibility theory):**
```python
class BetaBernoulli:
    """Bayesian belief about a Bernoulli probability p, updated one observation at a time.
    Directly analogous to actuarial credibility theory: blending prior belief with observed data."""
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha = alpha_prior   # prior "pseudo-successes"
        self.beta = beta_prior     # prior "pseudo-failures"

    def update(self, observed_success: bool) -> None:
        if observed_success:
            self.alpha += 1
        else:
            self.beta += 1

    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, level: float = 0.95):
        from scipy.stats import beta as beta_dist
        lower = beta_dist.ppf((1 - level) / 2, self.alpha, self.beta)
        upper = beta_dist.ppf(1 - (1 - level) / 2, self.alpha, self.beta)
        return lower, upper


model = BetaBernoulli(alpha_prior=2, beta_prior=8)   # weak prior belief: ~20% success rate
for outcome in [True, True, False, True, True, True]:
    model.update(outcome)
print(f"Posterior mean: {model.posterior_mean():.3f}")
print(f"95% credible interval: {model.credible_interval()}")
```
As more data arrives, the posterior mean shifts from the prior toward the observed data's empirical rate — with the *speed* of that shift governed exactly by how much prior "pseudo-data" ($\alpha + \beta$) versus real data you have — the precise mathematical content of actuarial credibility theory (Bühlmann credibility is a direct generalization of this exact mechanism).

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `BetaBernoulli` | `PyMC`/`Stan`/`NumPyro` — full probabilistic programming languages for arbitrary Bayesian models |
| Manual MLE derivation | `scipy.stats.<dist>.fit()` — numerical MLE fitting for dozens of built-in distributions |
| Manual Bayes' theorem | Same formula used directly in production diagnostic/fraud-detection systems — this genuinely IS the production approach for simple cases |

---

## 8. Visual Explanations

**Bayes' theorem as belief updating:**
```
PRIOR belief about θ           LIKELIHOOD of observed data       POSTERIOR belief
     ___                              given θ                          ___
    /   \          ×          (peaks where data is likely)   =    (narrower, shifted
   /     \                                                          toward the data)
  /       \                        ___
 /         \                      /   \
────────────                    ─/─────\──                    ────/‾\─────
        θ                            θ                              θ
```

**CLT: skewed individual distribution -> normal distribution of sample means:**
```
Individual claims (skewed):        Sample MEANS of many claims (normal-ish):
█▄                                        ▄██▄
██▄▄                                    ▄██████▄
████▄▄▄▄▄▄                            ▄██████████▄
(long right tail)                    (symmetric, bell-shaped)
```

---

## 9. Practical Examples

**Simple:** compute the probability of at least one claim across 5 independent policyholders, each with 10% individual claim probability.
**Medium:** fit a Poisson distribution to claim-count data via MLE and compute the probability of observing more than 3 claims next year.
**Real-world:** apply Bayes' theorem to properly interpret a fraud-detection model's alert — given the model's sensitivity/specificity and the true (low) base rate of fraud, compute the actual probability that a flagged claim is genuinely fraudulent (directly mirroring the diagnostic-test example, a very common real misinterpretation in fraud/risk operations).

---

## 10. Real Industry Use Cases

- **Actuarial credibility theory** (Bühlmann, Bayesian credibility): directly the Beta-Bernoulli/conjugate-prior framework from Section 6, used to blend limited company-specific claims experience with broader industry-wide priors.
- **Spam/fraud detection**: Naive Bayes classifiers directly apply Bayes' theorem with a (deliberately simplifying) conditional independence assumption across features.
- **LLM token generation** (Phase 6): the entire generation process is sampling from a learned categorical probability distribution over the vocabulary at each step.
- **A/B testing and clinical trials**: Bayesian A/B testing frameworks (increasingly popular over purely frequentist approaches at companies like Booking.com, Spotify) directly apply posterior-updating logic.

---

## 11. Common Mistakes

- **Confusing $P(A|B)$ with $P(B|A)$** ("prosecutor's fallacy") — a positive test result's predictive value critically depends on the base rate, not just sensitivity, as Section 5's worked example makes concrete.
- Assuming independence when features are actually correlated (Naive Bayes' core simplifying assumption, sometimes fine in practice, sometimes a serious modeling error).
- Treating MLE point estimates as certain when sample sizes are small — a from-scratch Bayesian approach (Section 6) naturally exposes this uncertainty via the posterior's width, where pure MLE gives a false sense of precision.
- Misapplying the CLT to a small sample size when the underlying distribution is heavily skewed or heavy-tailed (relevant to your EVT coursework — the CLT's normal approximation converges much more slowly, or the classical CLT doesn't apply as intended, for heavy-tailed/infinite-variance distributions).

---

## 12. Best Practices (2026)

- Always state your model's underlying probabilistic assumptions explicitly (independence, distributional family) — most ML/statistical bugs trace back to violated assumptions, not incorrect arithmetic.
- Use Bayesian methods (even lightweight conjugate-prior versions like Section 6) when data is scarce and prior domain knowledge is genuinely informative — directly relevant to actuarial credibility applications with thin claims experience.
- Prefer `scipy.stats`/`PyMC` for any non-trivial distribution fitting over hand-deriving MLE formulas, but understand the derivation (Section 4) well enough to sanity-check the library's output.
- When communicating probabilistic model outputs to stakeholders, always frame them relative to base rates (Section 5's Bayes example) — raw sensitivity/specificity numbers without base-rate context routinely mislead decision-makers.

---

## 13. Exercises

**Easy:** Compute, by hand, the probability of rolling at least one six in 4 rolls of a fair die.
**Medium:** Derive the MLE for a Gaussian distribution's mean and variance from i.i.d. samples (analogous to Section 4's Bernoulli derivation).
**Hard:** Extend the `BetaBernoulli` model (Section 6) to a full Bühlmann credibility calculation, comparing the Bayesian posterior mean against the classical actuarial credibility-weighted estimate for the same synthetic claims data.
**Mathematical:** Prove that for i.i.d. Poisson-distributed data, the MLE of the rate parameter $\lambda$ is exactly the sample mean.
**Coding:** Implement a Monte Carlo simulation empirically verifying the Central Limit Theorem for sample means drawn from a heavily skewed (e.g., Pareto) distribution, and discuss how quickly (or slowly) convergence to normality occurs compared to an exponential distribution.

---

## 14. Mini Project

Build a **Bayesian claims-frequency credibility model**: given synthetic claims data for several small policyholder cohorts (some with very little data, some with substantial history), implement a Poisson-Gamma conjugate Bayesian model (the standard actuarial credibility framework, a direct generalization of Section 6's Beta-Bernoulli), compute posterior claim-frequency estimates for each cohort, and compare them against naive per-cohort MLE (sample-mean-only) estimates — demonstrating how the Bayesian approach appropriately shrinks small-sample cohorts toward the overall population rate while leaving large-sample cohorts largely unaffected.

---

## 15. Interview Preparation

- Explain Bayes' theorem and walk through the classic "positive diagnostic test" base-rate example.
- Derive the MLE for a simple distribution (Bernoulli, Poisson, or Gaussian) from first principles.
- Why does the Central Limit Theorem matter for statistical inference, and when might it not apply well?
- Explain the difference between a frequentist point estimate and a Bayesian posterior distribution, and when you'd prefer one over the other.

---

## 16. Summary

Probability theory gives ML its entire vocabulary for reasoning under uncertainty: distributions model randomness in outcomes, Bayes' theorem formalizes belief updating given evidence (with the base-rate trap being the single most important practical lesson), Maximum Likelihood Estimation is the unifying principle that derives most ML loss functions from a probabilistic model, and the Central Limit Theorem explains why Gaussian assumptions pervade classical statistics. Actuarial credibility theory — likely already familiar from your coursework — is, when traced to its mathematical root, exactly Bayesian conjugate-prior updating.

---

## 17. References

- Wasserman, L. — *All of Statistics* (excellent bridge between probability theory and statistical practice)
- Bertsekas & Tsitsiklis — *Introduction to Probability*
- Klugman, Panjer, Willmot — *Loss Models* (the standard actuarial reference connecting probability theory directly to claims/credibility modeling)
- Gelman et al. — *Bayesian Data Analysis* (the definitive modern Bayesian statistics reference)
