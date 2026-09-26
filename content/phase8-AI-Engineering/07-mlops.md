# Phase 8 · Lesson 7 — MLOps

> Prerequisite: MLflow (Lesson 6), CI/CD (Lesson 4), Docker/Kubernetes (Lessons 2–3)

---

## 1. Introduction

### What is MLOps?
The set of practices, tooling, and organizational discipline applying DevOps principles (Lesson 4's CI/CD, automation, monitoring) specifically to the unique lifecycle of machine learning systems — which, unlike traditional software, must additionally manage data versioning, model training reproducibility, model quality drift over time, and the tight coupling between code, data, and model artifacts. This lesson synthesizes and extends Lessons 1-6 into a coherent, end-to-end operational discipline.

### Why does it exist?
ML systems fail in ways traditional software doesn't: a model's *code* can be unchanged while its real-world performance silently degrades because the data distribution shifted (Section 2's data/concept drift) — a failure mode with no direct analog in traditional software engineering, where correctness is typically a more static property. MLOps exists specifically to catch and manage these ML-specific failure modes systematically, extending traditional DevOps practices rather than replacing them.

### Historical background
"MLOps" as a named discipline crystallized around 2018-2020, as organizations that had successfully adopted DevOps practices for traditional software found those practices insufficient for the additional complexity of production ML systems — Google's influential 2015 paper "Hidden Technical Debt in Machine Learning Systems" (Sculley et al.) is widely credited with first systematically articulating why ML systems accumulate unique forms of technical debt (Phase 1 Lesson 8's concept, now applied specifically to ML) beyond what traditional software engineering discipline alone addresses.

### Real-world motivation
Every technique from Phases 4-7 (a trained XGBoost model, a fine-tuned LLM, a RAG pipeline) eventually needs ongoing operational management — retraining schedules, drift detection, rollback procedures — not just a one-time successful deployment; MLOps is the discipline ensuring these systems remain reliable and performant over their entire operational lifetime, not just at initial launch.

---

## 2. Theory

### Data drift vs. concept drift
- **Data drift (covariate shift)**: the distribution of input features $P(X)$ changes over time, even if the underlying relationship $P(Y|X)$ hasn't (e.g., your actuarial model's policyholder age distribution shifts as your customer base ages, but age's relationship to mortality risk remains the same).
- **Concept drift**: the underlying relationship $P(Y|X)$ itself changes (e.g., a genuinely new risk factor emerges, or regulatory changes alter claim patterns) — a more fundamental shift requiring model retraining, not just monitoring adjustment, since the model's learned relationships are now genuinely outdated.

### The ML-specific technical debt sources (directly extending Phase 1 Lesson 8)
Sculley et al.'s influential taxonomy identifies ML-specific debt sources beyond ordinary code debt: **entanglement** (changing one feature's preprocessing can silently affect all others, since ML models don't have traditional software's clean modular boundaries), **hidden feedback loops** (a model's predictions can influence the very data it's later trained on — e.g., a fraud model's flags influence which transactions get investigated, which then influences what "confirmed fraud" labels exist for future training), and **pipeline jungles** (ad hoc data preprocessing scripts accumulating without clear ownership or documentation, directly Phase 2's data cleaning discipline gone wrong at organizational scale).

### The feature store pattern
A **feature store** centralizes feature computation logic (Phase 2 Lesson 6) so the *exact same* feature engineering code runs identically at training time and serving time — directly preventing **train/serve skew** (a subtle, historically common bug class where slightly different feature computation between training and production silently degrades model performance) by ensuring there's only one authoritative implementation of each feature, not two independently-maintained copies.

### Automated retraining pipelines
Rather than manually retraining models on an ad hoc basis, mature MLOps practice establishes automated pipelines that retrain on a schedule (or trigger-based on detected drift), automatically evaluate the new model against the current production model using Lesson 4/6's statistically-rigorous quality gates, and only promote the new model if it genuinely, significantly outperforms the current one — directly extending this curriculum's CI/CD (Lesson 4) and MLflow (Lesson 6) content into a fully automated, ongoing lifecycle rather than a one-time deployment event.

---

## 3. Mathematical Foundations

### Detecting data drift statistically
Comparing a reference (training-time) feature distribution against a current (production) distribution can use the **Kolmogorov-Smirnov test** (for continuous features) or the **Population Stability Index (PSI)**, a common practitioner metric:

$$
\text{PSI} = \sum_{i} (p_i - q_i) \ln\left(\frac{p_i}{q_i}\right)
$$

where $p_i, q_i$ are the proportions of observations falling in bin $i$ for the reference and current distributions respectively — directly a KL-divergence-flavored quantity (Phase 3 Lesson 6), with common practitioner thresholds (PSI < 0.1: no significant shift; 0.1-0.25: moderate shift, investigate; > 0.25: significant shift, likely requiring retraining).

### Concept drift detection via performance monitoring
Since concept drift changes $P(Y|X)$, it's often detected indirectly by monitoring the model's *actual* prediction performance over time (once true labels become available, which may be delayed — e.g., actual mortality outcomes take years to materialize) — a statistical process control approach (tracking a performance metric over time, flagging when it falls outside expected control limits, directly reusing Phase 3 Lesson 4's hypothesis-testing framework applied to a *sequence* of observations rather than a single comparison).

### The cost of retraining frequency (a genuine tradeoff)
More frequent retraining better tracks genuine drift but incurs real compute cost (Lesson 10) and risk (each retraining is itself a deployment event with its own failure risk, Lesson 4) — the optimal retraining cadence balances the expected cost of operating on a stale, drifted model against the cost and risk of frequent retraining, a genuinely quantifiable tradeoff informed by how quickly drift empirically tends to accumulate for a given application's specific data.

---

## 4. Algorithm — A Complete MLOps Lifecycle Loop (fully specified, synthesizing Lessons 1-7)

```
CONTINUOUS MONITORING (in production, Lesson 8):
1. LOG every prediction request's input features and (eventually, when available) true outcomes
2. PERIODICALLY compute data drift metrics (PSI, Section 3) comparing current input distribution
   against the training-time reference distribution
3. PERIODICALLY compute actual model performance metrics (once labels are available) and compare
   against expected/historical performance (Section 3's statistical process control)

TRIGGER FOR RETRAINING (either scheduled, or triggered by drift/performance degradation):
4. PULL the latest available labeled data (via a feature store, Section 2, ensuring train/serve consistency)
5. RETRAIN the model, tracked via MLflow (Lesson 6): log parameters, metrics WITH confidence intervals
6. EVALUATE the new model against the CURRENT production model using a statistically rigorous
   quality gate (Lesson 4's CI/CD gate logic, Lesson 6's registry comparison)
7. IF the new model doesn't meaningfully outperform: DO NOT promote; investigate why retraining
   didn't help (may indicate a deeper data/pipeline issue, not just "try again later")
8. IF the new model DOES meaningfully outperform: promote it through the registry (Lesson 6),
   deploy via the CI/CD pipeline (Lesson 4) to Kubernetes (Lesson 3)
9. CONTINUE monitoring the newly-deployed model, returning to step 1
```

---

## 5. Python Implementation

```python
"""mlops_core.py — data drift detection (PSI) and an automated retraining decision function"""
import numpy as np


def population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Section 3's PSI formula, implemented directly."""
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf   # ensure ALL current-data values fall into SOME bin

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_props = np.clip(ref_counts / len(reference), 1e-6, None)   # avoid log(0)/div-by-0
    cur_props = np.clip(cur_counts / len(current), 1e-6, None)

    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return psi


def interpret_psi(psi: float) -> str:
    if psi < 0.1:
        return "no significant drift"
    elif psi < 0.25:
        return "moderate drift -- investigate"
    else:
        return "significant drift -- retraining likely needed"


def should_trigger_retraining(feature_drift_scores: dict[str, float], performance_degradation: float,
                                 psi_threshold: float = 0.25, performance_threshold: float = 0.05) -> tuple[bool, str]:
    """Combines Section 2's data-drift AND concept-drift (performance) signals into one decision."""
    drifted_features = [f for f, psi in feature_drift_scores.items() if psi >= psi_threshold]
    if drifted_features:
        return True, f"Significant data drift detected in: {drifted_features}"
    if performance_degradation >= performance_threshold:
        return True, f"Performance degraded by {performance_degradation:.1%}, exceeding threshold"
    return False, "No retraining trigger detected"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    reference_age = rng.normal(45, 15, 5000)          # training-time age distribution
    current_age_stable = rng.normal(45, 15, 1000)      # production age distribution, UNCHANGED
    current_age_drifted = rng.normal(55, 15, 1000)      # production age distribution, SHIFTED (customer base aged)

    psi_stable = population_stability_index(reference_age, current_age_stable)
    psi_drifted = population_stability_index(reference_age, current_age_drifted)
    print(f"PSI (stable): {psi_stable:.4f} -> {interpret_psi(psi_stable)}")
    print(f"PSI (drifted): {psi_drifted:.4f} -> {interpret_psi(psi_drifted)}")

    trigger, reason = should_trigger_retraining(
        feature_drift_scores={"age": psi_drifted, "region_risk": 0.05},
        performance_degradation=0.02,
    )
    print(f"Retrain? {trigger} -- {reason}")
```

---

## 6. Build From Scratch

**A minimal feature store abstraction (Section 2, ensuring train/serve consistency):**
```python
class FeatureStore:
    """A radically simplified feature store: ONE authoritative implementation per feature,
    used IDENTICALLY at both training time and serving time -- directly preventing train/serve skew."""
    def __init__(self):
        self._feature_functions: dict[str, callable] = {}

    def register_feature(self, name: str, compute_fn: callable) -> None:
        self._feature_functions[name] = compute_fn

    def compute_features(self, raw_record: dict, feature_names: list[str]) -> dict:
        """Called IDENTICALLY during training data preparation AND during real-time serving --
        the single-source-of-truth property that prevents skew."""
        return {name: self._feature_functions[name](raw_record) for name in feature_names}


store = FeatureStore()
store.register_feature("age_band", lambda r: r["age"] // 10)
store.register_feature("tenure_years", lambda r: (r["current_year"] - r["policy_start_year"]))

# TRAINING TIME: compute features for historical records
training_record = {"age": 55, "current_year": 2020, "policy_start_year": 2015}
print("Training-time features:", store.compute_features(training_record, ["age_band", "tenure_years"]))

# SERVING TIME: the EXACT SAME function computes the SAME feature for a live request
serving_record = {"age": 55, "current_year": 2026, "policy_start_year": 2015}
print("Serving-time features:", store.compute_features(serving_record, ["age_band", "tenure_years"]))
```
Using this single `FeatureStore` instance (or, in production, a shared library/service both the training pipeline and the serving API import) guarantees that `age_band`'s definition can never silently diverge between training and serving — directly eliminating the specific, historically common train/serve skew bug class Section 2 describes.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `population_stability_index` (manual) | `evidently`, `whylogs` — dedicated ML monitoring libraries implementing PSI and many other drift-detection metrics, with dashboards and alerting built in |
| `FeatureStore` (toy) | Feast, Tecton — production-grade feature stores with real-time serving infrastructure, versioning, and integration with training pipelines at scale |
| Manual retraining trigger logic | Orchestration tools (Airflow, Prefect, Kubeflow Pipelines) for scheduling and managing automated retraining pipelines end to end |

---

## 8. Visual Explanations

**Data drift vs. concept drift (Section 2):**
```
DATA DRIFT (P(X) changes, P(Y|X) stable):        CONCEPT DRIFT (P(Y|X) itself changes):
  Training: ages centered at 45                     Training: age 60+ = high risk (stable relationship)
  Production: ages centered at 55                    Production: a NEW factor changes what "high risk" means
  (the age-risk RELATIONSHIP is still valid,          (the model's LEARNED relationship is now genuinely
   just applied to a shifted population)              outdated -- retraining is necessary, not just monitoring)
```

**The complete MLOps lifecycle loop (Section 4):**
```
   ┌──────────────────────────────────────────────────────────┐
   │                                                            │
   ▼                                                            │
Monitor (drift + performance) ──▶ Trigger? ──NO──▶ (continue monitoring)
   │                                  │
   │                                 YES
   │                                  ▼
   │                          Retrain (MLflow-tracked, Lesson 6)
   │                                  │
   │                                  ▼
   │                    Evaluate vs current PROD (statistically rigorous gate)
   │                                  │
   │                       improved? ─┴─ NOT improved? ──▶ investigate, DON'T promote
   │                          │
   │                         YES
   │                          ▼
   │                Promote + Deploy (Lesson 4's CI/CD, Lesson 3's K8s)
   └──────────────────────────┘
```

---

## 9. Practical Examples

**Simple:** compute PSI (Section 5) between two synthetic distributions with varying degrees of shift, confirming the metric correctly scales with the magnitude of shift.
**Medium:** implement the feature store pattern (Section 6) for 2-3 features and verify identical computation results when "training-time" and "serving-time" code paths both call the same registered functions.
**Real-world:** instrument your Phase 8 Lesson 1 FastAPI actuarial API to log incoming request features, periodically compute PSI against your training data's reference distribution, and trigger an alert (or automated retraining pipeline invocation) when drift exceeds your chosen threshold.

---

## 10. Real Industry Use Cases

- **Every mature ML organization's production model monitoring**: PSI-style drift detection (or similar) is standard practice for catching silent model degradation before it causes real business harm.
- **Feature stores at scale** (Uber's Michelangelo, Airbnb's Zipline): directly address the train/serve skew problem this lesson covers, at organizational scale across many teams and models.
- **Automated retraining pipelines**: common at companies with high-velocity ML deployment (recommendation systems, fraud detection) where data distributions shift frequently enough that manual retraining cadence would be inadequate.
- **Regulated-industry model governance**: insurance/finance organizations (directly relevant to your domain) often have formal, audited requirements for monitoring deployed models' ongoing performance and documenting retraining decisions — exactly this lesson's discipline, with compliance stakes attached.

---

## 11. Common Mistakes

- Deploying a model once and never monitoring its ongoing production performance — the single most common MLOps failure mode, allowing silent drift-driven degradation to go unnoticed until it causes visible business harm.
- Maintaining separate, independently-evolving feature computation code for training versus serving — the classic train/serve skew bug, directly preventable via the feature store pattern (Section 2/6).
- Retraining on a fixed schedule regardless of whether genuine drift has occurred — wastes compute (Lesson 10) on unnecessary retraining while potentially still missing genuinely sudden drift that occurs between scheduled retraining events.
- Promoting every retrained model automatically without a genuine quality gate — risking promotion of a model that's actually no better (or worse) than the current production version due to training-run noise.

---

## 12. Best Practices (2026)

- Implement systematic drift monitoring (PSI or equivalent) for all production models' input features, not just final-output performance metrics, since input drift often precedes and helps explain later performance degradation.
- Use a genuine feature store (or at minimum, shared library code) to guarantee identical feature computation between training and serving.
- Combine scheduled and drift-triggered retraining rather than relying purely on one or the other.
- Always gate model promotion on statistically rigorous evaluation (Lessons 4, 6) comparing the new model against the current production model, not just against a fixed absolute threshold.

---

## 13. Exercises

**Easy:** Compute PSI between two synthetic distributions with a small, moderate, and large shift, and verify your interpretation function (Section 5) correctly categorizes each.
**Medium:** Implement the feature store pattern (Section 6) with at least 3 registered features and write a test verifying training-time and serving-time computation produce identical results for the same underlying record.
**Hard:** Build a complete automated retraining decision pipeline: monitor simulated incoming data for drift, trigger retraining when a threshold is exceeded, evaluate the new model against the current one using a statistically rigorous gate (Lesson 6), and only promote if genuinely improved.
**Mathematical:** Derive why PSI is structurally similar to (though not identical to) KL divergence (Phase 3 Lesson 6), and discuss why practitioners often prefer PSI's specific formulation and interpretability thresholds over raw KL divergence for drift monitoring.
**Coding:** Integrate the Section 5 drift-detection logic into your Phase 8 Lesson 1 FastAPI application as a background task, periodically logging drift metrics and triggering an alert (e.g., a log message or webhook call) when thresholds are exceeded.

---

## 14. Mini Project

Build a **complete MLOps lifecycle system for your actuarial mortality model**: implement a feature store ensuring training/serving consistency for your engineered features (Phase 2 Lesson 6), instrument production request logging and periodic PSI-based drift monitoring, build an automated retraining pipeline (triggered by either a schedule or detected drift) that retrains via MLflow (Lesson 6), evaluates the new model against the current production version using a statistically rigorous quality gate, and promotes/deploys automatically (via Lesson 4's CI/CD and Lesson 3's Kubernetes) only when genuinely improved — a complete, closed-loop MLOps system synthesizing this entire phase's content.

---

## 15. Interview Preparation

- Explain the difference between data drift and concept drift, and how you'd detect each.
- What is the Population Stability Index, and how do you interpret its typical threshold values?
- What is a feature store, and what specific bug class does it prevent?
- Design an automated retraining pipeline for a production model, including how you'd decide when to trigger retraining and when to promote a newly retrained model.

---

## 16. Summary

MLOps extends DevOps discipline (Lesson 4's CI/CD, Lessons 2-3's containerization/orchestration) to address ML systems' unique operational challenges: data and concept drift (detected via PSI or similar statistical distance metrics, directly extending Phase 3 Lesson 6's information theory), train/serve skew (prevented via the feature store pattern's single-source-of-truth feature computation), and the need for automated, statistically-rigorously-gated retraining pipelines rather than one-time deployment. This lesson synthesizes Lessons 1-6 into a complete, closed operational loop — monitor, detect drift, retrain, evaluate, promote, redeploy, and continue monitoring — the discipline that keeps every model from Phases 4-7 reliable and performant across its entire production lifetime, not just at initial launch.

---

## 17. References

- Sculley et al. — "Hidden Technical Debt in Machine Learning Systems" (2015, the foundational MLOps-motivating paper)
- Google Cloud's "MLOps: Continuous delivery and automation pipelines in machine learning" whitepaper
- `evidently` and `whylogs` official documentation (ML monitoring libraries)
- Feast (feature store) official documentation
