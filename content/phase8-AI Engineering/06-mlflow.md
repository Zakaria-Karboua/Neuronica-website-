# Phase 8 · Lesson 6 — MLflow

> Prerequisite: Phase 4 (Supervised Learning, Model Evaluation), Docker (Lesson 2)

---

## 1. Introduction

### What is MLflow?
An open-source platform (Databricks, 2018) for managing the machine learning lifecycle: experiment tracking (recording parameters, metrics, and artifacts from every training run), model packaging (a standardized format for saving/loading models regardless of the framework used), a model registry (versioning and stage-managing models from development through production), and model serving — the most widely-adopted open-source tool addressing the "how do we keep track of all our models and experiments" problem.

### Why does it exist?
Without systematic tracking, ML development quickly becomes chaotic: dozens of training runs with slightly different hyperparameters, unclear which model version is actually deployed, and no reliable way to reproduce a specific past result — directly the kind of unmanaged complexity Phase 1 Lesson 8's software engineering principles exist to prevent, now applied specifically to the ML development lifecycle's unique artifacts (models, metrics, datasets) rather than just code.

### Historical background
MLflow emerged from Databricks' internal experience helping customers manage increasingly complex ML workflows, released as an open-source, framework-agnostic tool (working with scikit-learn, XGBoost, PyTorch, Hugging Face Transformers, and virtually any other ML library) specifically to avoid vendor lock-in to any single framework or cloud provider — a design philosophy that contributed significantly to its broad, sustained industry adoption through the 2020s.

### Real-world motivation
Every model you've trained across Phases 4-6 — an XGBoost classifier, a fine-tuned LLM — benefits from exactly this lesson's tracking discipline: without it, six months later you (or a teammate) may have no reliable record of which hyperparameters produced your best model, or how its performance compared to previous versions.

---

## 2. Theory

### Experiment tracking
Every training run is logged as a **run** within an **experiment** (a named grouping of related runs, e.g., "mortality-model-tuning") — each run records parameters (hyperparameters used), metrics (evaluation results, Phase 4 Lesson 3), and artifacts (the trained model file, plots, or any other output file) — creating a permanent, queryable record of every attempt, not just the final chosen model.

### The MLmodel format and framework-agnostic packaging
MLflow defines a standardized packaging format (an `MLmodel` file plus supporting artifacts) capturing not just the model's weights but also its expected input/output schema and the exact environment (dependencies) needed to run it — enabling a saved model to be loaded and served consistently regardless of which specific framework (scikit-learn, XGBoost, PyTorch) originally produced it, directly addressing Phase 1 Lesson 6's environment-reproducibility concerns at the model-artifact level.

### Model registry and stage transitions
The **Model Registry** provides a central, versioned catalog of models, with explicit **stages** (e.g., "Staging," "Production," "Archived") — a model version can be promoted from Staging to Production only through an explicit, tracked action, providing an audit trail of exactly which model version was serving production traffic at any point in time — directly relevant to Phase 8 Lesson 4's CI/CD pipeline (which stage transition can be automated as part of the deployment pipeline) and to regulatory/compliance requirements (Lesson 9's security, and general actuarial/insurance auditability needs).

### Reproducibility as MLflow's central design goal
Every logged run captures enough information (parameters, code version via Git commit hash — Phase 1 Lesson 5 — dependency versions, and the training data's identifying information) to, in principle, reproduce that exact result later — directly extending this curriculum's repeated emphasis (Phase 2's data cleaning, Phase 4's evaluation) on treating reproducibility as a first-class engineering requirement, not an afterthought.

---

## 3. Mathematical Foundations

MLflow is fundamentally a systems/tooling topic, but its value proposition connects directly to earlier statistical/evaluation concepts:

### Systematic hyperparameter search tracking
When performing hyperparameter tuning (grid search, random search, or Bayesian optimization) across many candidate configurations, MLflow's experiment tracking provides the complete dataset needed for the kind of rigorous comparison Phase 4 Lesson 3 demands — every configuration's cross-validated performance (with confidence intervals, Phase 3 Lesson 4) can be logged and later queried/compared systematically, rather than relying on memory or ad hoc spreadsheet tracking of which configuration performed best.

### Model comparison with proper statistical rigor
Comparing two logged runs' metrics naively (just looking at which number is bigger) repeats exactly the error Phase 4 Lesson 3 warns against — MLflow's tracking makes it straightforward to log not just a point-estimate metric but also its bootstrap confidence interval (Phase 3 Lesson 4) per run, enabling genuinely rigorous "is this actually better" comparisons across the full experiment history, not just the two most recent runs.

### Model registry stage-transition as a formalized quality gate
Directly extending Phase 8 Lesson 4's CI/CD quality-gate concept: promoting a model from "Staging" to "Production" in the registry can be gated on the same statistically-rigorous evaluation criteria (a confidence-interval-based threshold check, not a raw point-estimate comparison) developed in that lesson, with MLflow providing the systematic record-keeping infrastructure underneath such an automated gate.

---

## 4. Algorithm — The Full MLflow Experiment Tracking + Registry Workflow (fully specified)

```
DURING TRAINING (for every experiment/hyperparameter configuration tried):
1. START an MLflow run within a named experiment
2. LOG all hyperparameters used for this run
3. TRAIN the model
4. EVALUATE on a held-out validation set (Phase 4 Lesson 3's rigor: compute metric + confidence interval)
5. LOG all evaluation metrics (and their confidence intervals) for this run
6. LOG the trained model artifact itself (in MLflow's standardized format)
7. LOG any relevant plots/diagnostics (confusion matrix, ROC curve, feature importances) as artifacts
8. END the run

AFTER COMPARING RUNS (via the MLflow UI or programmatic query):
9. IDENTIFY the best-performing run according to your chosen, statistically-sound criteria
10. REGISTER that run's model into the Model Registry, creating a new model VERSION

PROMOTING TO PRODUCTION (often as part of Lesson 4's CI/CD pipeline):
11. RUN additional validation (staging smoke tests, an evaluation-gate check, Lesson 4's quality gate)
12. IF validation passes: TRANSITION the model version's stage to "Production"
    (this transition is LOGGED, creating a permanent audit trail of deployment history)
13. THE SERVING INFRASTRUCTURE (Lesson 1's FastAPI app, or a dedicated MLflow serving deployment)
    loads the CURRENT "Production"-staged model version
```

---

## 5. Python Implementation

```python
"""mlflow_core.py — experiment tracking, model registration, and stage promotion"""
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

mlflow.set_experiment("mortality-model-tuning")


def bootstrap_ci(y_true, y_pred, n_bootstrap=500, ci=0.95):
    """Directly reusing Phase 3 Lesson 4's bootstrap technique -- logged ALONGSIDE the point estimate."""
    rng = np.random.default_rng(0)
    scores = [roc_auc_score(y_true[idx], y_pred[idx])
              for idx in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_bootstrap))]
    return np.mean(scores), np.percentile(scores, (1 - ci) / 2 * 100), np.percentile(scores, (1 + ci) / 2 * 100)


def train_and_log_run(n_estimators: int, max_depth: int, X_train, y_train, X_val, y_val) -> str:
    with mlflow.start_run() as run:
        # --- log hyperparameters (Section 4, step 2) ---
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        model.fit(X_train, y_train)

        # --- evaluate WITH confidence interval (Section 3's statistical rigor) ---
        y_pred = model.predict_proba(X_val)[:, 1]
        mean_auc, lower_ci, upper_ci = bootstrap_ci(y_val, y_pred)
        mlflow.log_metrics({"auc_mean": mean_auc, "auc_lower_ci": lower_ci, "auc_upper_ci": upper_ci})

        # --- log the model artifact itself (Section 2's framework-agnostic packaging) ---
        mlflow.sklearn.log_model(model, "model")

        return run.info.run_id


def register_best_run(experiment_name: str, model_name: str) -> str:
    """Finds the best run by lower-CI-bound AUC (a CONSERVATIVE, statistically sound selection criterion)."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.auc_lower_ci DESC"])
    best_run_id = runs.iloc[0]["run_id"]

    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    return result.version


def promote_to_production(model_name: str, version: str) -> None:
    """The registry stage transition (Section 4, step 12) -- creates a PERMANENT, logged audit trail."""
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name=model_name, version=version, stage="Production")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    for n_est in [50, 100, 200]:
        for depth in [3, 5, 7]:
            train_and_log_run(n_est, depth, X_train, y_train, X_val, y_val)

    version = register_best_run("mortality-model-tuning", "mortality-classifier")
    promote_to_production("mortality-classifier", version)
    print(f"Promoted version {version} to Production.")
```

---

## 6. Build From Scratch

**A minimal, from-scratch experiment-tracking log (to demystify what MLflow's tracking store actually provides):**
```python
import json
import uuid
from pathlib import Path
from datetime import datetime

class SimpleExperimentTracker:
    """A radically simplified illustration of MLflow's core tracking mechanism: a structured,
    queryable log of every run's parameters, metrics, and artifact locations."""
    def __init__(self, storage_dir: str = "./simple_mlflow_store"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def start_run(self) -> str:
        run_id = str(uuid.uuid4())[:8]
        run_dir = self.storage_dir / run_id
        run_dir.mkdir(exist_ok=True)
        (run_dir / "meta.json").write_text(json.dumps({
            "run_id": run_id, "start_time": datetime.now().isoformat(), "params": {}, "metrics": {}
        }))
        return run_id

    def log_param(self, run_id: str, key: str, value) -> None:
        self._update_run(run_id, "params", key, value)

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        self._update_run(run_id, "metrics", key, value)

    def _update_run(self, run_id: str, section: str, key: str, value) -> None:
        meta_path = self.storage_dir / run_id / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta[section][key] = value
        meta_path.write_text(json.dumps(meta))

    def search_runs(self) -> list[dict]:
        """Enables querying/comparing runs -- the core value-add over scattered notebook cells."""
        runs = []
        for run_dir in self.storage_dir.iterdir():
            if run_dir.is_dir():
                runs.append(json.loads((run_dir / "meta.json").read_text()))
        return sorted(runs, key=lambda r: r["metrics"].get("auc_mean", 0), reverse=True)


tracker = SimpleExperimentTracker()
run_id = tracker.start_run()
tracker.log_param(run_id, "n_estimators", 100)
tracker.log_metric(run_id, "auc_mean", 0.87)
print(tracker.search_runs())
```
This makes explicit MLflow's essential value proposition: a structured, permanent, *queryable* record of every experiment — the from-scratch version above is genuinely close to what a minimal tracking system needs, but real MLflow adds artifact storage, a web UI, framework-specific model serialization (Section 2), and the Model Registry's stage-transition audit trail on top of this basic pattern.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `SimpleExperimentTracker` | MLflow's Tracking Server — web UI, artifact storage backends (S3, Azure Blob, GCS), SQL-backed metadata store for real querying at scale |
| Manual JSON-based run storage | MLflow's standardized `MLmodel` packaging format, supporting dozens of ML frameworks out of the box |
| No model registry shown | MLflow Model Registry — full stage-transition audit trail, webhooks for CI/CD integration (Lesson 4) |
| Alternatives | Weights & Biases (`wandb`) — a popular commercial alternative with similar core tracking capabilities plus richer visualization/collaboration features |

---

## 8. Visual Explanations

**MLflow's core components:**
```
Training code ──▶ Tracking (params, metrics, artifacts logged per RUN, grouped by EXPERIMENT)
                        │
                        ▼
              Compare runs (UI or programmatic query) ──▶ identify BEST run
                        │
                        ▼
              Register model ──▶ Model Registry (versioned, e.g., v1, v2, v3...)
                        │
                        ▼
              Stage transitions: Staging ──▶ Production ──▶ (eventually) Archived
                        │
                        ▼
              Serving infrastructure loads whichever version is CURRENTLY "Production"
```

**Experiment tracking as a permanent, queryable record (vs. scattered notebook cells):**
```
WITHOUT tracking:                          WITH MLflow tracking:
[Notebook cell 1: try n_est=50]            Run 1: {n_est: 50, auc: 0.82, ci: [0.79, 0.85]}
[Notebook cell 2: try n_est=100]           Run 2: {n_est: 100, auc: 0.85, ci: [0.82, 0.88]}
[Notebook cell 3: OVERWRITE cell 1's       Run 3: {n_est: 200, auc: 0.86, ci: [0.83, 0.89]}
 variables, try n_est=200]                    (ALL runs preserved, queryable, comparable --
 (which config was BEST? scroll back         nothing overwritten or lost)
 through cell history and hope...)
```

---

## 9. Practical Examples

**Simple:** run the Section 5 hyperparameter sweep and inspect the results via the MLflow UI (`mlflow ui`), comparing runs' logged metrics.
**Medium:** register the best-performing run as a model version and manually transition it through Staging to Production via the MLflow client API.
**Real-world:** integrate MLflow tracking into your Phase 4 actuarial model training pipeline, logging every hyperparameter configuration tried along with bootstrap confidence intervals, and use the registry to formally track which model version is currently deployed in your Phase 8 Lesson 1 FastAPI application.

---

## 10. Real Industry Use Cases

- **Standard MLOps infrastructure at most organizations doing serious ML work**: MLflow (or a comparable tool, Weights & Biases) is close to a default choice for experiment tracking and model registry functionality.
- **Regulatory/compliance auditability**: in regulated industries (insurance, finance, healthcare — directly relevant to your domain), MLflow's permanent, versioned audit trail of which model was in production when, and what data/parameters produced it, directly supports compliance and audit requirements.
- **CI/CD integration** (Lesson 4): many organizations automate model registry stage transitions as part of their deployment pipeline, gating "Production" promotion on automated evaluation checks exactly as this lesson describes.
- **Cross-team model sharing**: the Model Registry provides a central, discoverable catalog letting different teams find and reuse validated models rather than each team maintaining its own, potentially inconsistent tracking.

---

## 11. Common Mistakes

- Not logging confidence intervals alongside point-estimate metrics — reintroducing exactly the statistical-rigor gap Phase 4 Lesson 3 warns against, even with proper tracking infrastructure in place.
- Manually managing "which model is currently in production" via file naming conventions or tribal knowledge rather than the Model Registry's explicit, audited stage-transition mechanism.
- Not logging enough information to actually reproduce a run later (missing code version/Git commit hash, missing exact data version) — undermining the reproducibility goal that's MLflow's central design purpose.
- Treating experiment tracking as optional/only for "important" runs — the value comes precisely from comprehensive coverage, since you often don't know in advance which run will turn out to matter later.

---

## 12. Best Practices (2026)

- Log confidence intervals (not just point estimates) for every evaluation metric, directly applying Phase 3 Lesson 4's and Phase 4 Lesson 3's statistical rigor within your tracking infrastructure.
- Use the Model Registry's explicit stage-transition mechanism (not ad hoc file/naming conventions) for tracking which model version is deployed where.
- Log enough metadata (code version, data version, environment) with every run to genuinely support later reproduction, not just the immediate metrics comparison.
- Integrate registry stage transitions into your CI/CD pipeline (Lesson 4) as an automated, gated step rather than a manual, undocumented process.

---

## 13. Exercises

**Easy:** Set up MLflow tracking for a simple scikit-learn model training script, logging parameters and metrics for 3 different hyperparameter configurations.
**Medium:** Extend your tracking to log bootstrap confidence intervals alongside point-estimate metrics, and use `mlflow.search_runs` to programmatically identify the run with the best lower-CI-bound performance.
**Hard:** Build a complete registry workflow: register a model, transition it through Staging and Production stages via the client API, and write a script that queries the registry to load "whichever model is currently in Production" for use in a serving application.
**Mathematical:** Given logged runs with both point-estimate and confidence-interval metrics, design a selection criterion (e.g., "highest lower-CI-bound" versus "highest point estimate") and discuss the tradeoffs between them for model selection.
**Coding:** Integrate MLflow model registry queries into your Phase 8 Lesson 1 FastAPI application's model-loading dependency, so the API always serves whichever model version is currently staged as "Production" without requiring a code change/redeployment to update the served model.

---

## 14. Mini Project

Build a **complete MLflow-tracked training and deployment pipeline for your actuarial mortality model**: implement a hyperparameter sweep with full experiment tracking (parameters, bootstrap-confidence-interval metrics, model artifacts, diagnostic plots), register the best-performing model, implement a CI/CD-integrated (Lesson 4) automated stage-transition workflow gating "Production" promotion on a statistically-rigorous evaluation threshold, and modify your Phase 8 Lesson 1 FastAPI application to dynamically load whichever model version is currently staged as "Production" from the registry.

---

## 15. Interview Preparation

- Explain the core components of MLflow (tracking, model packaging, registry) and what problem each solves.
- Why is it important to log confidence intervals, not just point-estimate metrics, when tracking ML experiments?
- How would you integrate MLflow's Model Registry into a CI/CD pipeline for automated, gated model deployment?
- What information should be logged with every training run to genuinely support future reproducibility?

---

## 16. Summary

MLflow addresses the ML lifecycle's unique tracking and reproducibility needs — experiment tracking creates a permanent, queryable record of every training run's parameters, metrics (ideally including confidence intervals, directly extending Phase 3-4's statistical rigor), and artifacts; the Model Registry provides an explicit, audited stage-transition workflow (Staging → Production → Archived) replacing ad hoc "which model is deployed" tribal knowledge; and framework-agnostic model packaging enables consistent serving regardless of the original training framework. This tooling is the direct, practical infrastructure making Phase 4-7's models genuinely manageable, auditable, and safely deployable at organizational scale — setting up Lesson 7's broader MLOps discipline, which extends these same principles across the full ML system lifecycle.

---

## 17. References

- Official MLflow documentation (mlflow.org/docs)
- Zaharia et al. — "Accelerating the Machine Learning Lifecycle with MLflow" (2018, the original MLflow paper)
- Databricks' MLflow best-practices guides
- Weights & Biases documentation (a widely-used alternative/complementary tool)
