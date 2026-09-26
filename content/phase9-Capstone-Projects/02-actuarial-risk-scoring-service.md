# Phase 9 · Project 2 — Actuarial Risk Scoring Service

> Difficulty: ★★☆☆☆ | Primary phases exercised: 2, 3, 4, 8 (Lessons 1–7)

---

## Overview

A mortality/lapse risk-scoring API for insurance underwriting, built on the full tabular-ML pipeline from Phases 2–4: engineered features (Phase 2 Lesson 6), an XGBoost model compared against a logistic regression baseline (Phase 4 Lesson 1), leakage-safe feature selection (Phase 4 Lesson 4), and a stacked ensemble (Phase 4 Lesson 5) — evaluated with SHAP explanations for regulatory interpretability. This is the first project requiring genuine feature-engineering rigor and a real MLOps loop, not just a serving wrapper.

---

## System Architecture

```
Raw policyholder data ──▶ Feature Store (Phase 8 Lesson 7) ──▶ Training pipeline
                                    │                                │
                                    │                                ▼
                                    │                    MLflow tracking + registry
                                    │                                │
                                    ▼                                ▼
                          FastAPI serving ◀── loads "Production"-staged model
                                    │
                          SHAP explanation endpoint (regulatory auditability)
                                    │
                          Drift monitor (PSI, Phase 8 Lesson 7) ──▶ retraining trigger
```

The **feature store** is the architectural centerpiece: the exact same age-banding, tenure, and target-encoded region features (Phase 2 Lesson 6, Phase 4 Lesson 4's leakage-safe selection) are computed identically whether preparing training data or scoring a live request — eliminating train/serve skew from day one, not retrofitted later.

---

## Folder Structure

```
risk-scoring-service/
├── src/
│   ├── features/
│   │   ├── feature_store.py     # single source of truth, Phase 8 Lesson 7
│   │   └── definitions.py        # age_band, tenure_years, kfold target encodings
│   ├── training/
│   │   ├── train_baseline.py     # logistic regression, interpretable baseline
│   │   ├── train_xgboost.py
│   │   ├── train_ensemble.py     # stacking, Phase 4 Lesson 5
│   │   └── feature_selection.py  # VIF + Lasso + stability selection, Phase 4 Lesson 4
│   ├── evaluation/
│   │   ├── nested_cv.py          # Phase 4 Lesson 3
│   │   └── shap_explain.py
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── explain_endpoint.py
│   └── monitoring/
│       └── drift_detector.py     # PSI, Phase 8 Lesson 7
├── tests/ (unit, integration, feature-consistency tests)
├── Dockerfile, docker-compose.yml
├── .github/workflows/ci-cd.yml
├── k8s/ (deployment, service, hpa, configmap)
└── docs/model_card.md
```

---

## Documentation

`docs/model_card.md`: intended use (underwriting *support*, not automated denial — a genuine regulatory/ethical line), feature list with plain-language descriptions, VIF/multicollinearity report, SHAP-based global feature importance, disaggregated performance by region/age-band (Phase 8 Lesson 10's bias-evaluation discipline, applied here since this is a consequential domain).

---

## Testing

Beyond Project 1's layers: **feature-consistency tests** asserting `feature_store.compute_features()` returns byte-identical output whether called from the training pipeline or the serving path (the specific bug class this architecture exists to prevent) — and a leakage regression test confirming K-fold target encoding never uses out-of-fold information (Phase 4 Lesson 4/Phase 2 Lesson 6).

---

## Dockerization

Same multi-stage pattern as Project 1, plus a second image for the drift-monitoring background job (a separate Kubernetes `CronJob`, not bundled into the API container — a genuine separation-of-concerns decision, Phase 1 Lesson 8).

---

## CI/CD

Extends Project 1's pipeline with a **statistically-rigorous quality gate** (Phase 8 Lesson 4's bootstrap-CI gate) comparing the newly-trained ensemble against the *currently deployed* model, not an absolute threshold — blocking promotion unless the new model's lower-CI-bound AUC genuinely exceeds the current production model's.

---

## Deployment

Kubernetes Deployment for the API + a `CronJob` running nightly PSI drift checks (Phase 8 Lesson 7) against a reference training-data snapshot, writing results to the monitoring stack and triggering a retraining pipeline run if any feature's PSI exceeds 0.25.

---

## Monitoring

RED metrics + **feature-level PSI dashboards** per engineered feature (age_band, region encoding, tenure) + SHAP-value distribution tracking over time (a subtle but real signal: if the *reasons* behind predictions shift even when accuracy looks stable, that's worth investigating).

---

## Evaluation

Full nested cross-validation (Phase 4 Lesson 3) during model selection; bootstrap-CI-gated promotion (above); disaggregated performance evaluation by policyholder subgroup as a standing, re-run-every-retraining check (Phase 8 Lesson 10).

---

## Scalability

Batch-scoring mode added alongside real-time single-record scoring: a nightly batch job scoring the full active policy book, using vectorized feature computation (Phase 2 Lesson 1's NumPy vectorization) rather than looping per-record — a genuine scale requirement once the book reaches hundreds of thousands of policies.

---

## Security Considerations

Regulatory audit trail: every scored request logs the model version (from the MLflow registry) used, enabling "which model version produced this specific historical score" reconstruction — a compliance necessity in insurance, directly built on Phase 8 Lesson 6's registry versioning.

---

## Definition of Done

- [ ] Feature store verified byte-identical between training and serving via automated test
- [ ] Ensemble model beats logistic-regression baseline with statistically significant lower-CI-bound margin
- [ ] SHAP explanation endpoint returns per-prediction feature attributions
- [ ] Nightly PSI drift job deployed and alerting correctly on injected synthetic drift
- [ ] Disaggregated bias evaluation report included in the model card
- [ ] Batch-scoring mode benchmarked against real-time mode for a 100K-record book
