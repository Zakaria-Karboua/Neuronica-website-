# Phase 9 · Project 1 — Sentiment Analysis API

> Difficulty: ★☆☆☆☆ (foundational) | Primary phases exercised: 1, 2, 4, 8 (Lessons 1–2)

---

## Overview

A production-grade REST API that classifies short text (e.g., customer reviews, support tickets) as Positive/Negative/Neutral, trained on a classical ML pipeline (TF-IDF + Logistic Regression, or a small gradient-boosted model). Deliberately the simplest project in this capstone sequence — its purpose is to exercise the *full* production lifecycle end to end on the lowest-complexity possible model, so every subsequent project can build on this scaffold rather than re-deriving it.

**Why this project first:** every later project reuses this one's skeleton (API structure, test layout, CI/CD pipeline, Dockerfile). Getting this right, in full, is more valuable than a half-finished harder project.

---

## System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────┐
│   Client    │───▶│  FastAPI     │───▶│  Trained sklearn │───▶│ Response │
│  (HTTP req) │    │  (Phase 8.1) │    │  Pipeline        │    │  (JSON)  │
└─────────────┘    └──────┬───────┘    └─────────────────┘    └──────────┘
                           │
                           ▼
                   Structured logging ──▶ Prometheus metrics (Phase 8.8)
```

- **Training pipeline** (offline, Phase 2 Lessons 1–6 + Phase 4 Lesson 1): clean text, TF-IDF vectorize, train Logistic Regression, evaluate with Phase 4 Lesson 3 rigor (precision/recall/F1 + bootstrap CI), log to MLflow (Phase 8 Lesson 6).
- **Serving layer**: FastAPI app loading the MLflow-registered "Production"-staged model at startup (Phase 8 Lesson 1's `lifespan` pattern).
- **No GPU, no vector DB, no LLM** — deliberately minimal, so the *engineering scaffold* is the focus, not model complexity.

---

## Folder Structure

```
sentiment-api/
├── src/
│   ├── training/
│   │   ├── data_prep.py       # Phase 2 Lesson 3 cleaning
│   │   ├── train.py            # Phase 4 Lesson 1 + MLflow logging
│   │   └── evaluate.py         # Phase 4 Lesson 3 bootstrap CI evaluation
│   ├── api/
│   │   ├── main.py             # FastAPI app + lifespan model loading
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── dependencies.py     # model-loading dependency injection
│   └── config.py
├── tests/
│   ├── unit/test_data_prep.py
│   ├── unit/test_schemas.py
│   └── integration/test_api.py
├── Dockerfile                    # multi-stage, Phase 8 Lesson 2
├── docker-compose.yml            # API + MLflow tracking server
├── .github/workflows/ci-cd.yml   # Phase 8 Lesson 4
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Documentation

`README.md` must include: problem statement, model card (training data description, known limitations — e.g., poor performance on sarcasm, non-English text per Phase 6 Lesson 1's tokenization-efficiency point if multilingual), API usage examples (`curl` + Python client), and a link to the auto-generated `/docs` (Phase 8 Lesson 1). Every module has docstrings; every non-obvious design decision gets an inline comment explaining *why*, not *what* (Phase 1 Lesson 8).

---

## Testing

- **Unit tests** (Phase 1 Lesson 9): data cleaning functions (edge cases — empty string, all-punctuation, very long input), Pydantic schema validation boundaries.
- **Integration tests**: full API request/response cycle against a loaded test-fixture model, including the `/health` endpoint and malformed-request rejection (422).
- **Model quality gate** (Phase 8 Lesson 4): CI blocks merge if bootstrap-CI-lower-bound F1 < 0.75 on a fixed holdout set.

---

## Dockerization

Multi-stage build (Phase 8 Lesson 2): build stage installs dependencies; final stage copies only installed packages + `src/`, runs as non-root `appuser`, includes a `HEALTHCHECK` hitting `/health`.

---

## CI/CD

GitHub Actions pipeline (Phase 8 Lesson 4): lint → unit tests → integration tests → model quality gate → build/push Docker image (tagged by commit SHA) → deploy to staging → smoke test → manual-approval gate → deploy to production → 10-minute automated rollback monitor.

---

## Deployment

Kubernetes (Phase 8 Lesson 3): a `Deployment` with 2 replicas, resource requests/limits (`250m`/`256Mi` requests), liveness + readiness probes on `/health`, and an `HPA` scaling on CPU utilization (target 50%, min 2/max 6 replicas). Runs on a managed cluster (EKS/GKE/AKS, Phase 8 Lesson 5) with on-demand pricing (traffic is presumed too variable for reserved pricing at this stage).

---

## Monitoring

RED metrics (Phase 8 Lesson 8): request rate, error rate, p50/p95/p99 latency. No AI-specific metrics needed yet beyond prediction-class distribution (useful for spotting a stuck/degenerate model always predicting one class). Alert on sustained (3-consecutive-check) p99 latency > 200ms or error rate > 2%.

---

## Evaluation

Phase 4 Lesson 3's full rigor: nested cross-validation during development, bootstrap confidence intervals reported in the model card, and a fixed, versioned holdout set re-evaluated on every retraining (never touched during model selection) as the CI/CD quality gate.

---

## Scalability

At this model's scale (a lightweight sklearn pipeline), a single small pod handles hundreds of requests/second — scalability here is really about *demonstrating* the HPA/Kubernetes patterns correctly rather than solving a genuine scale problem, in preparation for later projects where it will matter.

---

## Security Considerations

Phase 8 Lesson 9's traditional-security half applies fully even without AI-specific attack surface: API-key authentication (Phase 8 Lesson 1's `Depends`), no secrets hardcoded (environment variables / K8s Secrets), non-root container user, dependency vulnerability scanning in CI.

---

## Definition of Done

- [ ] Model trained, tracked in MLflow, registered, promoted to "Production" stage
- [ ] API passes all unit + integration tests locally and in CI
- [ ] Docker image builds under 200MB, runs as non-root, passes `HEALTHCHECK`
- [ ] Deployed to a local/cloud Kubernetes cluster with working HPA
- [ ] Dashboard (even a simple script) shows live RED metrics
- [ ] README includes a model card and complete usage documentation
