# Phase 9 · Project 9 — Continuous Retraining MLOps Platform

> Difficulty: ★★★★☆ | Primary phases exercised: 8 (Lessons 4, 6, 7, 8, 11) — applied across Projects 2, 3, and 5

---

## Overview

Rather than a new model, this project builds the **operational platform** wrapping Projects 2 (tabular risk scoring), 3 (image classification), and 5 (fine-tuned LLM classifier) into one unified, closed-loop MLOps system: automated drift detection, statistically-gated retraining, registry-based promotion, and CI/CD-integrated deployment — the fully-automated Phase 8 Lesson 7 lifecycle loop, generalized across three architecturally different model types.

---

## System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │       Unified Monitoring Layer            │
                    │  (drift, performance, cost — Lesson 8)    │
                    └──────┬────────────┬────────────┬──────────┘
                           ▼             ▼             ▼
                   Project 2's      Project 3's    Project 5's
                   tabular model    CNN model      LLM adapter
                           │             │             │
                           ▼             ▼             ▼
                  ┌────────────────────────────────────────────┐
                  │   Retraining Orchestrator (Airflow/Prefect)  │
                  │   - drift-triggered OR scheduled              │
                  │   - model-TYPE-AWARE retraining logic          │
                  └──────┬────────────┬────────────┬──────────────┘
                           ▼             ▼             ▼
                  Statistically-rigorous quality gate PER model type
                           │             │             │
                           ▼             ▼             ▼
                  MLflow registry promotion ──▶ CI/CD deploy (Lesson 4)
```

The core engineering challenge: drift detection (PSI, Phase 8 Lesson 7) and quality gating must be adapted per model type — PSI on tabular features (Project 2) is straightforward; "drift" for the image classifier (Project 3) means monitoring the *distribution of prediction confidence/classes* since raw pixel-level PSI is less meaningful; "drift" for the LLM classifier (Project 5) means monitoring input-document topic distribution shift.

---

## Folder Structure

```
mlops-platform/
├── src/
│   ├── drift_detection/
│   │   ├── tabular_drift.py      # PSI on engineered features
│   │   ├── vision_drift.py       # prediction-distribution monitoring
│   │   └── llm_drift.py          # input-topic-distribution monitoring
│   ├── orchestration/
│   │   ├── dags/                  # Airflow/Prefect DAG definitions, one per model
│   │   └── retraining_triggers.py
│   ├── quality_gates/
│   │   ├── tabular_gate.py        # bootstrap-CI AUC comparison
│   │   ├── vision_gate.py         # per-class F1 non-regression
│   │   └── llm_gate.py            # per-category accuracy + calibration
│   └── registry_integration.py    # unified MLflow promotion logic
├── tests/
│   ├── unit/ (per-drift-detector tests with synthetic injected drift)
│   └── integration/test_full_retraining_cycle.py
├── dashboards/
│   └── unified_model_health.py    # one dashboard, three model types
└── docs/runbook.md
```

---

## Documentation

`docs/runbook.md`: for each of the three model types, exactly what triggers retraining, what the quality gate requires for promotion, and what an on-call engineer should do if a retraining cycle fails or a gate rejects a new model repeatedly (Phase 8 Lesson 12's incident-response discipline, applied specifically to ML pipeline failures rather than serving outages).

---

## Testing

Each drift detector tested against synthetic data with known injected drift magnitude, verifying correct trigger/no-trigger decisions at the documented thresholds; a full-cycle integration test (inject drift → verify retraining triggers → verify quality gate correctly blocks a deliberately-degraded retrained model → verify a genuinely-improved model is correctly promoted) for at least one model type end to end.

---

## Dockerization

The orchestrator (Airflow/Prefect) runs as its own service; each model type's training job runs in its own container (reusing Projects 2/3/5's training images) invoked by the orchestrator — a genuine separation between "the system that decides when to retrain" and "the system that actually does the retraining," directly Phase 1 Lesson 8's separation-of-concerns principle applied at the platform level.

---

## CI/CD

This project's CI/CD pipeline tests the *platform itself* (the orchestration logic, drift detectors, quality gates) — distinct from, and running alongside, each individual model's own CI/CD pipeline from its originating project. A platform change (e.g., adjusting a PSI threshold) goes through its own review/testing/staged-rollout process before affecting any live model's retraining behavior.

---

## Deployment

The orchestrator and dashboards run as their own Kubernetes Deployments; retraining jobs run as Kubernetes `Job`s (not long-running Deployments) triggered by the orchestrator, scaling GPU/CPU resources up only for the duration of an actual training run (Phase 8 Lesson 5's spot-instance cost-optimization applies directly here, since retraining jobs are interruption-tolerant if properly checkpointed).

---

## Monitoring

A unified dashboard surfacing all three models' health (drift status, days-since-last-retrain, last-gate-result, current-production-version) side by side — the platform's actual deliverable is this consolidated visibility, not any single model's performance.

---

## Evaluation

Per-model-type quality gates as already specified in Projects 2/3/5, now triggered automatically rather than manually; a platform-level evaluation of the *orchestrator itself* — did it correctly identify drift when it was synthetically injected, and correctly avoid unnecessary retraining when it wasn't?

---

## Scalability

The orchestrator must handle a growing number of monitored models over time without linear increase in operational burden — achieved via the model-type-aware but not model-*instance*-aware drift/gate abstractions (adding a fourth tabular model reuses `tabular_drift.py`/`tabular_gate.py` unchanged).

---

## Security Considerations

Retraining jobs require access to production data (a genuine sensitive-data-handling concern, especially for Project 5's document classifier) — the orchestrator's service account is scoped to read-only access on production data stores and write-only access to the model registry, never broader permissions than each specific job type requires (Phase 8 Lesson 9's least-privilege principle, applied to pipeline infrastructure rather than agent tools).

---

## Definition of Done

- [ ] All three model types' drift detectors implemented and validated against synthetic drift
- [ ] Full retraining cycle (trigger → retrain → gate → promote → deploy) automated end to end for at least one model
- [ ] Unified dashboard operational across all three model types
- [ ] Runbook documented and (ideally) exercised via a simulated retraining-failure scenario
- [ ] Platform's own CI/CD pipeline tests the orchestration logic independently of any single model
