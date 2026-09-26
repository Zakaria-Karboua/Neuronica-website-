# Phase 8 · Lesson 4 — CI/CD

> Prerequisite: Docker, Kubernetes (Lessons 2–3), Phase 1 Lessons 5 and 9 (Git, Testing)

---

## 1. Introduction

### What is CI/CD?
**Continuous Integration** (automatically building and testing every code change as soon as it's proposed) and **Continuous Delivery/Deployment** (automatically packaging and deploying validated changes toward production) — together forming an automated pipeline that takes code from a developer's commit through to a running, verified production deployment, with minimal or no manual intervention.

### Why does it exist?
Manual testing and deployment don't scale reliably: humans forget steps, skip tests under deadline pressure, and introduce inconsistency between "what was tested" and "what was actually deployed." CI/CD exists to make the entire path from code change to production deployment repeatable, automated, and consistently rigorous — directly extending Phase 1 Lesson 9's testing discipline and Lesson 5's Git workflow into a fully automated pipeline.

### Historical background
Continuous Integration as a formal practice traces to Extreme Programming (Beck, 1990s) and was popularized by early CI tools (CruiseControl, Jenkins, 2000s-2010s). The "CD" half (Continuous Delivery, later Continuous Deployment) matured alongside the DevOps movement and cloud-native infrastructure (Docker, Kubernetes) through the 2010s, with modern platforms (GitHub Actions, GitLab CI, and others) making CI/CD accessible to virtually any project by 2026, not just large engineering organizations.

### Real-world motivation
Every serious ML/LLM application built across this curriculum needs a reliable, automated path from "I changed the code" to "the change is tested, packaged, and safely running in production" — CI/CD is precisely that automated path, and is expected infrastructure for any production system, not an optional nicety.

---

## 2. Theory

### The CI/CD pipeline stages
1. **Trigger**: a code push, pull request, or scheduled event starts the pipeline.
2. **Build**: compile/package the code (for Python, often simply installing dependencies; for containerized apps, building the Docker image, Lesson 2).
3. **Test**: run automated tests (Phase 1 Lesson 9) — unit tests, integration tests, and for ML specifically, potentially model-quality checks (Phase 4 Lesson 3's evaluation rigor, automated).
4. **Package**: produce a deployable artifact (a tagged Docker image, pushed to a registry).
5. **Deploy**: apply the new artifact to a target environment (staging, then production) — often via updating a Kubernetes Deployment (Lesson 3) to reference the new image tag.

### Continuous Integration vs. Continuous Delivery vs. Continuous Deployment
- **Continuous Integration (CI)**: every code change is automatically built and tested — the "did this change break anything" check.
- **Continuous Delivery**: every validated change is automatically packaged and made *ready* to deploy, but a human still approves the actual production release (a deliberate gate for higher-risk changes).
- **Continuous Deployment**: every validated change is automatically deployed to production with **no** human approval gate — the fastest, most automated end of the spectrum, appropriate for mature, well-tested systems with strong automated safety nets.

### Pipeline-as-code
Modern CI/CD platforms define the entire pipeline (build/test/deploy steps) as a version-controlled configuration file (e.g., a GitHub Actions YAML workflow) living *in the same repository* as the application code — directly extending Phase 1 Lesson 5's version-control discipline to the deployment process itself, ensuring pipeline changes are reviewed and tracked exactly like code changes.

### ML-specific CI/CD considerations
Beyond standard software testing, ML/LLM pipelines benefit from additional automated checks: data validation (Phase 2 Lesson 3's cleaning/schema checks, automated via tools like `pandera`), model quality gates (automatically failing a pipeline if a newly trained model's evaluation metric, Phase 4 Lesson 3, falls below a threshold or regresses versus the previous version), and, for LLM applications specifically, automated evaluation suites (Phase 7 Lesson 10's faithfulness/relevance checks) run against a fixed test set before allowing deployment.

---

## 3. Mathematical Foundations

CI/CD is primarily a process/engineering discipline, but a few quantitative framings are genuinely useful:

### Deployment frequency and batch size (a real risk/speed tradeoff)
Smaller, more frequent deployments (each containing fewer changes) reduce the "blast radius" of any single deployment going wrong (fewer simultaneous changes to investigate if something breaks) and make root-cause diagnosis faster — a direct, empirically well-supported argument (from DevOps research, notably the "Accelerate"/DORA research program) for frequent small deployments over infrequent large ones, all else equal.

### Statistical confidence in automated quality gates
An automated model-quality gate (Section 2) comparing a new model's evaluation metric against a threshold or the previous version's score should apply Phase 4 Lesson 3's statistical rigor — a naive single-point-estimate comparison risks blocking a genuinely-equivalent-or-better model due to noise, or worse, passing a genuinely-worse model that happened to score well on a particular evaluation run; a proper gate should account for confidence intervals (Phase 3 Lesson 4's bootstrap), not just raw score comparison.

### Rollback speed as a risk-mitigation lever
If a deployment issue is detected, the speed of rolling back to the previous known-good version directly bounds the duration/impact of any production incident — Kubernetes' rolling-update mechanism (Lesson 3) supports fast rollback (reverting to the previous image tag, itself just another rolling update in the opposite direction) precisely because deployment artifacts are versioned/tagged, not overwritten in place — a direct, quantifiable argument for why immutable, versioned deployment artifacts (rather than in-place server modification) are foundational to reliable operations.

---

## 4. Algorithm — A Complete CI/CD Pipeline for an ML/LLM Application (fully specified)

```
TRIGGER: a pull request is opened, or code is merged to the main branch

STAGE 1 -- CONTINUOUS INTEGRATION:
  1. CHECKOUT the code at the triggering commit
  2. INSTALL dependencies (cached where possible, Lesson 2's caching principle applied to CI)
  3. RUN static analysis/linting (Phase 1 Lesson 8's SOLID/style checks, automated: ruff, mypy)
  4. RUN unit tests (Phase 1 Lesson 9) -- FAIL the pipeline immediately if any test fails
  5. RUN integration tests (testing against a real or realistic test database/service)
  6. (ML-SPECIFIC) RUN data validation checks (Phase 2 Lesson 3) and, if a model was retrained,
     model quality gate checks (Phase 4 Lesson 3's evaluation, WITH confidence intervals)
  7. IF all checks pass: mark the commit as CI-VALIDATED; ELSE: block merging/deployment

STAGE 2 -- CONTINUOUS DELIVERY/DEPLOYMENT (typically only on merge to main, not every PR):
  8. BUILD the Docker image (Lesson 2), tagged with a unique identifier (commit SHA, or semantic version)
  9. PUSH the image to a container registry
  10. DEPLOY to a STAGING environment first; run smoke tests / a small evaluation suite (Phase 7 Lesson 10)
  11. IF staging validation passes:
        (Continuous Delivery: WAIT for human approval)
        (Continuous Deployment: PROCEED automatically)
        UPDATE the production Kubernetes Deployment to the new image tag (Lesson 3's rolling update)
  12. MONITOR post-deployment metrics (Lesson 8) for a defined period; AUTOMATICALLY ROLL BACK
      (Section 3's fast-rollback principle) if error rates/latency regress beyond a defined threshold
```

---

## 5. Python Implementation (GitHub Actions workflow + supporting scripts)

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"                       # Lesson 2's caching principle, applied to CI dependency installs

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Lint
        run: ruff check .

      - name: Type check
        run: mypy app/

      - name: Unit + integration tests
        run: pytest --cov=app --cov-report=xml tests/

      - name: Data/model quality gate (ML-specific)
        run: python scripts/check_model_quality_gate.py --min-auc 0.75 --confidence-level 0.95

  build-and-push:
    needs: test
    if: github.ref == 'refs/heads/main'       # only build/push on actual merges, not every PR
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t actuarial-api:${{ github.sha }} .   # tag with commit SHA -- immutable, traceable
      - name: Push to registry
        run: |
          docker tag actuarial-api:${{ github.sha }} registry.example.com/actuarial-api:${{ github.sha }}
          docker push registry.example.com/actuarial-api:${{ github.sha }}

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: kubectl set image deployment/actuarial-api-staging actuarial-api=registry.example.com/actuarial-api:${{ github.sha }}
      - name: Run staging smoke tests
        run: python scripts/staging_smoke_test.py --endpoint https://staging.example.com

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production                  # GitHub's environment protection -- can require manual approval here
    steps:
      - name: Deploy to production
        run: kubectl set image deployment/actuarial-api actuarial-api=registry.example.com/actuarial-api:${{ github.sha }}
      - name: Monitor post-deployment
        run: python scripts/monitor_and_rollback.py --deployment actuarial-api --duration-minutes 10
```

```python
"""check_model_quality_gate.py — Section 3's statistically-rigorous quality gate"""
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95):
    """Directly reuses Phase 3 Lesson 4's bootstrap confidence interval technique."""
    rng = np.random.default_rng(0)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        scores.append(roc_auc_score(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    return np.mean(scores), lower

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-auc", type=float, required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    y_true, y_pred = load_holdout_predictions()   # loads the newly retrained model's holdout predictions
    mean_auc, lower_bound = bootstrap_ci(y_true, y_pred, ci=args.confidence_level)

    print(f"AUC: {mean_auc:.4f} (lower {args.confidence_level:.0%} CI bound: {lower_bound:.4f})")
    if lower_bound < args.min_auc:
        print(f"QUALITY GATE FAILED: lower confidence bound {lower_bound:.4f} < threshold {args.min_auc}")
        exit(1)                                     # non-zero exit -> CI pipeline FAILS, blocking deployment
    print("Quality gate PASSED.")

def load_holdout_predictions():
    # Placeholder -- in practice, loads real model predictions on a fixed holdout set
    rng = np.random.default_rng(1)
    return rng.integers(0, 2, 500), rng.random(500)

if __name__ == "__main__":
    main()
```

---

## 6. Build From Scratch

**A minimal automatic-rollback monitor (making Section 4's post-deployment monitoring/rollback step concrete):**
```python
import time

def monitor_and_rollback(deployment_name: str, get_error_rate_fn, rollback_fn,
                            duration_seconds: int = 60, error_threshold: float = 0.05, check_interval: int = 5):
    """Polls error rate after a deployment; automatically rolls back if it exceeds a threshold."""
    start = time.time()
    while time.time() - start < duration_seconds:
        current_error_rate = get_error_rate_fn(deployment_name)
        print(f"Current error rate: {current_error_rate:.3%}")
        if current_error_rate > error_threshold:
            print(f"ERROR RATE {current_error_rate:.3%} EXCEEDS THRESHOLD {error_threshold:.3%} -- ROLLING BACK")
            rollback_fn(deployment_name)
            return False
        time.sleep(check_interval)
    print("Deployment monitoring window completed successfully -- no rollback needed.")
    return True

# Illustrative mocks (a real implementation queries Lesson 8's monitoring/observability stack)
mock_error_rates = iter([0.01, 0.02, 0.08])   # simulating error rate spiking after deployment
mock_get_error_rate = lambda name: next(mock_error_rates, 0.01)
mock_rollback = lambda name: print(f"Rolling back {name} to previous version...")

monitor_and_rollback("actuarial-api", mock_get_error_rate, mock_rollback, duration_seconds=20, check_interval=5)
```
This directly implements Section 4's final pipeline step: automated post-deployment monitoring with a hard, threshold-based automatic rollback trigger — removing the need for a human to be actively watching dashboards immediately after every deployment, while still providing a fast, automatic safety net if something goes wrong.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| Manual test-then-deploy scripts | GitHub Actions, GitLab CI, Jenkins — full pipeline orchestration with triggers, parallel jobs, environment protections |
| `check_model_quality_gate.py` (custom) | MLflow (Lesson 6) or dedicated ML CI tools (Weights & Biases' automation features) for model-quality-gate integration into CI |
| `monitor_and_rollback` (polling loop) | Kubernetes-native progressive delivery tools (Argo Rollouts, Flagger) — implement canary/blue-green deployments with automated metric-based rollback natively |

---

## 8. Visual Explanations

**The full CI/CD pipeline (Section 4):**
```
Code Push/PR ──▶ CI: Lint + Test + Quality Gate ──▶ PASS? ──NO──▶ Block merge/deploy
                                                        │
                                                       YES
                                                        ▼
                                          Build + Push Docker Image (Lesson 2)
                                                        │
                                                        ▼
                                          Deploy to STAGING ──▶ Smoke Tests
                                                        │
                                                       PASS
                                                        ▼
                              (CD: human approval)  or  (Continuous Deployment: automatic)
                                                        │
                                                        ▼
                                          Deploy to PRODUCTION (Lesson 3's rolling update)
                                                        │
                                                        ▼
                                          Monitor post-deployment ──▶ metrics BAD? ──▶ AUTO-ROLLBACK
```

**Deployment frequency vs. batch size and risk (Section 3):**
```
INFREQUENT, LARGE deployments:        FREQUENT, SMALL deployments:
[============ 50 changes ============]   [==5==][==4==][==6==][==3==] ...
   if something breaks: HARD to           if something breaks: easy to isolate
   isolate WHICH of 50 changes             WHICH small batch caused it, faster
   caused it                                to diagnose and roll back
```

---

## 9. Practical Examples

**Simple:** set up a basic GitHub Actions workflow running linting and unit tests on every pull request to a repository.
**Medium:** extend the workflow to build and push a Docker image (Lesson 2) on merge to main, tagged with the commit SHA.
**Real-world:** build the complete pipeline (Section 5) for your Phase 8 Lesson 3 Kubernetes-deployed actuarial API, including the statistically-rigorous model-quality gate and the automated post-deployment rollback monitor, and walk through a full simulated deployment (including a deliberately-triggered rollback scenario) end to end.

---

## 10. Real Industry Use Cases

- **Every serious software engineering organization**: CI/CD is now near-universal standard practice, not a differentiator — its absence is considered a significant engineering-maturity red flag.
- **ML platforms at scale** (Netflix, Uber, and others' internal ML infrastructure): extend standard CI/CD with ML-specific quality gates (data validation, model evaluation thresholds) exactly as covered in Section 4.
- **LLM application deployment**: increasingly incorporates automated evaluation suites (Phase 7 Lesson 10) as a CI/CD quality gate specifically for RAG/agentic application changes, not just traditional software tests.
- **Canary and blue-green deployment strategies**: many organizations extend basic rolling updates (Lesson 3) with more sophisticated progressive-delivery patterns (routing a small percentage of real traffic to a new version before full rollout) specifically to catch issues with real production traffic before a full deployment.

---

## 11. Common Mistakes

- No automated tests in the CI pipeline (or tests that are frequently skipped/ignored when failing) — defeats the entire purpose of continuous integration, allowing broken code to merge.
- Deploying directly to production without a staging environment/smoke-test step — misses an opportunity to catch environment-specific issues before they affect real users.
- No automated rollback mechanism — relying on a human noticing a problem and manually reverting, which is slower and less reliable than an automated threshold-based rollback (Section 6).
- Using a naive point-estimate comparison for ML model quality gates without confidence intervals (Section 3) — risks both false-positive gate failures (blocking a genuinely fine model due to noise) and false-negative passes (allowing a genuinely worse model through due to favorable noise on one evaluation run).

---

## 12. Best Practices (2026)

- Treat pipeline configuration as code, version-controlled alongside the application it builds/deploys/tests.
- Include ML-specific quality gates (data validation, statistically-rigorous model evaluation thresholds) in any pipeline building/deploying an ML/LLM component.
- Always deploy to staging with smoke tests before production, and prefer progressive delivery (canary/blue-green) over an all-at-once production rollout for high-stakes services.
- Implement automated, metric-threshold-based rollback rather than relying solely on manual incident response.

---

## 13. Exercises

**Easy:** Set up a GitHub Actions workflow that runs `pytest` on every pull request to a simple repository.
**Medium:** Extend the workflow to build a Docker image and push it to a registry on merge to main, tagged with the commit SHA.
**Hard:** Implement the statistically-rigorous model quality gate (Section 5) and integrate it into a CI pipeline, testing that it correctly blocks a deliberately-degraded model while passing an equivalent-or-better one.
**Mathematical:** Given a deployment's observed error rate over time following a known distribution, design a monitoring window and threshold (Section 6) that balances catching genuine regressions quickly against false-alarm rate from normal statistical fluctuation.
**Coding:** Implement the automated rollback monitor (Section 6) integrated with a real (or realistically mocked) metrics endpoint, and test it against both a healthy deployment (no rollback) and a deliberately degraded one (triggers rollback).

---

## 14. Mini Project

Build a **complete CI/CD pipeline for your actuarial API application**: implement the full GitHub Actions workflow (Section 5) covering linting, testing, a statistically-rigorous model-quality gate, Docker image build/push, staged deployment (staging then production) to your Phase 8 Lesson 3 Kubernetes cluster, and an automated post-deployment rollback monitor; walk through and document a complete deployment cycle including a deliberately-triggered rollback scenario, producing a runbook another engineer could follow to understand and safely operate this pipeline.

---

## 15. Interview Preparation

- Explain the difference between Continuous Integration, Continuous Delivery, and Continuous Deployment.
- Why is deploying frequently in small batches generally safer than infrequent large deployments?
- What ML-specific quality gates would you add to a standard CI/CD pipeline for a model-serving application?
- How would you design an automated rollback mechanism, and what tradeoffs exist in choosing its monitoring window and threshold?

---

## 16. Summary

CI/CD automates the entire path from code change to verified production deployment — continuous integration catches problems early via automated linting, testing, and (for ML applications) statistically-rigorous quality gates (directly extending Phase 4 Lesson 3's evaluation rigor and Phase 3 Lesson 4's confidence-interval discipline into an automated pass/fail check); continuous delivery/deployment then packages (Lesson 2's Docker) and deploys (Lesson 3's Kubernetes rolling updates) validated changes, with automated post-deployment monitoring and rollback closing the loop as a fast, reliable safety net. This pipeline is the connective tissue binding together every engineering practice from Phase 1 (testing, version control) through this phase's containerization and orchestration into the fully automated, production-grade deployment process every serious application requires.

---

## 17. References

- Humble, J. & Farley, D. — *Continuous Delivery*
- Forsgren, Humble, Kim — *Accelerate* (the DORA research program, empirically validating deployment-frequency/batch-size best practices)
- GitHub Actions, GitLab CI official documentation
- Argo Rollouts / Flagger documentation (progressive delivery tooling for Kubernetes)
