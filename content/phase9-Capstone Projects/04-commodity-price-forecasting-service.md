# Phase 9 · Project 4 — Commodity Price Forecasting Service

> Difficulty: ★★★☆☆ | Primary phases exercised: 4 (Lesson 6), 5 (Lessons 4–5), 8 (Lessons 1–8)
> Domain tie-in: Brent crude oil / DZD exchange-rate forecasting

---

## Overview

A time-series forecasting API comparing three architectural families head-to-head — classical (ARIMA), gradient-boosted trees with lag features, and a small LSTM — served behind one unified API, with mandatory walk-forward validation and a naive-forecast baseline as a hard floor every model must beat (Phase 4 Lesson 6). This project's difficulty comes from getting *validation methodology* right, not model complexity.

---

## System Architecture

```
Historical price data ──▶ Walk-forward validation harness (Phase 4 Lesson 6)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
                 ARIMA            XGBoost+lags            LSTM
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        ▼
                          Model comparison report (vs naive baseline)
                                        │
                                        ▼
                          FastAPI serving (best-validated model, hot-swappable)
```

Critically: **no random train/test splitting anywhere in this codebase** — a CI check (Section: Testing) statically greps for and rejects any use of `train_test_split` without `shuffle=False`, enforcing Phase 4 Lesson 6's leakage warning at the tooling level, not just as a reminder in documentation.

---

## Folder Structure

```
price-forecasting-service/
├── src/
│   ├── validation/
│   │   ├── walk_forward.py       # the ONLY sanctioned validation method
│   │   └── naive_baseline.py     # mandatory comparison point
│   ├── models/
│   │   ├── arima_model.py
│   │   ├── xgboost_lag_model.py  # engineered lag/rolling features, Phase 2 Lesson 6
│   │   └── lstm_model.py         # Phase 5 Lesson 5
│   ├── stationarity/adf_test.py
│   ├── api/main.py
│   └── comparison_report.py
├── tests/
│   ├── unit/test_walk_forward_no_leakage.py   # asserts train indices always precede test indices
│   └── ci_checks/no_random_split.py            # static analysis check, run in CI
├── Dockerfile
├── k8s/
└── docs/forecasting_methodology.md
```

---

## Documentation

`docs/forecasting_methodology.md` explicitly documents: the stationarity testing procedure (ADF test results per series), why walk-forward (not k-fold) validation is used, and — most importantly — an honest statement of whether *any* model actually beats the naive random-walk baseline with statistical significance, for each series. If none do, that's a documented, legitimate finding, not a failure to hide.

---

## Testing

A dedicated `test_walk_forward_no_leakage.py` asserting, for every generated fold, that every training-set timestamp strictly precedes every test-set timestamp — the single most important test in this project, directly enforcing Phase 4 Lesson 6's core correctness requirement as executable code rather than a documentation note. A CI static-analysis step scans for accidental use of ordinary (shuffled) cross-validation utilities.

---

## Dockerization

Standard multi-stage build; the training image additionally bundles `statsmodels` (ARIMA) and includes a scheduled retraining entrypoint distinct from the serving entrypoint (two `CMD` targets via build args, Phase 8 Lesson 2).

---

## CI/CD

Quality gate compares each candidate model's walk-forward MAE against the naive baseline's MAE **with a paired bootstrap confidence interval on the difference** (Phase 3 Lesson 4) — promotion requires the lower CI bound of the improvement to exceed zero, not just a favorable point estimate.

---

## Deployment

The API serves whichever model the comparison report identifies as best-validated for each specific series (oil vs. DZD may have different winners) — a per-series model registry entry (Phase 8 Lesson 6), not a single global "the model."

---

## Monitoring

Forecast-error tracking in production: as true future prices become known, compute realized forecast error and compare against the walk-forward-validated expected error distribution — a genuine post-deployment validation loop distinct from pre-deployment backtesting, catching cases where backtest performance doesn't hold up on truly new, live data.

---

## Evaluation

Full three-way comparison (ARIMA/XGBoost/LSTM) against the naive baseline, reported with bootstrap confidence intervals, for both the oil and DZD series, at multiple forecast horizons (1-day, 5-day, 20-day) since a model winning at one horizon may not win at another.

---

## Scalability

Forecasting many independent series (extending beyond oil/DZD to a broader commodity basket) via parallelized walk-forward validation across series (Phase 1 Lesson 2's multiprocessing, since each series' validation is independent).

---

## Security Considerations

Standard API auth/secrets practices (Phase 8 Lesson 9); no AI-specific attack surface here (no untrusted text processed by an LLM), making this project a useful contrast case for recognizing when Lesson 9's LLM-specific concerns simply don't apply.

---

## Definition of Done

- [ ] Walk-forward-no-leakage test passes and is enforced in CI
- [ ] All three model families implemented and compared against the naive baseline with bootstrap CIs
- [ ] Methodology document honestly reports which (if any) models beat the naive baseline significantly
- [ ] Production forecast-error tracking deployed and compared against backtested expectations
- [ ] Per-series model registry correctly serves the best-validated model for each series
