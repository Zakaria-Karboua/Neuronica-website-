# Phase 4 · Lesson 6 — Time Series

> Prerequisite: Supervised Learning, Model Evaluation (Lessons 1, 3), Phase 3 Statistics and Probability

---

## 1. Introduction

### What is time series analysis?
The study of data indexed in time order, where — critically — observations are typically **not** independent (violating the i.i.d. assumption underlying most of Phase 3-4's earlier lessons). Modeling temporal dependence, trend, seasonality, and forecasting future values requires a distinct toolkit: autocorrelation analysis, classical statistical models (ARIMA), and modern ML approaches adapted for temporal structure.

### Why does it exist?
Time series data is everywhere in your existing work — exchange rates (DZD), commodity prices (Brent crude), and claims arriving over time — and the standard i.i.d. train/test-split assumptions from Lesson 3 actively mislead if applied naively (randomly shuffling time-ordered data before splitting leaks future information into the training set, a specific and serious form of the data leakage problem from Phase 2 Lesson 5).

### Historical background
Box-Jenkins ARIMA methodology (1970) systematized classical time series forecasting; more recent decades added state-space models (Kalman filtering, itself connecting to your SDE background), and, since the 2010s-2020s, deep learning approaches (RNNs/LSTMs — Phase 5; and increasingly, transformer-based foundation models for time series in 2024-2026) now compete with and sometimes surpass classical methods, especially with many related series or rich exogenous covariates.

### Real-world motivation
Your Brent crude oil forecasting project and DZD exchange-rate work are time series problems. This lesson formalizes exactly the concepts (stationarity, autocorrelation, proper temporal validation) that should already underpin that work, and extends it with the classical ARIMA framework and a principled comparison against ML-based approaches.

---

## 2. Theory

### Stationarity — the foundational assumption of classical time series models
A time series is (weakly) **stationary** if its mean, variance, and autocovariance structure don't change over time. Most real financial/economic series (exchange rates, oil prices) are **not** stationary in their raw level (they trend, their variance changes) — differencing (analyzing $x_t - x_{t-1}$ rather than $x_t$ itself) is the standard technique to induce stationarity before applying classical models.

### Autocorrelation and partial autocorrelation
$$
\text{ACF}(k) = \text{Corr}(x_t, x_{t-k})
$$
measures how correlated a series is with its own past, at lag $k$. **Partial autocorrelation (PACF)** measures the correlation at lag $k$ *after controlling for* shorter lags — the ACF/PACF plots together are the classical diagnostic tool for identifying appropriate ARIMA model orders.

### ARIMA (AutoRegressive Integrated Moving Average)
$$
\text{ARIMA}(p,d,q): \quad \underbrace{\left(1 - \sum_{i=1}^p \phi_i L^i\right)}_{\text{AR: past values}} (1-L)^d x_t = \underbrace{\left(1 + \sum_{j=1}^q \theta_j L^j\right)}_{\text{MA: past errors}} \epsilon_t
$$
where $L$ is the lag operator ($Lx_t = x_{t-1}$), $d$ is the differencing order (to achieve stationarity), $p$ is the autoregressive order (how many past *values* directly predict the current one), and $q$ is the moving-average order (how many past *forecast errors* feed into the current prediction).

### Decomposition: trend, seasonality, residual
$$
x_t = T_t + S_t + R_t \quad \text{(additive)}, \qquad x_t = T_t \cdot S_t \cdot R_t \quad \text{(multiplicative)}
$$
Separating a series into trend, seasonal, and residual (irregular/noise) components — both for interpretability and because some models (e.g., classical ARIMA without seasonal extensions) require the seasonal component to be removed or explicitly modeled (SARIMA) first.

---

## 3. Mathematical Foundations

### The Augmented Dickey-Fuller (ADF) test for stationarity
Tests $H_0$: the series has a unit root (is non-stationary), against $H_1$: stationary, via the regression:
$$
\Delta x_t = \alpha + \beta t + \gamma x_{t-1} + \sum_{i=1}^k \delta_i \Delta x_{t-i} + \epsilon_t
$$
testing whether $\gamma = 0$ (unit root, non-stationary) using a specialized (non-standard, tabulated) critical-value distribution — directly a hypothesis test (Phase 3 Lesson 4) adapted for the specific statistical behavior of time series data.

### Why naive cross-validation fails for time series (formalized)
Standard k-fold CV (Lesson 3) assumes exchangeability — that folds can be randomly assigned without consequence. For time series, this assumption is **false**: randomly assigning a future observation to a "training" fold and an earlier observation to a "test" fold means the model is effectively trained using future information to predict the past — a severe, systematic leakage. The correct approach is **time-series cross-validation** (walk-forward validation): always train on a contiguous past window and validate/test on a subsequent, strictly later window.

### Exponential smoothing, derived
Simple exponential smoothing: $\hat x_{t+1} = \alpha x_t + (1-\alpha)\hat x_t$ — expanding this recursively:
$$
\hat x_{t+1} = \alpha \sum_{i=0}^{t} (1-\alpha)^i x_{t-i}
$$
a weighted average of *all* past observations, with exponentially decaying weights — the "exponential" in the name refers directly to this geometric decay structure, giving recent observations more influence without discarding older data entirely (unlike a simple moving average's hard cutoff).

### Random walk theory and market efficiency (directly relevant to your DZD/oil forecasting work)
A random walk $x_t = x_{t-1} + \epsilon_t$ (with $\epsilon_t$ i.i.d. noise) is stationary in its *differences* but not in levels, and — crucially — is **unpredictable**: the best forecast for $x_{t+1}$ is simply $x_t$ (a "naive forecast"). Financial time series (exchange rates, commodity prices) are frequently close to random walks in efficient markets, which is precisely why the naive forecast is such a stubbornly strong baseline that any more sophisticated model (ARIMA, ML, deep learning) must genuinely beat, not just numerically approach, to be considered actually adding value.

---

## 4. Algorithm — Walk-Forward (Time Series) Cross-Validation

```
GIVEN a time-ordered dataset of length n, and a chosen number of validation folds k:
FOR fold i = 1 to k:
    training window: ALL observations from the start up to some cutoff point t_i
    validation window: observations from t_i+1 to t_i + horizon  (STRICTLY AFTER the training window)
    TRAIN the model using ONLY the training window
    EVALUATE on the validation window
    ADVANCE t_i forward (expanding window: grow the training set; OR rolling window: slide both forward)
REPORT: performance averaged across all k folds
```
This never allows any validation observation to precede (in time) any training observation used to predict it — the single defining correctness property that ordinary k-fold CV violates for temporal data.

---

## 5. Python Implementation

```python
"""time_series_core.py"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error


def check_stationarity(series: pd.Series) -> dict:
    result = adfuller(series.dropna())
    return {"adf_statistic": result[0], "p_value": result[1], "is_stationary": result[1] < 0.05}


def walk_forward_validation(series: pd.Series, order: tuple, initial_train_size: int, horizon: int = 1):
    errors = []
    for cutoff in range(initial_train_size, len(series) - horizon, horizon):
        train = series.iloc[:cutoff]
        test = series.iloc[cutoff:cutoff + horizon]
        model = ARIMA(train, order=order).fit()
        forecast = model.forecast(steps=horizon)
        errors.append(mean_absolute_error(test, forecast))
    return np.mean(errors), np.std(errors)


def naive_forecast_baseline(series: pd.Series, horizon: int = 1) -> pd.Series:
    """The random-walk baseline EVERY more sophisticated model must beat (Section 3)."""
    return series.shift(horizon)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    # Simulate a random-walk-like price series with mild drift and seasonality (like a commodity price)
    trend = np.linspace(0, 5, n)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.cumsum(rng.normal(0, 0.5, n))    # random walk component
    prices = 100 + trend + seasonal + noise
    series = pd.Series(prices, index=pd.date_range("2020-01-01", periods=n, freq="D"))

    print("Raw series stationarity:", check_stationarity(series))
    print("Differenced series stationarity:", check_stationarity(series.diff()))

    decomposition = seasonal_decompose(series, model="additive", period=12)

    mae, std = walk_forward_validation(series, order=(1, 1, 1), initial_train_size=400, horizon=5)
    print(f"ARIMA(1,1,1) walk-forward MAE: {mae:.3f} +/- {std:.3f}")

    naive_preds = naive_forecast_baseline(series, horizon=5).dropna()
    actual = series[naive_preds.index]
    naive_mae = mean_absolute_error(actual, naive_preds)
    print(f"Naive (random walk) baseline MAE: {naive_mae:.3f}")
```

**Notes:** always compute the naive baseline MAE alongside any model's MAE — if ARIMA (or an ML model) doesn't clearly beat the naive forecast, the added modeling complexity likely isn't earning its keep on this particular series.

---

## 6. Build From Scratch

**A minimal AR(1) model fit via least squares (to demystify what ARIMA does internally for the simplest case):**
```python
import numpy as np

def fit_ar1(series: np.ndarray) -> tuple[float, float]:
    """x_t = phi * x_{t-1} + c + epsilon_t -- fit via ordinary least squares (Phase 3 Lesson 1's Normal Equation)."""
    x_t = series[1:]
    x_lag = series[:-1]
    X = np.column_stack([np.ones(len(x_lag)), x_lag])   # [intercept, lagged value]
    coeffs = np.linalg.solve(X.T @ X, X.T @ x_t)          # EXACTLY Phase 3 Lesson 1's Normal Equation
    c, phi = coeffs
    return phi, c

def forecast_ar1(last_value: float, phi: float, c: float, steps: int) -> list[float]:
    forecasts = []
    current = last_value
    for _ in range(steps):
        current = phi * current + c
        forecasts.append(current)
    return forecasts
```
This makes explicit that an AR(1) model is *just linear regression of $x_t$ on $x_{t-1}$* — the "time series" framing adds interpretation (autoregression, stationarity conditions on $\phi$: $|\phi|<1$ required for stationarity) but the actual fitting mechanism is the exact same Normal Equation from Phase 3 Lesson 1.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `fit_ar1` | `statsmodels.tsa.arima.model.ARIMA` — full $(p,d,q)$ support, MLE-based fitting (more robust than plain OLS for MA terms), confidence intervals |
| Manual walk-forward loop | `sklearn.model_selection.TimeSeriesSplit` (mechanically equivalent, integrates with sklearn pipelines) |
| Manual stationarity reasoning | `statsmodels.tsa.stattools.adfuller`/`kpss` — formal hypothesis tests, not just visual inspection |
| Classical ARIMA/SARIMA | Modern alternatives: `Prophet` (Meta, handles multiple seasonalities/holidays automatically), gradient-boosted trees with lag features (often very strong, simple baselines), and transformer-based time series foundation models (2024-2026 research/production frontier) |

---

## 8. Visual Explanations

**ACF/PACF signature for identifying AR vs. MA order (classical Box-Jenkins diagnostic):**
```
Pure AR(p) process:                    Pure MA(q) process:
ACF: decays gradually                  ACF: cuts off sharply after lag q
PACF: cuts off sharply after lag p     PACF: decays gradually
   │█                                     │█ █
   │█ █                                   │
   │█ █ █ █ ▄ ▄ ▁ ▁                       │  (near zero after lag q)
   └────────────── lag                    └────────────── lag
```

**Walk-forward validation (correctly respecting time order) vs. naive k-fold (leakage):**
```
Naive k-fold (WRONG for time series):        Walk-forward (CORRECT):
[test][train][train][test][train]            [train][test]
 (test fold can be BEFORE training            [train train][test]
  data in time -- leakage!)                   [train train train][test]
                                               (test ALWAYS strictly after training)
```

---

## 9. Practical Examples

**Simple:** compute and plot the ACF/PACF for a synthetic AR(2) process and confirm the PACF cuts off after lag 2.
**Medium:** apply the ADF test to a raw and differenced price series, confirming differencing induces stationarity.
**Real-world:** apply proper walk-forward validation (never a random train/test split) to your Brent crude oil or DZD exchange-rate forecasting model, comparing ARIMA against a naive random-walk baseline and an XGBoost model using engineered lag features (Phase 2 Lesson 6) — a direct, rigorous re-evaluation of existing work using this lesson's correct validation methodology.

---

## 10. Real Industry Use Cases

- **Central banks and financial institutions**: ARIMA/SARIMA and their modern successors remain standard tools for macroeconomic and exchange-rate forecasting, directly relevant to your DZD work.
- **Commodity trading desks**: combine classical time series models with ML approaches (gradient-boosted trees on engineered lag/technical-indicator features) for price forecasting, always benchmarked rigorously against naive/random-walk baselines.
- **Retail demand forecasting** (Amazon, Walmart): `Prophet`-style decomposition models and, increasingly, deep learning (temporal fusion transformers, foundation time series models) handle millions of individual product-level series simultaneously.
- **Actuarial reserving**: claims development patterns over time are themselves time series problems, often modeled via specialized actuarial "chain-ladder" methods that share conceptual DNA with the trend/seasonality decomposition covered here.

---

## 11. Common Mistakes

- Using ordinary random train/test splitting (or k-fold CV) on time series data — a severe, easily-overlooked leakage error (Section 3) that inflates apparent performance dramatically and misleadingly.
- Fitting ARIMA on a non-stationary series without differencing (or without SARIMA's seasonal handling on seasonal data) — parameter estimates and forecasts become unreliable.
- Failing to compare against a naive random-walk baseline — a common way sophisticated-looking models get deployed while actually performing no better (or worse) than "tomorrow's price = today's price."
- Over-interpreting ACF/PACF plots' exact cutoff points as unambiguous — real data is noisy, and model order selection in practice usually also involves information criteria (AIC/BIC) and out-of-sample validation, not just visual pattern-matching.

---

## 12. Best Practices (2026)

- Always use walk-forward (time-series-respecting) validation, never ordinary k-fold CV, for any time-ordered data.
- Always benchmark against the naive forecast baseline — it's a genuinely strong, hard-to-beat competitor for many financial series, and its inclusion is now a standard expectation in any credible forecasting report.
- Consider modern alternatives (gradient-boosted trees with rich lag/rolling-window features, `Prophet` for series with multiple seasonalities and known events/holidays, or transformer-based foundation models for time series when many related series are available) alongside classical ARIMA — no single approach dominates universally across different series characteristics.
- Formally test stationarity (ADF/KPSS) rather than relying purely on visual inspection before committing to a differencing order.

---

## 13. Exercises

**Easy:** Simulate an AR(1) process with $\phi = 0.7$ and fit it using the from-scratch `fit_ar1` (Section 6), verifying the estimated $\phi$ is close to the true value.
**Medium:** Compute ACF and PACF for a simulated MA(2) process and confirm the ACF cuts off after lag 2 while PACF decays gradually.
**Hard:** Implement walk-forward cross-validation from scratch for an ARIMA model and empirically show the performance difference (likely inflated, wrongly optimistic) if ordinary random k-fold CV were used instead on the same time series.
**Mathematical:** Derive the stationarity condition for an AR(1) process ($|\phi| < 1$) by analyzing the process's variance as $t \to \infty$.
**Coding:** Build an XGBoost time series forecaster using engineered lag and rolling-window features (Phase 2 Lesson 6), evaluated via walk-forward validation, and compare its performance against ARIMA and the naive baseline on the same series.

---

## 14. Mini Project

Perform a **complete, rigorous re-analysis of your Brent crude oil (or DZD exchange rate) forecasting project**: test for stationarity (ADF), examine ACF/PACF to inform ARIMA order selection, fit ARIMA/SARIMA alongside an XGBoost model using engineered lag features, evaluate all models via proper walk-forward validation (never random splitting) with the naive random-walk forecast as a mandatory baseline, and write a final assessment of which approach genuinely adds forecasting value beyond the naive baseline, with honest confidence intervals on the performance differences (bootstrap, Phase 3 Lesson 4, adapted for time-dependent data via the block bootstrap).

---

## 15. Interview Preparation

- Why is ordinary k-fold cross-validation inappropriate for time series data, and what should be used instead?
- Explain stationarity and why it matters for classical time series models like ARIMA.
- What is the random walk hypothesis, and why is the naive forecast such a strong baseline for financial time series?
- Explain the ACF/PACF-based approach to identifying appropriate AR and MA orders.

---

## 16. Summary

Time series analysis requires abandoning the i.i.d. assumption underlying most of this phase's earlier lessons: stationarity (often achieved via differencing) is the prerequisite for classical ARIMA modeling, ACF/PACF diagnostics guide model order selection, and — most critically for practical correctness — walk-forward validation (never ordinary k-fold CV) is mandatory to avoid severe temporal data leakage. The naive random-walk baseline is a deceptively strong competitor that every more sophisticated model, classical or ML-based, must genuinely outperform to justify its added complexity — a discipline directly applicable to re-evaluating your existing Brent oil and DZD forecasting work with full rigor.

---

## 17. References

- Box, G.E.P. & Jenkins, G.M. — *Time Series Analysis: Forecasting and Control* (the foundational ARIMA reference)
- Hyndman, R.J. & Athanasopoulos, G. — *Forecasting: Principles and Practice* (free online, modern and highly practical)
- Taylor, S.J. & Letham, B. — "Forecasting at Scale" (2018, the Prophet paper)
- Bergmeir, C. & Benítez, J.M. — "On the Use of Cross-Validation for Time Series Predictor Evaluation" (2012, the key paper on walk-forward validation's necessity)
