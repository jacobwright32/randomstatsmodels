# randomstatsmodels
Check out medium story here: [Medium Story](https://medium.com/@jacoblouiswright/univarient-forecasting-models-2025-c483d04f04d8) <br></br>
Lightweight utilities for benchmarking, forecasting, and statistical modeling — with simple `Auto*` model wrappers that tune hyperparameters for you.

## Installation

```bash
pip install randomstatsmodels
```

Requires: Python 3.9+ and NumPy.

---

## Quick Start

```python
from randomstatsmodels import AutoNEO, AutoFourier, AutoKNN, AutoPolymath, AutoThetaAR
import numpy as np

# Toy data: sine wave + noise
rng = np.random.default_rng(42)
t = np.arange(200)
y = np.sin(2*np.pi*t/24) + 0.1*rng.normal(size=t.size)

h = 12  # forecast horizon

model = AutoNEO().fit(y)
yhat = model.predict(h)
print("Forecast:", yhat[:5])
```

---

## Models

Each `Auto*` class:
- accepts a **parameter grid** (or uses sensible defaults),
- fits/evaluates candidates using a chosen metric,
- exposes a unified API: `.fit(y[, X])` and `.predict(h)`.

### AutoNEO

```python
from randomstatsmodels import AutoNEO

neo = AutoNEO(
    param_grid={"n_components": [8, 16, 32]},
    metric="mae",
)
neo.fit(y)
print("Best params:", neo.best_params_)
print("Prediction:", neo.predict(h))
```

### AutoFourier

```python
from randomstatsmodels import AutoFourier

fourier = AutoFourier(
    param_grid={"season_length": [12, 24], "n_terms": [3, 5]},
    metric="smape",
)
fourier.fit(y)
print("Prediction:", fourier.predict(h))
```

### AutoKNN

```python
from randomstatsmodels import AutoKNN

knn = AutoKNN(
    param_grid={"k": [3, 5, 7], "window": [12, 24]},
    metric="rmse",
)
knn.fit(y)
print("Prediction:", knn.predict(h))
```

### AutoPolymath

```python
from randomstatsmodels import AutoPolymath

poly = AutoPolymath(
    param_grid={"degree": [2, 3], "ridge": [0.0, 0.1]},
    metric="mae",
)
poly.fit(y)
print("Prediction:", poly.predict(h))
```

### AutoThetaAR

```python
from randomstatsmodels import AutoThetaAR

theta = AutoThetaAR(
    param_grid={"theta": [0.5, 1.0, 2.0]},
    metric="mape",
)
theta.fit(y)
print("Prediction:", theta.predict(h))
```

### AutoHybridForecaster

```python
from randomstatsmodels import AutoHybridForecaster

hybrid = AutoHybridForecaster(
    candidate_fourier=(0, 3, 6),
    candidate_trend=(0, 1),
    candidate_ar=(0, 3, 5),
    candidate_hidden=(8, 16, 32),
)
hybrid.fit(y)
print("Best config:", hybrid.best_config)
print("Prediction:", hybrid.predict(h))
```

### AutoMELD

```python
from randomstatsmodels import AutoMELD

meld = AutoMELD(
    lags_grid=(8, 12),
    scales_grid=((1, 3, 7), (1, 2, 4, 8)),
    rff_features_grid=(64, 128),
)
meld.fit(y)
print("Best config:", meld.best_["config"])
print("Prediction:", meld.predict(h))
```

### AutoPALF

```python
from randomstatsmodels import AutoPALF

palf = AutoPALF(
    p_candidates=(4, 8, 12),
    penalties=("huber", "l2"),
)
palf.fit(y)
print("Validation score:", palf.best_["val_score"])
print("Prediction:", palf.predict(h))
```

### AutoNaive

Essential baseline forecasters for proper model evaluation.

```python
from randomstatsmodels import AutoNaive

naive = AutoNaive(
    method_options=("last", "seasonal", "drift", "mean"),
    seasonal_periods=(1, 7, 12, 24),
)
naive.fit(y)
print("Best config:", naive.best_["config"])
print("Prediction:", naive.predict(h))
```

Methods:
- `"last"`: Repeat the last observed value
- `"seasonal"`: Repeat values from one seasonal period ago
- `"drift"`: Linear extrapolation from first to last value
- `"mean"`: Rolling or global mean

### AutoHoltWinters

Classic Holt-Winters exponential smoothing with level, trend, and seasonal components.

```python
from randomstatsmodels import AutoHoltWinters

hw = AutoHoltWinters(
    seasonal_periods=(12, 24),
    trend_options=("add", "none", "damped"),
    seasonal_options=("add", "none"),
)
hw.fit(y)
print("Best config:", hw.best_["config"])
print("Prediction:", hw.predict(h))
```

### AutoSSA

Singular Spectrum Analysis - decomposes time series using SVD to discover adaptive oscillatory modes.

```python
from randomstatsmodels import AutoSSA

ssa = AutoSSA(
    window_fracs=(0.25, 0.33, 0.5),
    n_components_grid=(None, 2, 4, 8),
)
ssa.fit(y)
print("Best config:", ssa.best_["config"])
print("Prediction:", ssa.predict(h))
```

### AutoLocalLinear

Weighted local regression with exponential decay for older observations.

```python
from randomstatsmodels import AutoLocalLinear

ll = AutoLocalLinear(
    decay_grid=(0.9, 0.95, 0.98, 1.0),
    degree_grid=(1, 2),
)
ll.fit(y)
print("Best config:", ll.best_["config"])
print("Prediction:", ll.predict(h))
```

### AutoEnsemble

Combines multiple base forecasters with learned weights using validation performance.

```python
from randomstatsmodels import AutoEnsemble

ensemble = AutoEnsemble(
    weighting_options=("uniform", "validation", "optimal"),
)
ensemble.fit(y)
print("Best config:", ensemble.best_["config"])
print("Base model scores:", ensemble.base_scores_)
print("Prediction:", ensemble.predict(h))
```

Weighting methods:
- `"uniform"`: Equal weights for all models
- `"validation"`: Weights inversely proportional to validation error
- `"optimal"`: Solve for weights that minimize validation error

### AutoRIFT (Novel: Recursive Information Flow Tensor)

**A cutting-edge forecasting model based on original "Predictive Information Field Dynamics" theory.**

RIFT introduces a fundamentally new paradigm: instead of modeling values directly, it models how *predictive information* flows and transforms between different temporal channels (level, trend, curvature, oscillations) as the forecast horizon increases.

```python
from randomstatsmodels import AutoRIFT

rift = AutoRIFT(
    n_frequencies_grid=(2, 4, 6),
    embedding_dim_grid=(2, 3),
    regularization_grid=(0.001, 0.01),
)
rift.fit(y)
print("Best config:", rift.best_["config"])
print("Prediction:", rift.predict(h))

# Analyze which channels hold predictive information
info = rift.get_information_analysis(horizon=5)
print("Information by channel:", info)
```

**Theoretical Innovation:**
- **Information Channels**: Decomposes predictive power into orthogonal channels (level, trend, curvature, spectral components)
- **Information Flow Matrix**: Learns how information transfers between channels as horizon increases
- **Fisher Information Estimation**: Uses local variance reduction to estimate channel informativeness
- **Adaptive Reconstruction**: Combines channel extrapolations weighted by propagated information state

---

## Benchmarks

Comprehensive evaluation of all 13 models across 12 diverse time series datasets with 12-step ahead forecasting.

### Overall Model Rankings

| Model | Avg Rank | #1 Finishes | Top 3 Finishes |
|-------|----------|-------------|----------------|
| **AutoPolymath** | **4.4** | 2 | 6 |
| AutoLocalLinear | 4.9 | 3 | 7 |
| AutoNEO | 5.2 | 1 | 4 |
| AutoMELD | 5.3 | 1 | 3 |
| AutoNaive | 6.0 | 1 | 2 |
| AutoSSA | 6.0 | 2 | 4 |
| AutoHoltWinters | 6.7 | 1 | 2 |
| AutoKNN | 7.5 | 0 | 3 |
| AutoEnsemble | 8.5 | 0 | 1 |
| AutoPALF | 8.8 | 1 | 1 |
| AutoThetaAR | 8.8 | 0 | 1 |
| AutoFourier | 9.1 | 0 | 1 |
| AutoRIFT | 9.9 | 0 | 1 |

### 1. Airline Passengers
_Monthly airline passengers (1949-1960). Trend + multiplicative seasonality._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoLocalLinear | 13.43 | 17.30 |
| 2 | AutoPolymath | 14.39 | 17.44 |
| 3 | AutoNEO | 15.85 | 18.89 |

### 2. Sunspots
_Monthly sunspot numbers. Cyclical, stationary._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoPALF | 5.65 | 7.05 |
| 2 | **AutoRIFT** | **5.73** | **8.05** |
| 3 | AutoPolymath | 5.91 | 7.10 |

### 3. Milk Production
_Monthly milk production per cow (1962-1975). Trend + seasonality._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoNaive | 11.58 | 14.19 |
| 2 | AutoKNN | 17.47 | 22.11 |
| 3 | AutoLocalLinear | 31.88 | 32.67 |

### 4. CO2 Mauna Loa
_Monthly atmospheric CO2 (ppm). Strong trend + seasonality._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoSSA | 0.18 | 0.25 |
| 2 | AutoHoltWinters | 0.20 | 0.22 |
| 3 | AutoLocalLinear | 0.35 | 0.39 |

### 5. Beer Production
_Quarterly Australian beer production. Strong seasonality._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoMELD | 22.51 | 26.79 |
| 2 | AutoLocalLinear | 28.33 | 35.06 |
| 3 | AutoKNN | 29.13 | 33.32 |

### 6. Car Sales
_Monthly car sales Quebec (1960-1968). Trend + seasonality._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoLocalLinear | 1435.64 | 2096.46 |
| 2 | AutoPolymath | 1462.57 | 1801.44 |
| 3 | AutoKNN | 1585.77 | 2153.23 |

### 7. Daily Temperature
_Daily minimum temperatures (synthetic). Annual cycle + noise._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoNEO | 1.81 | 2.44 |
| 2 | AutoLocalLinear | 1.82 | 2.34 |
| 3 | AutoMELD | 1.84 | 2.51 |

### 8. Synthetic Trend
_Linear trend with Gaussian noise._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoLocalLinear | 1.68 | 2.55 |
| 2 | AutoFourier | 1.68 | 2.56 |
| 3 | AutoSSA | 1.71 | 2.54 |

### 9. Multi-Seasonal
_Synthetic daily data with weekly + yearly patterns._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoSSA | 2.52 | 3.13 |
| 2 | AutoNaive | 5.06 | 6.07 |
| 3 | AutoPolymath | 5.65 | 6.56 |

### 10. Mackey-Glass
_Chaotic time series (tau=17). Tests nonlinear dynamics._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoPolymath | 0.00 | 0.00 |
| 2 | AutoNEO | 0.00 | 0.00 |
| 3 | AutoMELD | 0.01 | 0.02 |

### 11. Random Walk
_Random walk with drift. Non-stationary._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoHoltWinters | 0.39 | 0.55 |
| 2 | AutoThetaAR | 0.44 | 0.60 |
| 3 | AutoEnsemble | 0.47 | 0.64 |

### 12. Damped Sine
_Exponentially damped oscillation._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoPolymath | 0.80 | 0.92 |
| 2 | AutoNEO | 0.85 | 1.01 |
| 3 | AutoSSA | 0.87 | 1.07 |

### Key Findings

- **AutoPolymath** achieves the best average rank (4.4) with strong performance across diverse data types
- **AutoLocalLinear** excels on trending data (3 first-place finishes)
- **AutoRIFT** performs well on cyclical/stationary data (2nd place on Sunspots)
- **AutoSSA** dominates multi-seasonal and smooth trend patterns
- **AutoNEO/AutoPolymath** excel on chaotic dynamics (Mackey-Glass)
- No single model dominates all data types - model selection matters!

---

## Metrics

Available out of the box:

```python
from randomstatsmodels.metrics import mae, mse, rmse, mape, smape
```

---

## License

MIT © 2025 Jacob Wright
