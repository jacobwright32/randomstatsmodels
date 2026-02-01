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

All models benchmarked on two classic time series datasets with 12-step ahead forecasting.

### Airline Passengers Dataset

Monthly airline passenger numbers (1949-1960). Classic Box-Jenkins dataset with trend and seasonality.

| Model | MAE | RMSE |
|-------|-----|------|
| AutoLocalLinear | 13.43 | 17.30 |
| AutoPolymath | 14.39 | 17.44 |
| AutoNEO | 15.85 | 18.89 |
| AutoMELD | 24.98 | 29.53 |
| AutoSSA | 36.20 | 43.06 |
| AutoHoltWinters | 45.02 | 60.23 |
| AutoNaive | 47.83 | 50.71 |
| AutoFourier | 58.66 | 78.82 |
| AutoPALF | 60.07 | 83.03 |
| AutoKNN | 60.32 | 65.23 |
| AutoEnsemble | 61.47 | 85.20 |
| AutoThetaAR | 66.77 | 93.18 |
| AutoRIFT | 130.64 | 155.99 |

### Sunspots Dataset

Monthly sunspot numbers. Classic cyclical dataset without strong trend.

| Model | MAE | RMSE |
|-------|-----|------|
| AutoPALF | 5.65 | 7.05 |
| **AutoRIFT** | **5.73** | **8.05** |
| AutoPolymath | 5.91 | 7.10 |
| AutoNEO | 6.95 | 8.23 |
| AutoThetaAR | 7.40 | 8.51 |
| AutoNaive | 9.74 | 10.86 |
| AutoMELD | 12.15 | 13.96 |
| AutoEnsemble | 12.20 | 13.58 |
| AutoSSA | 13.85 | 16.08 |
| AutoKNN | 14.55 | 17.38 |
| AutoFourier | 18.37 | 19.73 |
| AutoHoltWinters | 24.32 | 25.84 |
| AutoLocalLinear | 30.30 | 32.70 |

**Key Observations:**
- **AutoLocalLinear** excels on trending data (Airline Passengers)
- **AutoRIFT** performs excellently on cyclical/stationary data (Sunspots, 2nd place)
- **AutoPALF** and **AutoPolymath** show consistent performance across both datasets
- Model performance varies significantly by data characteristics - no single model dominates

---

## Metrics

Available out of the box:

```python
from randomstatsmodels.metrics import mae, mse, rmse, mape, smape
```

---

## License

MIT © 2025 Jacob Wright
