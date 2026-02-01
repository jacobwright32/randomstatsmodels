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

Comprehensive evaluation of all 13 models across 12 real-world time series datasets with 12-step ahead forecasting.

### Overall Model Rankings

| Model | Avg Rank | #1 Finishes | Top 3 Finishes |
|-------|----------|-------------|----------------|
| **AutoLocalLinear** | **5.2** | 4 | 5 |
| AutoPolymath | 5.4 | 1 | 6 |
| AutoNaive | 5.8 | 0 | 3 |
| AutoNEO | 5.8 | 0 | 5 |
| AutoSSA | 6.2 | 1 | 3 |
| AutoMELD | 6.2 | 2 | 4 |
| AutoKNN | 6.8 | 1 | 2 |
| AutoThetaAR | 7.8 | 1 | 2 |
| AutoHoltWinters | 7.9 | 0 | 1 |
| AutoPALF | 8.0 | 1 | 1 |
| AutoEnsemble | 8.2 | 0 | 1 |
| AutoRIFT | 8.6 | 1 | 2 |
| AutoFourier | 8.8 | 0 | 1 |

### 1. US GDP Growth
_Quarterly US GDP growth rate (1947-1960). Economic indicator._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoPolymath | 2.68 | 3.71 |
| 2 | AutoFourier | 3.25 | 4.00 |
| 3 | AutoNEO | 3.29 | 4.06 |

### 2. US Unemployment
_Monthly US unemployment rate (1948-1952). Labor market._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoThetaAR | 0.15 | 0.17 |
| 2 | AutoNaive | 0.18 | 0.21 |
| 3 | AutoEnsemble | 0.21 | 0.24 |

### 3. Gold Prices
_Monthly gold prices USD/oz (1979-1984). Commodity prices._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | **AutoRIFT** | **13.68** | **15.90** |
| 2 | AutoHoltWinters | 16.55 | 19.04 |
| 3 | AutoSSA | 22.83 | 26.73 |

### 4. Electricity Production
_Monthly US electricity production (1973-1978). Energy sector._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoMELD | 3.11 | 3.75 |
| 2 | AutoNEO | 4.04 | 4.38 |
| 3 | AutoPolymath | 4.48 | 4.86 |

### 5. Wine Sales
_Monthly Australian wine sales in litres (1980-1985). Retail._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoLocalLinear | 1689.65 | 2074.68 |
| 2 | AutoNEO | 2194.88 | 2553.63 |
| 3 | AutoPolymath | 2195.09 | 2554.02 |

### 6. Lynx Trappings
_Annual Canadian lynx trappings (1821-1900). Ecological cycles._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoKNN | 803.05 | 1158.40 |
| 2 | AutoNaive | 835.83 | 1563.64 |
| 3 | AutoThetaAR | 841.47 | 1566.57 |

### 7. Lake Erie Level
_Monthly Lake Erie water level (1921-1925). Environmental._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoLocalLinear | 0.27 | 0.32 |
| 2 | AutoSSA | 0.27 | 0.35 |
| 3 | AutoMELD | 0.39 | 0.45 |

### 8. US Retail Sales
_Monthly US retail sales index (1967-1971). Economic._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoMELD | 0.62 | 0.90 |
| 2 | AutoPolymath | 0.62 | 0.76 |
| 3 | AutoNEO | 0.82 | 1.07 |

### 9. Australia Passengers
_Monthly international passengers Australia (1991-1994). Travel._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoSSA | 85.23 | 97.95 |
| 2 | AutoMELD | 96.81 | 113.58 |
| 3 | AutoLocalLinear | 113.33 | 120.66 |

### 10. Accidental Deaths
_Monthly US accidental deaths (1973-1978). Public health._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoLocalLinear | 238.24 | 266.90 |
| 2 | AutoNaive | 259.50 | 341.16 |
| 3 | AutoKNN | 259.50 | 341.16 |

### 11. Airline Passengers
_Monthly airline passengers (1949-1960). Classic benchmark._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoLocalLinear | 13.43 | 17.30 |
| 2 | AutoPolymath | 14.39 | 17.44 |
| 3 | AutoNEO | 15.85 | 18.89 |

### 12. Sunspots
_Monthly sunspot numbers. Astronomical cycles._

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | AutoPALF | 5.65 | 7.05 |
| 2 | **AutoRIFT** | **5.73** | **8.05** |
| 3 | AutoPolymath | 5.91 | 7.10 |

### Key Findings

- **AutoLocalLinear** achieves the best average rank (5.2) with 4 first-place finishes
- **AutoPolymath** shows consistent top-3 performance (6 top-3 finishes)
- **AutoRIFT** wins on Gold Prices and places 2nd on Sunspots - excels on financial/cyclical data
- **AutoMELD** dominates on Electricity and Retail Sales
- **AutoKNN** performs best on ecological cycles (Lynx Trappings)
- **AutoSSA** excels on smooth seasonal patterns (Lake Erie, Australia Passengers)
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
