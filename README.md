# randomstatsmodels

Check out medium story here: [Medium Story](https://medium.com/@jacoblouiswright/univarient-forecasting-models-2025-c483d04f04d8)

Lightweight univariate time-series forecasting with auto-tuned models — **NumPy only**, no heavy dependencies.

19 forecasting models with a unified `.fit(y)` / `.predict(h)` API. Each `Auto*` wrapper grid-searches hyperparameters on a validation split and refits the best configuration on the full series.

## Installation

```bash
pip install randomstatsmodels
```

Requires: Python 3.9+ and NumPy.

---

## Quick Start

```python
from randomstatsmodels import AutoKoopman, AutoPolymath, AutoSSA
import numpy as np

rng = np.random.default_rng(42)
t = np.arange(200)
y = 10 + 0.05 * t + np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=t.size)

h = 12  # forecast horizon

model = AutoKoopman().fit(y)
yhat = model.predict(h)
print("Forecast:", yhat[:5])
print("Best config:", model.best_)
```

---

## Models

19 models organised by approach. Every `Auto*` class accepts a parameter grid, fits/evaluates candidates, and exposes `.fit(y)` and `.predict(h)`.

### Classical

| Model | Description |
|-------|-------------|
| `AutoNaive` | Baselines: last value, seasonal, drift, mean |
| `AutoHoltWinters` | Triple exponential smoothing (level + trend + seasonal) |
| `AutoThetaAR` | Theta method with AR(1) residual correction |
| `AutoFourier` | Harmonic regression with optional linear trend |

### Regression-Based

| Model | Description |
|-------|-------------|
| `AutoNEO` | Nonlinear Evolution Operator — polynomial AR features |
| `AutoPolymath` | Polynomial + Fourier basis + ridge regression |
| `AutoLocalLinear` | Weighted local regression with exponential decay |
| `AutoPALF` | Proximal Aggregation Lag Forecaster — penalised lag weighting |

### Decomposition / Spectral

| Model | Description |
|-------|-------------|
| `AutoSSA` | Singular Spectrum Analysis — SVD on trajectory matrix |
| `AutoKoopman` | Dynamic Mode Decomposition / Koopman operator via delay embedding |
| `AutoSpectralGradient` | Spectral derivative flow — extrapolates Fourier mode dynamics |

### Advanced / Calculus-Based

| Model | Description |
|-------|-------------|
| `AutoFracDiff` | Fractional calculus — Grunwald-Letnikov fractional differencing + AR |
| `AutoGreensKernel` | Integral equation — Green's function convolution kernel |
| `AutoPDEField` | Partial differential equations — advection-diffusion on time-scale field |
| `AutoVariationalPath` | Calculus of variations — Euler-Lagrange optimal path |

### Hybrid / Meta

| Model | Description |
|-------|-------------|
| `AutoHybridForecaster` | Linear (Fourier + trend + AR) + GRU residual network |
| `AutoMELD` | Multiscale embedding with Random Fourier Features + kNN |
| `AutoRIFT` | Recursive Information Flow Tensor — information-channel dynamics |
| `AutoEnsemble` | Combines multiple base forecasters with learned weights |

### Advanced Ensembles

| Model | Description |
|-------|-------------|
| `AutoStacked` | Meta-learner stacking — ridge regression on base model predictions |
| `AutoBagged` | Block-bootstrap bagging — median of models trained on resampled series |
| `AutoDynamic` | Horizon-adaptive weighting — model weights change per forecast step |

---

## Benchmarks

22 models evaluated on **20 real-world time series** (10 from FRED with 300-600 points, 10 classic hardcoded) using `speed="slow"` grids. Ranked by RMSE per dataset. Includes 5 statsforecast baselines. AutoRIFT and AutoHybridForecaster excluded from fast benchmark (>30s fit time per dataset).

### Overall Rankings

| Rank | Model | Type | Avg Rank | #1st | #Top3 | #Top7 | Median MAE | Median RMSE | Median MAPE | Median sMAPE |
|------|-------|------|----------|------|-------|-------|-----------|------------|------------|-------------|
| 1 | SF_AutoTBATS | statsforecast | 7.30 | 1 | 5 | 10 | 18.28 | 23.53 | 7.66% | 7.77% |
| 2 | SF_AutoARIMA | statsforecast | 8.00 | 1 | 3 | 10 | 18.03 | 23.29 | 7.14% | 7.65% |
| 3 | AutoNEO | randomstatsmodels | 8.94 | 2 | 4 | 8 | 25.86 | 36.51 | 7.38% | 7.59% |
| 4 | AutoMELD | randomstatsmodels | 8.95 | 2 | 4 | 7 | 20.80 | 27.83 | 7.79% | 7.84% |
| 5 | SF_AutoETS | statsforecast | 9.35 | 0 | 0 | 9 | 33.65 | 38.18 | 11.53% | 12.08% |
| 6 | SF_AutoCES | statsforecast | 9.80 | 5 | 6 | 9 | 18.96 | 23.27 | 10.70% | 10.71% |
| 7 | AutoNaive | randomstatsmodels | 10.05 | 1 | 1 | 7 | 24.43 | 30.87 | 12.94% | 13.68% |
| 8 | AutoKoopman | randomstatsmodels | 10.75 | 0 | 5 | 5 | 31.60 | 40.47 | 12.63% | 12.59% |
| 9 | AutoPolymath | randomstatsmodels | 10.84 | 0 | 4 | 6 | 22.65 | 29.33 | 9.48% | 9.00% |
| 10 | SF_AutoTheta | statsforecast | 11.00 | 0 | 3 | 7 | 20.07 | 25.67 | 8.97% | 9.27% |
| 11 | AutoPALF | randomstatsmodels | 11.00 | 2 | 2 | 7 | 42.30 | 49.53 | 12.77% | 13.84% |
| 12 | AutoKNN | randomstatsmodels | 11.05 | 1 | 2 | 8 | 35.92 | 43.24 | 12.71% | 13.89% |
| 13 | AutoPDEField | randomstatsmodels | 11.25 | 0 | 2 | 5 | 36.12 | 41.58 | 11.16% | 11.66% |
| 14 | AutoSSA | randomstatsmodels | 11.75 | 0 | 3 | 6 | 31.47 | 38.00 | 12.24% | 13.28% |
| 15 | AutoThetaAR | randomstatsmodels | 11.80 | 0 | 1 | 5 | 38.03 | 43.81 | 15.11% | 16.76% |
| 16 | AutoHoltWinters | randomstatsmodels | 12.20 | 2 | 2 | 7 | 49.16 | 60.66 | 14.24% | 14.63% |
| 17 | AutoGreensKernel | randomstatsmodels | 12.60 | 1 | 3 | 6 | 40.53 | 48.61 | 15.87% | 16.51% |
| 18 | AutoVariationalPath | randomstatsmodels | 12.80 | 0 | 3 | 5 | 57.52 | 67.07 | 13.10% | 13.52% |
| 19 | AutoSpectralGradient | randomstatsmodels | 13.25 | 1 | 3 | 4 | 54.54 | 64.10 | 15.53% | 15.02% |
| 20 | AutoFourier | randomstatsmodels | 14.15 | 1 | 3 | 4 | 69.65 | 83.66 | 22.71% | 22.45% |
| 21 | AutoLocalLinear | randomstatsmodels | 15.10 | 0 | 1 | 5 | 56.44 | 63.38 | 20.47% | 25.01% |
| 22 | AutoFracDiff | randomstatsmodels | 18.60 | 0 | 0 | 0 | 110.80 | 113.95 | 30.43% | 36.00% |

### Speed Presets

Every Auto* model accepts a `speed` parameter controlling grid search thoroughness:

```python
from randomstatsmodels import AutoKoopman

model = AutoKoopman(speed="super_fast")   # ~1 combo, seconds
model = AutoKoopman(speed="fast")         # ~6 combos
model = AutoKoopman(speed="normal")       # default grids
model = AutoKoopman(speed="slow")         # ~864 combos
model = AutoKoopman(speed="super_slow")   # ~2640 combos
```

### Datasets

20 real-world datasets (10 from FRED with 300-600 points, 10 classic):

| Source | Datasets |
|--------|----------|
| FRED (300-600 pts) | USUnemployment, USConsumerPrices, USIndProdIndex, USHousingStarts, USMoneySupply, FedFundsRate, USRetailSales, USElectricity, SP500, US10YrTreasury |
| Classic (89-168 pts) | MilkProduction, GoldPrice, NileMinLevel, AirPassengers, CO2, GlobalTemp, IntlAirline, Lynx, Nile, Sunspots |

### Key Findings

- **SF_AutoTBATS** is #1 by RMSE — strong on seasonal FRED data
- **SF_AutoARIMA** is #2 — most consistent across dataset types
- **AutoNEO** is the best randomstatsmodels model (#3) — polynomial AR features
- **AutoMELD** is #4 — multiscale embedding excels on longer series
- **AutoKoopman** (#8) — DMD eigenvalue propagation, 5 top-3 finishes
- **No single model dominates** — model selection matters for your data type

### Per-Model Speed Benchmarks (100 real-world datasets, optimized v4.1)

#### AutoNaive

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 48440.59 | 83895.75 | 32.57% | 41.87 | 51.48 | 20.43% | 0.00 | 100 |
| fast | 46862.86 | 82234.30 | 31.02% | 39.12 | 47.55 | 19.36% | 0.00 | 100 |
| normal | 55666.28 | 82470.79 | 29.79% | 27.33 | 33.06 | 17.35% | 0.00 | 100 |
| slow | 55666.04 | 82470.60 | 29.85% | 33.33 | 40.96 | 16.32% | 0.00 | 100 |
| super_slow | 55666.49 | 82471.03 | 29.84% | 33.33 | 40.96 | 16.32% | 0.00 | 100 |

#### AutoThetaAR

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 59734.22 | 100616.79 | 25.07% | 31.75 | 38.25 | 15.65% | 0.01 | 89 |
| fast | 59734.22 | 100616.79 | 25.07% | 31.75 | 38.25 | 15.65% | 0.02 | 89 |
| normal | 59734.22 | 100616.79 | 25.07% | 31.75 | 38.25 | 15.65% | 0.01 | 89 |
| slow | 59734.22 | 100616.79 | 25.07% | 31.75 | 38.25 | 15.65% | 0.02 | 89 |
| super_slow | 59734.22 | 100616.79 | 25.07% | 31.75 | 38.25 | 15.65% | 0.02 | 89 |

#### AutoKNN

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 141.63 | 170.22 | 33.37% | 21.72 | 26.96 | 14.00% | 0.00 | 22 |
| fast | 141.84 | 173.49 | 34.04% | 28.46 | 33.81 | 12.63% | 0.01 | 22 |
| normal | 162.47 | 185.92 | 35.97% | 17.40 | 22.40 | 12.81% | 0.01 | 22 |
| slow | 151.96 | 179.56 | 35.32% | 25.24 | 31.73 | 13.05% | 0.02 | 21 |
| super_slow | 138.83 | 165.14 | 31.64% | 20.87 | 24.37 | 13.05% | 0.07 | 21 |

#### AutoPDEField

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 782.58 | 987.40 | 39.86% | 38.46 | 44.41 | 28.94% | 0.00 | 22 |
| fast | 862.98 | 1070.32 | 40.49% | 38.46 | 44.41 | 30.49% | 0.00 | 22 |
| normal | 809.10 | 893.87 | 38.63% | 27.90 | 33.07 | 28.05% | 0.04 | 22 |
| slow | 809.39 | 894.12 | 36.69% | 27.57 | 32.56 | 25.64% | 0.06 | 22 |
| super_slow | 489.65 | 551.15 | 35.70% | 28.38 | 33.59 | 25.02% | 0.78 | 22 |

#### AutoFourier

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 333.49 | 392.03 | 32.90% | 40.88 | 47.43 | 19.89% | 0.00 | 22 |
| fast | 333.53 | 392.82 | 33.82% | 38.81 | 45.37 | 19.92% | 0.00 | 22 |
| normal | 307.05 | 373.76 | 34.42% | 34.47 | 40.43 | 24.43% | 0.00 | 22 |
| slow | 303.04 | 367.82 | 34.10% | 34.47 | 40.43 | 19.47% | 0.00 | 22 |
| super_slow | 310.10 | 377.30 | 34.76% | 34.47 | 40.43 | 24.92% | 0.01 | 22 |

#### AutoGreensKernel

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 293.93 | 360.27 | 43.61% | 20.59 | 25.07 | 26.09% | 0.00 | 22 |
| fast | 288.15 | 351.86 | 42.06% | 20.46 | 25.11 | 25.72% | 0.00 | 22 |
| normal | 274.41 | 342.24 | 38.22% | 21.18 | 25.83 | 17.46% | 0.03 | 22 |
| slow | 288.81 | 351.23 | 36.52% | 20.62 | 24.86 | 14.98% | 0.07 | 22 |
| super_slow | 227.05 | 275.98 | 33.16% | 21.08 | 25.80 | 13.17% | 0.75 | 22 |

#### AutoSpectralGradient

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 271.50 | 348.52 | 38.37% | 51.17 | 58.79 | 22.70% | 0.00 | 22 |
| fast | 416.19 | 499.78 | 38.36% | 46.10 | 52.42 | 24.04% | 0.00 | 22 |
| normal | 317.41 | 391.21 | 33.80% | 53.35 | 61.39 | 16.28% | 0.05 | 22 |
| slow | 409.29 | 476.95 | 38.59% | 46.29 | 54.66 | 21.05% | 0.15 | 22 |
| super_slow | 464.12 | 553.02 | 36.83% | 38.95 | 45.73 | 18.30% | 2.16 | 22 |

#### AutoFracDiff

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 2193.28 | 2520.65 | 181.40% | 501.87 | 571.23 | 183.74% | 0.00 | 22 |
| fast | 1930.98 | 2087.62 | 181.73% | 455.90 | 483.54 | 184.46% | 0.00 | 22 |
| normal | 895.57 | 954.82 | 97.80% | 254.84 | 256.02 | 91.12% | 0.07 | 22 |
| slow | 541.99 | 570.95 | 56.25% | 110.80 | 113.95 | 42.22% | 0.20 | 22 |
| super_slow | 390.52 | 429.68 | 48.24% | 87.38 | 88.91 | 29.98% | 1.28 | 22 |

---

## Metrics

```python
from randomstatsmodels.metrics import mae, mse, rmse, mape, smape
```

The evaluation framework also provides MASE, MSSE, and Median Absolute Error.

---

## License

MIT