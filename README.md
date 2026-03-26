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

### Per-Model Speed Benchmarks (100 real-world datasets)

#### AutoNaive

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 48402.79 | 83881.03 | 32.56% | 41.87 | 51.48 | 20.43% | 0.00 | 100 |
| fast | 46825.06 | 82219.58 | 31.01% | 39.12 | 47.55 | 19.36% | 0.00 | 100 |
| normal | 55665.87 | 82470.84 | 29.79% | 27.33 | 33.06 | 17.35% | 0.00 | 100 |
| slow | 55665.62 | 82470.65 | 29.85% | 33.33 | 40.96 | 16.32% | 0.00 | 100 |
| super_slow | 55666.08 | 82471.08 | 29.84% | 33.33 | 40.96 | 16.32% | 0.00 | 100 |

#### AutoThetaAR

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 66231.58 | 105091.88 | 28.39% | 38.03 | 43.80 | 15.98% | 0.01 | 100 |
| fast | 66231.58 | 105091.88 | 28.39% | 38.03 | 43.80 | 15.98% | 0.01 | 100 |
| normal | 66231.58 | 105091.88 | 28.39% | 38.03 | 43.80 | 15.98% | 0.01 | 100 |
| slow | 66231.58 | 105091.88 | 28.39% | 38.03 | 43.80 | 15.98% | 0.01 | 100 |
| super_slow | 66231.58 | 105091.88 | 28.39% | 38.03 | 43.80 | 15.98% | 0.01 | 100 |

#### AutoKNN

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 45495.17 | 81223.33 | 30.94% | 37.34 | 43.61 | 17.77% | 0.01 | 100 |
| fast | 46081.70 | 81754.65 | 32.17% | 39.10 | 45.58 | 19.54% | 0.04 | 100 |
| normal | 44846.01 | 80618.64 | 32.61% | 38.31 | 47.24 | 19.55% | 0.06 | 100 |
| slow | 45077.83 | 81224.90 | 31.70% | 40.49 | 51.34 | 18.16% | 0.19 | 99 |
| super_slow | 45711.61 | 81671.07 | 30.70% | 40.49 | 51.34 | 18.02% | 0.60 | 99 |

#### AutoPDEField

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 71364.72 | 113802.04 | 34.27% | 46.73 | 54.63 | 16.75% | 0.00 | 100 |
| fast | 62836.93 | 100782.31 | 34.26% | 46.73 | 54.63 | 16.66% | 0.01 | 100 |
| normal | 61394.96 | 98643.02 | 32.35% | 30.15 | 36.51 | 15.19% | 0.14 | 100 |
| slow | 59601.86 | 96607.74 | 32.43% | 30.10 | 36.19 | 14.63% | 0.22 | 100 |
| super_slow | 59880.46 | 96029.28 | 31.13% | 30.90 | 36.42 | 14.23% | 3.22 | 100 |

#### AutoFourier

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 55563.07 | 80172.78 | 33.44% | 57.76 | 68.20 | 22.97% | 0.00 | 100 |
| fast | 51677.30 | 76308.97 | 36.55% | 57.76 | 68.20 | 23.72% | 0.00 | 100 |
| normal | 52139.33 | 79995.06 | 35.59% | 61.08 | 66.96 | 23.36% | 0.00 | 100 |
| slow | 49132.03 | 77423.79 | 35.22% | 61.08 | 66.96 | 23.36% | 0.01 | 100 |
| super_slow | 48872.80 | 77122.95 | 33.60% | 61.08 | 66.96 | 23.36% | 0.02 | 100 |

#### AutoGreensKernel

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 72348.83 | 99254.58 | 51.74% | 111.93 | 126.21 | 35.64% | 0.01 | 100 |
| fast | 70411.59 | 96782.30 | 50.11% | 104.02 | 112.41 | 34.89% | 0.03 | 100 |
| normal | 50902.81 | 77641.40 | 34.90% | 47.15 | 56.76 | 22.73% | 0.21 | 100 |
| slow | 49221.30 | 76075.91 | 34.15% | 45.95 | 55.78 | 21.83% | 0.61 | 100 |
| super_slow | 49778.84 | 78193.66 | 33.07% | 44.38 | 54.47 | 20.69% | 6.14 | 100 |

#### AutoSpectralGradient

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 52722.21 | 84979.30 | 33.96% | 58.04 | 67.81 | 19.47% | 0.00 | 100 |
| fast | 44047.84 | 74975.53 | 33.86% | 51.63 | 63.20 | 19.38% | 0.01 | 100 |
| normal | 51565.84 | 80725.74 | 33.57% | 49.33 | 60.86 | 18.60% | 0.32 | 100 |
| slow | 59388.87 | 86750.79 | 35.76% | 53.36 | 64.10 | 21.49% | 0.92 | 100 |
| super_slow | 59750.35 | 86300.03 | 36.95% | 52.61 | 58.78 | 21.71% | 13.32 | 100 |

#### AutoFracDiff

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 135625.12 | 165178.89 | 85.97% | 501.87 | 571.23 | 86.19% | 0.01 | 100 |
| fast | 130154.81 | 157580.15 | 81.32% | 455.90 | 483.54 | 81.51% | 0.02 | 100 |
| normal | 95424.54 | 126662.35 | 56.29% | 254.84 | 256.02 | 51.90% | 0.30 | 100 |
| slow | 85441.64 | 119020.80 | 38.24% | 111.51 | 115.23 | 31.00% | 1.13 | 100 |
| super_slow | 87305.06 | 121693.81 | 36.59% | 94.22 | 98.15 | 27.19% | 7.01 | 100 |

#### AutoKoopman

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 45411.65 | 72908.35 | 33.11% | 35.13 | 43.04 | 19.67% | 0.00 | 100 |
| fast | 44888.15 | 72263.73 | 33.14% | 34.39 | 41.02 | 21.42% | 0.01 | 100 |
| normal | 52062.77 | 81243.92 | 29.71% | 29.11 | 34.77 | 18.06% | 0.77 | 100 |
| slow | 52822.65 | 82219.21 | 30.71% | 28.06 | 34.25 | 18.18% | 3.75 | 100 |
| super_slow | 50844.52 | 79799.29 | 28.94% | 35.61 | 44.43 | 17.06% | 28.62 | 100 |

#### AutoLocalLinear

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 80495.97 | 113829.66 | 35.81% | 33.08 | 40.21 | 15.17% | 0.01 | 100 |
| fast | 45131.52 | 70541.57 | 32.41% | 28.64 | 35.57 | 15.60% | 0.01 | 100 |
| normal | 75742.64 | 104015.39 | 39.67% | 31.57 | 37.60 | 15.31% | 0.03 | 100 |
| slow | 124413.16 | 151654.08 | 48.26% | 31.57 | 37.60 | 18.08% | 0.06 | 100 |
| super_slow | 168910.25 | 206433.85 | 49.40% | 46.44 | 52.03 | 25.45% | 0.12 | 100 |

#### AutoPolymath

| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |
|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|
| super_fast | 49431.52 | 81177.78 | 26.29% | 33.47 | 38.35 | 11.78% | 0.00 | 100 |
| fast | 67358.56 | 102976.44 | 27.13% | 27.62 | 32.61 | 11.43% | 0.05 | 95 |
| normal | 83892.45 | 139210.80 | 27.52% | 24.51 | 32.43 | 12.03% | 2.97 | 88 |
| slow | 84919.11 | 137332.12 | 29.43% | 24.29 | 32.35 | 12.59% | 6.15 | 90 |
| super_slow | 188379.08 | 377591.15 | 29.79% | 34.81 | 41.63 | 13.12% | 31.60 | 84 |

---

## Metrics

```python
from randomstatsmodels.metrics import mae, mse, rmse, mape, smape
```

The evaluation framework also provides MASE, MSSE, and Median Absolute Error.

---

## License

MIT