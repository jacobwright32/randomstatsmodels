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

All 27 models evaluated on **36 real-world time series** (20% holdout), ranked by MAE. Includes 5 industry-standard statsforecast baselines for comparison.

### Overall Rankings

Avg Rank is computed by averaging each model's per-dataset rank across all four metrics (MAE, RMSE, MAPE, sMAPE).

| Rank | Model | Type | Avg Rank | #1st | #Top3 | #Top7 | Median MAE | Median RMSE | Median MAPE | Median sMAPE |
|------|-------|------|----------|------|-------|-------|-----------|------------|------------|-------------|
| 1 | AutoHybridForecaster | randomstatsmodels | 8.33 | 7 | 31 | 71 | 23.94 | 27.46 | 8.79% | 8.92% |
| 2 | SF_AutoARIMA | statsforecast | 8.33 | 19 | 37 | 84 | 14.01 | 16.67 | 6.56% | 6.43% |
| 3 | AutoPolymath | randomstatsmodels | 9.07 | 8 | 32 | 64 | 10.75 | 13.41 | 10.16% | 9.54% |
| 4 | SF_AutoTBATS | statsforecast | 9.40 | 4 | 26 | 67 | 14.73 | 17.59 | 10.95% | 10.71% |
| 5 | AutoKoopman | randomstatsmodels | 10.15 | 9 | 19 | 47 | 23.06 | 28.66 | 12.69% | 13.00% |
| 6 | SF_AutoCES | statsforecast | 10.50 | 17 | 36 | 59 | 13.01 | 16.05 | 10.25% | 10.27% |
| 7 | AutoNEO | randomstatsmodels | 10.85 | 5 | 28 | 62 | 26.78 | 36.34 | 12.67% | 13.57% |
| 8 | AutoDynamic | ensemble | 10.93 | 2 | 15 | 44 | 24.24 | 29.12 | 14.34% | 13.75% |
| 9 | SF_AutoETS | statsforecast | 11.04 | 4 | 13 | 47 | 15.43 | 18.24 | 11.30% | 11.46% |
| 10 | AutoSSA | randomstatsmodels | 11.82 | 8 | 17 | 54 | 15.72 | 19.07 | 10.60% | 10.80% |
| 11 | AutoNaive | randomstatsmodels | 12.44 | 7 | 12 | 35 | 13.27 | 16.32 | 12.46% | 12.56% |
| 12 | SF_AutoTheta | statsforecast | 12.49 | 5 | 15 | 41 | 16.50 | 20.94 | 12.33% | 12.23% |
| 13 | AutoLocalLinear | randomstatsmodels | 13.20 | 7 | 15 | 48 | 15.34 | 20.87 | 13.64% | 13.30% |
| 14 | AutoKNN | randomstatsmodels | 13.37 | 8 | 35 | 48 | 10.65 | 12.91 | 11.58% | 11.74% |
| 15 | AutoFourier | randomstatsmodels | 14.09 | 8 | 17 | 33 | 32.74 | 40.36 | 19.44% | 18.93% |
| 16 | AutoGreensKernel | randomstatsmodels | 14.15 | 8 | 16 | 43 | 21.18 | 25.83 | 16.47% | 17.46% |
| 17 | AutoPALF | randomstatsmodels | 14.90 | 6 | 8 | 16 | 24.32 | 28.59 | 15.04% | 13.89% |
| 18 | AutoMELD | randomstatsmodels | 15.24 | 3 | 12 | 32 | 6.30 | 8.20 | 13.70% | 13.88% |
| 19 | AutoThetaAR | randomstatsmodels | 15.30 | 0 | 4 | 22 | 24.03 | 29.14 | 22.09% | 24.03% |
| 20 | AutoBagged | ensemble | 16.21 | 2 | 16 | 29 | 29.73 | 34.69 | 23.87% | 24.07% |
| 21 | AutoPDEField | randomstatsmodels | 16.28 | 1 | 9 | 16 | 27.90 | 33.07 | 23.77% | 23.29% |
| 22 | AutoSpectralGradient | randomstatsmodels | 16.72 | 0 | 3 | 12 | 53.35 | 61.39 | 15.31% | 16.28% |
| 23 | AutoHoltWinters | randomstatsmodels | 17.50 | 3 | 7 | 11 | 30.23 | 36.09 | 20.24% | 23.47% |
| 24 | AutoVariationalPath | randomstatsmodels | 18.13 | 1 | 4 | 8 | 23.45 | 30.16 | 20.14% | 18.16% |
| 25 | AutoRIFT | randomstatsmodels | 19.60 | 0 | 0 | 8 | 27.19 | 35.14 | 27.37% | 29.67% |
| 26 | AutoFracDiff | randomstatsmodels | 23.35 | 2 | 5 | 7 | 214.44 | 217.35 | 61.11% | 88.03% |
| 27 | AutoStacked | ensemble | 24.76 | 0 | 0 | 0 | 118.98 | 131.70 | 168.43% | 95.01% |

### Dataset Coverage

36 real-world datasets across 11 challenge categories:

| Category | Datasets |
|----------|----------|
| Trend + Seasonality | AirPassengers, MilkProduction, JohnsonJohnson, AusBeer, CO2, WineSales |
| Pure Seasonality | Nottem, USAccDeaths, UKGas, MelbourneTemp |
| Trend-Dominant | Shampoo, USGDPGrowth, WorldPopulation |
| Cyclical | Sunspots, Lynx, SOI |
| Level Shift | Nile, UKDriverDeaths, LakeHuron |
| Volatile / Financial | GoldPrice, USIndProduction |
| Short Series | TornadoDeaths, WheatYield, Discoveries, USStrikes |
| Long Memory | NileMinLevel, GlobalTemp |
| Count / Intermittent | VolcanicEruptions, IntlAirline, LondonRain |
| Nonlinear | SingaporeHumidity, FedFundsRate, ChampagneSales |
| Additional | PigSlaughter, HousingStarts, WikiPageviews |

### Key Findings

- **AutoHybridForecaster** ties for #1 (8.33) — linear + GRU residuals, best randomstatsmodels model
- **SF_AutoARIMA** ties for #1 (8.33) — most #1st finishes (19) and #Top3 (37)
- **AutoPolymath** is the best fast single model (#3) — lowest median MAE (10.75) and RMSE (13.41)
- **AutoKoopman** (#5) — Koopman/DMD, most #1st finishes (9) among randomstatsmodels
- **AutoDynamic** (#8) — horizon-adaptive ensemble using all 18 randomstatsmodels base models
- **No single model dominates** — model selection matters for your data type

### Benchmarking Your Own Data

```python
from randomstatsmodels.benchmarking.datasets import load_datasets
from randomstatsmodels.benchmarking.evaluation import evaluate_all, print_summary
from randomstatsmodels import AutoKoopman, AutoPolymath, AutoSSA

datasets = load_datasets()
results = evaluate_all([AutoKoopman, AutoPolymath, AutoSSA], datasets)
print_summary(results["summary"])
```

---

## Metrics

```python
from randomstatsmodels.metrics import mae, mse, rmse, mape, smape
```

The evaluation framework also provides MASE, MSSE, and Median Absolute Error.

---

## License

MIT