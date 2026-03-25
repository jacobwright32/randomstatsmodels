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

| Rank | Model | Type | Avg Rank | #1st | #Top3 | Median sMAPE |
|------|-------|------|----------|------|-------|-------------|
| 1 | SF_AutoARIMA | statsforecast | 8.25 | 4 | 9 | 6.43% |
| 2 | AutoHybridForecaster | randomstatsmodels | 8.28 | 1 | 8 | 8.92% |
| 3 | AutoPolymath | randomstatsmodels | 9.08 | 2 | 9 | 9.54% |
| 4 | SF_AutoTBATS | statsforecast | 9.39 | 1 | 7 | 10.71% |
| 5 | AutoNEO | randomstatsmodels | 9.68 | 1 | 8 | 13.57% |
| 6 | AutoKoopman | randomstatsmodels | 10.11 | 3 | 5 | 13.00% |
| 7 | SF_AutoCES | statsforecast | 10.47 | 4 | 9 | 10.27% |
| 8 | AutoDynamic | ensemble | 10.97 | 0 | 4 | 13.75% |
| 9 | SF_AutoETS | statsforecast | 11.50 | 1 | 2 | 11.46% |
| 10 | AutoSSA | randomstatsmodels | 11.78 | 2 | 4 | 10.80% |
| 11 | SF_AutoTheta | statsforecast | 12.64 | 1 | 4 | 12.23% |
| 12 | AutoNaive | randomstatsmodels | 12.67 | 2 | 3 | 12.56% |
| 13 | AutoKNN | randomstatsmodels | 12.68 | 2 | 9 | 11.74% |
| 14 | AutoLocalLinear | randomstatsmodels | 13.25 | 2 | 3 | 13.30% |
| 15 | AutoFourier | randomstatsmodels | 14.03 | 2 | 4 | 18.93% |
| 16 | AutoGreensKernel | randomstatsmodels | 14.11 | 3 | 4 | 17.46% |
| 17 | AutoMELD | randomstatsmodels | 14.18 | 1 | 3 | 13.88% |
| 18 | AutoPALF | randomstatsmodels | 14.97 | 2 | 2 | 13.89% |
| 19 | AutoThetaAR | randomstatsmodels | 15.14 | 0 | 1 | 24.03% |
| 20 | AutoBagged | ensemble | 16.08 | 1 | 4 | 24.07% |
| 21 | AutoPDEField | randomstatsmodels | 16.50 | 0 | 2 | 23.29% |
| 22 | AutoSpectralGradient | randomstatsmodels | 16.89 | 0 | 1 | 16.28% |
| 23 | AutoHoltWinters | randomstatsmodels | 17.64 | 1 | 2 | 23.47% |
| 24 | AutoVariationalPath | randomstatsmodels | 17.92 | 0 | 1 | 18.16% |
| 25 | AutoRIFT | randomstatsmodels | 18.79 | 0 | 0 | 29.67% |
| 26 | AutoFracDiff | randomstatsmodels | 23.36 | 0 | 0 | 88.03% |
| 27 | AutoStacked | ensemble | 24.89 | 0 | 0 | 95.01% |

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

- **SF_AutoARIMA** (statsforecast) edges out as #1 overall — the industry gold standard
- **AutoHybridForecaster** is virtually tied at #2 (8.28 vs 8.25) — linear + GRU residuals
- **AutoPolymath** is the best fast single model (#3) — polynomial + Fourier + ridge
- **AutoKoopman** (#6) — Koopman/DMD eigenvalue propagation, extremely fast
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