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

---

## Benchmarks

All 19 models evaluated on **36 real-world time series** (20% holdout), ranked by MAE.

### Overall Rankings

| Rank | Model | Avg Rank | #1st | #Top3 | Mdn sMAPE |
|------|-------|----------|------|-------|-----------|
| 1 | AutoHybridForecaster | 5.56 | 3 | 12 | 8.92% |
| 2 | AutoPolymath | 6.14 | 5 | 12 | 9.54% |
| 3 | AutoNEO | 6.56 | 2 | 12 | 13.57% |
| 4 | **AutoKoopman** | **6.81** | **4** | **9** | **13.00%** |
| 5 | AutoSSA | 8.03 | 3 | 11 | 10.80% |
| 6 | AutoKNN | 8.68 | 4 | 9 | 11.74% |
| 7 | AutoNaive | 8.75 | 2 | 6 | 12.56% |
| 8 | AutoLocalLinear | 8.94 | 2 | 6 | 13.30% |
| 9 | AutoFourier | 9.67 | 2 | 6 | 18.93% |
| 10 | AutoMELD | 9.85 | 0 | 6 | 12.79% |
| 11 | AutoGreensKernel | 9.89 | 4 | 4 | 17.46% |
| 12 | AutoPALF | 10.42 | 2 | 2 | 13.89% |
| 13 | AutoThetaAR | 10.56 | 1 | 4 | 24.03% |
| 14 | AutoSpectralGradient | 11.50 | 0 | 1 | 16.28% |
| 15 | AutoPDEField | 11.58 | 1 | 3 | 23.29% |
| 16 | AutoHoltWinters | 12.31 | 1 | 2 | 23.47% |
| 17 | AutoVariationalPath | 12.78 | 0 | 1 | 18.16% |
| 18 | AutoRIFT | 13.29 | 0 | 1 | 29.67% |
| 19 | AutoFracDiff | 16.78 | 0 | 1 | 88.03% |

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

- **AutoHybridForecaster** leads overall — linear decomposition + GRU residuals, but slowest to fit
- **AutoPolymath** is the best fast model — polynomial + Fourier features with ridge regression
- **AutoKoopman** (new) is #4 overall — Koopman/DMD eigenvalue propagation, extremely fast
- **AutoGreensKernel** (new) gets the most 1st-place wins (4) among calculus-based models
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