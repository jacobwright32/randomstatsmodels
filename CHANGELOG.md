# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.0.0] - 2026-03-25
### Added
- **Speed presets** on every Auto* model: `speed="super_fast"`, `"fast"`, `"normal"`, `"slow"`, `"super_slow"`
  - Controls grid search size (from ~2 combos to 2000+)
  - Explicit kwargs always override the speed preset
- `create_model(name, speed)` factory function and `get_grid()` API
- `presets.py` module with 5-level grid definitions for all 22 models
- 10 real-world FRED datasets (300-600 pts) downloaded at runtime
- Separate benchmark scripts: `benchmark_fast_models.py` (<30s models) and `benchmark_slow_models.py` (>30s models)
- Incremental CSV saving during benchmarks (results not lost on crash)

### Changed
- Ensemble base pool now uses top 5 randomstatsmodels models only
- Ensembles removed from benchmark rankings (they wrap other models)
- AusBeer dataset replaced with CementProduction (data integrity fix)
- `train_test_split` now raises on too-short series
- Datasets reduced from 36 to 20 (10 long FRED + 10 classic hardcoded)
- Benchmark rankings use RMSE, counts per-dataset (max 20)

### Removed
- Removed synthetic data extension (`_extend_series`)
- Removed old benchmark scripts

## [3.1.0] - 2026-03-25
### Added
- **statsforecast comparison benchmarks**: SF_AutoARIMA, SF_AutoETS, SF_AutoCES, SF_AutoTheta, SF_AutoTBATS wrappers
- statsforecast wrappers module (`benchmarking/statsforecast_wrappers.py`)
- Full 27-model benchmark script (`examples/benchmark_all_v31.py`)
- `model_type` column in benchmark results (randomstatsmodels, statsforecast, ensemble)
- statsforecast as optional dependency

### Changed
- README rankings now show all 27 models with model_type column
- SF_AutoARIMA ranks #1 overall (avg rank 8.36), AutoHybridForecaster #2 (8.44)

## [3.0.0] - 2026-03-24
### Added
- **3 advanced ensemble forecasters**:
  - **AutoStacked**: Meta-learner stacking — base models' predictions become features for a ridge meta-learner
  - **AutoBagged**: Block-bootstrap bagging — fits same model on resampled series, combines via median
  - **AutoDynamic**: Horizon-adaptive ensemble — model weights vary per forecast step based on horizon-specific accuracy
- AutoDynamic ranks **#1 overall** across 36 datasets with median sMAPE of 12.01%
- Tests for all 3 new ensemble models

### Changed
- Total model count: 19 → 22 Auto* forecasters
- Updated README with full 22-model ranking table

## [2.0.0] - 2026-03-24
### Added
- **6 new forecasting models** based on advanced calculus and dynamical systems:
  - **AutoKoopman**: Dynamic Mode Decomposition / Koopman operator via delay embedding — ranks #4 overall
  - **AutoFracDiff**: Fractional calculus forecaster using Grunwald-Letnikov fractional derivatives
  - **AutoSpectralGradient**: Spectral derivative flow — extrapolates Fourier mode dynamics
  - **AutoGreensKernel**: Integral equation forecaster via Green's function convolution kernels
  - **AutoPDEField**: PDE-based forecaster — advection-diffusion on a time-scale field
  - **AutoVariationalPath**: Calculus of variations — Euler-Lagrange optimal forecast path
- **36 real-world benchmark datasets** (`benchmarking/datasets.py`) across 11 challenge categories
- **Comprehensive evaluation framework** (`benchmarking/evaluation.py`) with 7 metrics (MAE, RMSE, MAPE, sMAPE, MASE, MSSE, Median AE), per-dataset rankings, model summary tables, and CSV export
- Benchmark runner script (`examples/comprehensive_benchmark.py`)
- Test suite for all 6 new models and the benchmark infrastructure (64 new tests)

### Changed
- Total model count: 13 → 19 Auto* forecasters
- Benchmark scope: 12 → 36 real-world datasets
- Summary statistics now use **median** instead of mean for raw metric aggregation
- Updated README with full 19-model ranking table and dataset coverage

## [1.6.1] - 2026-02-01
### Added
- Unit test suite with pytest (tests/test_models.py, tests/test_metrics.py)
- Test coverage for all 13 Auto* tuners and base forecasters
- Test coverage for all 5 error metrics (MAE, MSE, RMSE, MAPE, SMAPE)
- pytest as optional dev dependency (`pip install randomstatsmodels[dev]`)

### Fixed
- Removed dead code in models_old.py (1,135 lines of legacy models)
- Fixed FourierForecaster.predict() docstring (was incorrectly copied from fit())

### Changed
- Cleaned up models/__init__.py by removing legacy model imports

## [1.6.0] - 2026-02-01
### Changed
- Replaced synthetic datasets with 10 additional real-world datasets
- New real datasets: US GDP Growth, US Unemployment, Gold Prices, Electricity Production, Wine Sales, Lynx Trappings, Lake Erie Level, US Retail Sales, Australia Passengers, Accidental Deaths
- Updated model rankings based on real data performance
- AutoLocalLinear now leads with avg rank 5.2 and 4 first-place finishes
- AutoRIFT shows strong performance on financial data (1st on Gold Prices)

## [1.5.0] - 2026-02-01
### Added
- Expanded benchmarks to 12 diverse datasets (up from 2)
- Overall model rankings table with average rank, #1 finishes, and top-3 finishes
- Key findings section summarizing model strengths by data type

## [1.4.0] - 2026-02-01
### Added
- Initial benchmark results in README comparing all 13 models
- Benchmarks on Airline Passengers dataset (trend + seasonality)
- Benchmarks on Sunspots dataset (cyclical, stationary)

## [1.3.0] - 2026-02-01
### Added
- **AutoRIFT**: Novel "Recursive Information Flow Tensor" forecaster based on original "Predictive Information Field Dynamics" theory
  - Models how predictive information flows between temporal channels (level, trend, curvature, oscillations)
  - Learns Information Flow Matrix to capture channel dynamics across forecast horizons
  - Uses Fisher Information estimation for channel weighting
- **AutoNaive**: Essential baseline forecasters (last, seasonal, drift, mean methods)
- **AutoHoltWinters**: Classic Holt-Winters exponential smoothing with level, trend, and seasonal components
- **AutoSSA**: Singular Spectrum Analysis via SVD decomposition for adaptive oscillatory modes
- **AutoLocalLinear**: Weighted local regression with exponential decay for older observations
- **AutoEnsemble**: Model stacking with learned weights (uniform, validation, optimal weighting)

## [1.2.0] - 2026-02-01
### Added
- Initial implementation of 5 new forecasting models (released alongside 1.3.0)

## [0.1.0] - 2025-08-26
### Added
- Initial release of `randomstatsmodels`.
- Core error metrics: MAE, RMSE, MAPE, SMAPE.
- Forecasting models: HybridForecastNet, MELDForecaster, KNNForecaster, PALF, NEOForecaster, AutoThetaAR, PolymathForecaster.
- Auto hyperparameter search classes for models.
- Utility functions in `model_utils`.
- Basic package structure and PyPI publishing.

## [1.0.0] - 2025-08-28
### Added
- Move forecasting models into seperate files.
- Add example within README.

## [1.0.1] - 2025-08-28
### Fix
- AutoNeo Prediction

## [1.0.2] - 2025-08-28
### Added
- update readme to include medium article

## [1.0.3] - 2025-08-28
### Added
- update readme


## [1.0.3] - 2025-08-29
### Added
- adds new models to readme
- adds docstrings for all models

## [1.1.1] - 2025-08-29
### Fix
- fixes bug with fourier.py imports

## [1.1.2] - 2025-08-29
### Fix
- fixes bug with hybrid.py imports
