# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
