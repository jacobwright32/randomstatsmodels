# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.0] - 2026-02-01
### Added
- Comprehensive benchmark results in README comparing all 13 models
- Benchmarks on Airline Passengers dataset (trend + seasonality)
- Benchmarks on Sunspots dataset (cyclical, stationary)
- Key findings: AutoRIFT excels on cyclical data (2nd place), AutoLocalLinear leads on trending data

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
