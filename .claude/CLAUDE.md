# randomstatsmodels

Lightweight Python package for univariate time-series forecasting with 19 auto-tuned models — NumPy only.

## Project overview

- **Package**: `randomstatsmodels` (PyPI), currently v2.0.0
- **Author**: Jacob Wright
- **License**: MIT
- **Python**: 3.9+
- **Only runtime dependency**: NumPy >= 1.24

## Package structure

```
randomstatsmodels/
  __init__.py              # Public API — all Auto* classes and metrics
  metrics/
    metrics.py             # mae, mse, rmse, mape, smape
  models/
    neo.py                 # NEOForecaster / AutoNEO
    fourier.py             # FourierForecaster / AutoFourier
    knn.py                 # KNNForecaster / AutoKNN
    polymath.py            # PolymathForecaster / AutoPolymath
    theta_ar.py            # AutoThetaAR
    hybrid.py              # HybridForecastNet / AutoHybridForecaster
    meld.py                # MELDForecaster / AutoMELD
    palf.py                # PALF / AutoPALF
    naive.py               # NaiveForecaster / AutoNaive
    holt_winters.py        # HoltWintersForecaster / AutoHoltWinters
    ssa.py                 # SSAForecaster / AutoSSA
    local_linear.py        # LocalLinearForecaster / AutoLocalLinear
    ensemble.py            # EnsembleForecaster / AutoEnsemble
    rift.py                # RIFTForecaster / AutoRIFT
    fracdiff.py            # FracDiffForecaster / AutoFracDiff
    spectral_gradient.py   # SpectralGradientForecaster / AutoSpectralGradient
    greens_kernel.py       # GreensKernelForecaster / AutoGreensKernel
    pde_field.py           # PDEFieldForecaster / AutoPDEField
    variational_path.py    # VariationalPathForecaster / AutoVariationalPath
    koopman.py             # KoopmanForecaster / AutoKoopman
    model_utils.py         # Shared model utilities
  benchmarking/
    benchmarking.py        # benchmark_model(), benchmark_models()
    datasets.py            # 36 real-world time series datasets
    evaluation.py          # Multi-model multi-dataset evaluation framework
tests/
  test_metrics.py
  test_models.py
  test_new_models.py
```

## Model API convention

Every model follows a consistent pattern:
- **Base class** (e.g. `NEOForecaster`): `.fit(y)` and `.predict(h)`
- **Auto wrapper** (e.g. `AutoNEO`): accepts parameter grids, performs grid search on validation split, stores `.best_` dict with `"config"` and `"val_score"`, refits on full data

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Build distribution
python -m build

# Publish to PyPI
twine upload dist/*
```

## Development guidelines

- All models are NumPy-only — do not add scipy/sklearn/pandas as runtime dependencies
- Every new model needs both a base forecaster class and an Auto* tuner wrapper
- New models must be exported from `models/__init__.py` and `randomstatsmodels/__init__.py`
- New models should have test coverage in `tests/test_new_models.py`
- Follow Keep a Changelog format in CHANGELOG.md
- Use semantic versioning
- Benchmark new models on the 36-dataset suite before releasing
