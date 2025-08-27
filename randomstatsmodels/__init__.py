# randomstatsmodels/__init__.py
# This makes the folder a Python package.

from .metrics.metrics import mae, mape, smape, rmse
from .models.models import (
    AutoHybridForecaster,
    AutoKNN,
    AutoMELD,
    AutoNEO,
    AutoPALF,
    AutoThetaAR,
    AutoPolymath,
    AutoSeasonalAR,
    AutoFourier,
    AutoRollingMedian,
    AutoTrimmedMean,
    AutoWindow,
    AutoRankInsertion,
)
from .benchmarking.benchmarking import benchmark_model, benchmark_models

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "mae",
    "mape",
    "smape",
    "rmse",
    "AutoHybridForecaster",
    "AutoKNN",
    "AutoMELD",
    "AutoNEO",
    "AutoPALF",
    "AutoThetaAR",
    "AutoPolymath",
    "AutoSeasonalAR",
    "benchmark_models",
    "benchmark_model",
]
