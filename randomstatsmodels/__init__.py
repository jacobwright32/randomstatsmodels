__version__ = "1.1.2"

from .benchmarking.benchmarking import benchmark_model, benchmark_models
from .metrics import mae, mse, rmse, mape, smape
from .models import (
    HybridForecastNet,
    AutoHybridForecaster,
    MELDForecaster,
    AutoMELD,
    KNNForecaster,
    AutoKNN,
    PALF,
    AutoPALF,
    NEOForecaster,
    AutoNEO,
    AutoThetaAR,
    PolymathForecaster,
    AutoPolymath,
    FourierForecaster,
    AutoFourier,
)

__all__ = [
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "HybridForecastNet",
    "AutoHybridForecaster",
    "MELDForecaster",
    "AutoMELD",
    "KNNForecaster",
    "AutoKNN",
    "PALF",
    "AutoPALF",
    "NEOForecaster",
    "AutoNEO",
    "AutoThetaAR",
    "PolymathForecaster",
    "AutoPolymath",
    "FourierForecaster",
    "AutoFourier",
    "benchmark_model",
    "benchmark_models",
]
