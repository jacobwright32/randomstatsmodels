from .fourier import FourierForecaster, AutoFourier
from .hybrid import HybridForecastNet, AutoHybridForecaster
from .meld import MELDForecaster, AutoMELD
from .knn import KNNForecaster, AutoKNN
from .palf import PALF, AutoPALF
from .neo import NEOForecaster, AutoNEO
from .theta_ar import AutoThetaAR
from .polymath import PolymathForecaster, AutoPolymath

# Legacy models (NEED IMPROVEMENT)
from .models_old import (
    AutoSeasonalAR,
    AutoRollingMedian,
    AutoTrimmedMean,
    AutoWindow,
    AutoRankInsertion,
)

__all__ = [
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
]
