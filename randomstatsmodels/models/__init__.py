from .fourier import FourierForecaster, AutoFourier
from .hybrid import HybridForecastNet, AutoHybridForecaster
from .meld import MELDForecaster, AutoMELD
from .knn import KNNForecaster, AutoKNN
from .palf import PALF, AutoPALF
from .neo import NEOForecaster, AutoNEO
from .theta_ar import AutoThetaAR
from .polymath import PolymathForecaster, AutoPolymath
from .naive import NaiveForecaster, AutoNaive
from .holt_winters import HoltWintersForecaster, AutoHoltWinters
from .ssa import SSAForecaster, AutoSSA
from .local_linear import LocalLinearForecaster, AutoLocalLinear
from .ensemble import EnsembleForecaster, AutoEnsemble
from .rift import RIFTForecaster, AutoRIFT

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
    "NaiveForecaster",
    "AutoNaive",
    "HoltWintersForecaster",
    "AutoHoltWinters",
    "SSAForecaster",
    "AutoSSA",
    "LocalLinearForecaster",
    "AutoLocalLinear",
    "EnsembleForecaster",
    "AutoEnsemble",
    "RIFTForecaster",
    "AutoRIFT",
]
