"""
Model speed presets — 5 levels from super_fast to super_slow.

Every Auto* model accepts a ``speed`` parameter::

    from randomstatsmodels import AutoKoopman

    model = AutoKoopman(speed="slow")      # large grid
    model = AutoKoopman(speed="super_fast") # tiny grid, seconds
    model = AutoKoopman()                   # normal (default)

Explicit kwargs always override the speed preset::

    model = AutoKoopman(speed="slow", embed_grid=(4, 8))  # uses (4,8)

Levels (grid size multiplier is approximate):

    super_fast  ~2-4 combos     seconds
    fast        ~4-12 combos    under a minute
    normal      default grids   minutes
    slow        ~50-200 combos  several minutes
    super_slow  ~200-2000+      long, thorough
"""

import functools

SPEEDS = ("super_fast", "fast", "normal", "slow", "super_slow")

# -----------------------------------------------------------------------
# Grid definitions:  model_name -> {speed -> kwargs}
# "normal" always returns {} so the class uses its own defaults.
# -----------------------------------------------------------------------

_GRIDS = {
    "AutoNaive": {
        "super_fast": dict(method_options=("last",), seasonal_periods=(1,)),
        "fast":       dict(method_options=("last", "seasonal"), seasonal_periods=(1, 12)),
        "normal":     {},
        "slow":       dict(method_options=("last", "seasonal", "drift", "mean"),
                           seasonal_periods=(1, 4, 7, 12, 24, 52)),
        "super_slow": dict(method_options=("last", "seasonal", "drift", "mean"),
                           seasonal_periods=(1, 2, 4, 6, 7, 12, 24, 36, 52)),
    },
    "AutoHoltWinters": {
        "super_fast": dict(seasonal_periods=(12,), trend_options=("add",), seasonal_options=("add",)),
        "fast":       dict(seasonal_periods=(12,), trend_options=("add", "none"), seasonal_options=("add",)),
        "normal":     {},
        "slow":       dict(seasonal_periods=(4, 6, 12, 24), trend_options=("add", "none", "damped"),
                           seasonal_options=("add", "none")),
        "super_slow": dict(seasonal_periods=(3, 4, 6, 7, 12, 24, 52),
                           trend_options=("add", "none", "damped"),
                           seasonal_options=("add", "none")),
    },
    "AutoFourier": {
        "super_fast": dict(n_harmonics_grid=(0, 3), trend_options=("linear",)),
        "fast":       dict(n_harmonics_grid=(0, 2, 5), trend_options=("linear",)),
        "normal":     {},
        "slow":       dict(n_harmonics_grid=(0, 1, 2, 3, 5, 8, 12), trend_options=("none", "linear")),
        "super_slow": dict(n_harmonics_grid=(0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16),
                           trend_options=("none", "linear")),
    },
    "AutoNEO": {
        "super_fast": dict(lags_grid=(4,), degree_grid=(1,), window_fracs=(None,)),
        "fast":       dict(lags_grid=(4, 8), degree_grid=(1, 2), window_fracs=(None,)),
        "normal":     {},
        "slow":       dict(lags_grid=(3, 4, 6, 8, 12, 16, 24), degree_grid=(1, 2, 3),
                           window_fracs=(None, 0.5, 0.75, 0.9)),
        "super_slow": dict(lags_grid=(2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32),
                           degree_grid=(1, 2, 3),
                           window_fracs=(None, 0.3, 0.5, 0.6, 0.75, 0.9)),
    },
    "AutoKNN": {
        "super_fast": dict(window_grid=(8,), k_grid=(3,)),
        "fast":       dict(window_grid=(8, 12), k_grid=(3, 5)),
        "normal":     {},
        "slow":       dict(window_grid=(4, 8, 12, 16, 24), k_grid=(1, 3, 5, 7, 10)),
        "super_slow": dict(window_grid=(3, 4, 6, 8, 10, 12, 16, 20, 24, 32),
                           k_grid=(1, 2, 3, 5, 7, 10, 15, 20)),
    },
    "AutoPolymath": {
        "super_fast": dict(lags_grid=(6,), degree_grid=(1,), fourier_terms_grid=(0,),
                           window_fracs=(None,)),
        "fast":       dict(lags_grid=(6, 12), degree_grid=(1, 2), fourier_terms_grid=(0, 2),
                           window_fracs=(None,)),
        "normal":     {},
        "slow":       dict(lags_grid=(4, 6, 8, 12, 16, 24), degree_grid=(1, 2, 3),
                           fourier_terms_grid=(0, 2, 4, 6), window_fracs=(None, 0.5, 0.75)),
        "super_slow": dict(lags_grid=(3, 4, 6, 8, 10, 12, 16, 20, 24, 32),
                           degree_grid=(1, 2, 3),
                           fourier_terms_grid=(0, 1, 2, 3, 4, 6, 8),
                           window_fracs=(None, 0.3, 0.5, 0.6, 0.75, 0.9)),
    },
    "AutoThetaAR": {
        "super_fast": {},
        "fast":       {},
        "normal":     {},
        "slow":       {},
        "super_slow": {},
    },
    "AutoSSA": {
        "super_fast": dict(window_fracs=(0.33,), n_components_grid=(None,)),
        "fast":       dict(window_fracs=(0.25, 0.5), n_components_grid=(None, 4)),
        "normal":     {},
        "slow":       dict(window_fracs=(0.2, 0.25, 0.33, 0.5, 0.67),
                           n_components_grid=(None, 2, 4, 6, 8, 12)),
        "super_slow": dict(window_fracs=(0.15, 0.2, 0.25, 0.3, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75),
                           n_components_grid=(None, 1, 2, 3, 4, 6, 8, 10, 12, 16)),
    },
    "AutoLocalLinear": {
        "super_fast": dict(decay_grid=(0.95,), degree_grid=(1,)),
        "fast":       dict(decay_grid=(0.95, 1.0), degree_grid=(1,)),
        "normal":     {},
        "slow":       dict(decay_grid=(0.85, 0.9, 0.95, 0.98, 1.0), degree_grid=(1, 2, 3)),
        "super_slow": dict(decay_grid=(0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0),
                           degree_grid=(1, 2, 3)),
    },
    "AutoPALF": {
        "super_fast": dict(p_candidates=(4,), penalties=("l2",)),
        "fast":       dict(p_candidates=(4, 8), penalties=("l2",)),
        "normal":     {},
        "slow":       dict(p_candidates=(2, 4, 6, 8, 12, 16), penalties=("l1", "l2", "huber")),
        "super_slow": dict(p_candidates=(2, 3, 4, 6, 8, 10, 12, 16, 24),
                           penalties=("l1", "l2", "huber", "pinball")),
    },
    "AutoMELD": {
        "super_fast": dict(lags_grid=(8,), scales_grid=((1, 3, 7),), rff_features_grid=(64,)),
        "fast":       dict(lags_grid=(8, 12), scales_grid=((1, 3, 7),), rff_features_grid=(64,)),
        "normal":     {},
        "slow":       dict(lags_grid=(6, 8, 12, 16), scales_grid=((1, 3, 7), (1, 2, 4, 8)),
                           rff_features_grid=(64, 128, 256)),
        "super_slow": dict(lags_grid=(4, 6, 8, 10, 12, 16, 20),
                           scales_grid=((1, 3, 7), (1, 2, 4, 8), (1, 3, 5, 7, 14)),
                           rff_features_grid=(32, 64, 128, 256, 512)),
    },
    "AutoRIFT": {
        "super_fast": dict(n_frequencies_grid=(3,), embedding_dim_grid=(2,)),
        "fast":       dict(n_frequencies_grid=(2, 4), embedding_dim_grid=(2,)),
        "normal":     {},
        "slow":       dict(n_frequencies_grid=(2, 4, 6, 8), embedding_dim_grid=(2, 3, 4),
                           regularization_grid=(0.001, 0.01, 0.1)),
        "super_slow": dict(n_frequencies_grid=(1, 2, 3, 4, 6, 8, 10),
                           embedding_dim_grid=(2, 3, 4, 5),
                           regularization_grid=(0.0001, 0.001, 0.01, 0.1, 1.0)),
    },
    "AutoHybridForecaster": {
        "super_fast": dict(candidate_fourier=(3,), candidate_trend=(1,),
                           candidate_ar=(3,), candidate_hidden=(16,), epochs=30),
        "fast":       dict(candidate_fourier=(0, 3), candidate_trend=(1,),
                           candidate_ar=(3,), candidate_hidden=(16,), epochs=50),
        "normal":     {},
        "slow":       dict(candidate_fourier=(0, 3, 6, 9), candidate_trend=(0, 1, 2),
                           candidate_ar=(0, 3, 5, 8), candidate_hidden=(8, 16, 32, 64), epochs=200),
        "super_slow": dict(candidate_fourier=(0, 1, 3, 6, 9, 12), candidate_trend=(0, 1, 2),
                           candidate_ar=(0, 1, 3, 5, 8, 12),
                           candidate_hidden=(8, 16, 32, 64, 128), epochs=300),
    },
    "AutoFracDiff": {
        "super_fast": dict(d_grid=(0.4,), ar_grid=(5,), gl_window_grid=(32,)),
        "fast":       dict(d_grid=(0.3, 0.5), ar_grid=(3, 5), gl_window_grid=(32,)),
        "normal":     {},
        "slow":       dict(d_grid=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                           ar_grid=(2, 3, 5, 8, 12), gl_window_grid=(16, 32, 64, 128)),
        "super_slow": dict(d_grid=tuple(i / 20 for i in range(1, 20)),
                           ar_grid=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16),
                           gl_window_grid=(8, 16, 32, 48, 64, 96, 128)),
    },
    "AutoSpectralGradient": {
        "super_fast": dict(window_grid=(32,), hop_grid=(8,), damping_grid=(0.9,),
                           trend_weight_grid=(0.5,)),
        "fast":       dict(window_grid=(16, 32), hop_grid=(8,), damping_grid=(0.9,),
                           trend_weight_grid=(0.5,)),
        "normal":     {},
        "slow":       dict(window_grid=(8, 16, 24, 32, 48, 64), hop_grid=(4, 8, 16),
                           damping_grid=(0.7, 0.85, 0.95), trend_weight_grid=(0.2, 0.4, 0.6, 0.8)),
        "super_slow": dict(window_grid=(4, 8, 12, 16, 20, 24, 32, 48, 64),
                           hop_grid=(2, 4, 8, 12, 16),
                           damping_grid=(0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99),
                           trend_weight_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
    },
    "AutoGreensKernel": {
        "super_fast": dict(n_components_grid=(3,), max_lag_grid=(30,), kernel_window_grid=(0,)),
        "fast":       dict(n_components_grid=(2, 3), max_lag_grid=(20, 40), kernel_window_grid=(0,)),
        "normal":     {},
        "slow":       dict(n_components_grid=(2, 3, 5, 7), max_lag_grid=(15, 30, 50, 80),
                           kernel_window_grid=(0, 20, 40, 80)),
        "super_slow": dict(n_components_grid=(1, 2, 3, 4, 5, 6, 7, 9),
                           max_lag_grid=(10, 15, 20, 30, 40, 50, 60, 80, 100),
                           kernel_window_grid=(0, 10, 20, 30, 40, 60, 80, 100)),
    },
    "AutoPDEField": {
        "super_fast": dict(n_scales_grid=(5,), max_sigma_grid=(10.0,), dt_steps_grid=(2,)),
        "fast":       dict(n_scales_grid=(5, 8), max_sigma_grid=(10.0,), dt_steps_grid=(2,)),
        "normal":     {},
        "slow":       dict(n_scales_grid=(4, 6, 8, 12), max_sigma_grid=(5.0, 10.0, 20.0, 40.0),
                           dt_steps_grid=(1, 2, 4)),
        "super_slow": dict(n_scales_grid=(3, 4, 5, 6, 8, 10, 12, 16),
                           max_sigma_grid=(2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0),
                           dt_steps_grid=(1, 2, 3, 4, 6, 8)),
    },
    "AutoVariationalPath": {
        "super_fast": dict(smoothness_grid=(1.0,), jerk_grid=(0.1,), potential_grid=(0.5,),
                           seasonal_weight_grid=(1.0,), attractor_grid=("mean",)),
        "fast":       dict(smoothness_grid=(0.5, 1.0), jerk_grid=(0.1,), potential_grid=(0.5,),
                           seasonal_weight_grid=(1.0,), attractor_grid=("mean", "last")),
        "normal":     {},
        "slow":       dict(smoothness_grid=(0.1, 0.5, 1.0, 2.0, 5.0),
                           jerk_grid=(0.01, 0.1, 0.5, 1.0),
                           potential_grid=(0.1, 0.5, 1.0, 2.0),
                           seasonal_weight_grid=(0.5, 1.0, 2.0, 5.0),
                           attractor_grid=("mean", "last", "trend")),
        "super_slow": dict(smoothness_grid=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
                           jerk_grid=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0),
                           potential_grid=(0.01, 0.1, 0.5, 1.0, 2.0, 5.0),
                           seasonal_weight_grid=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
                           attractor_grid=("mean", "last", "trend")),
    },
    "AutoKoopman": {
        "super_fast": dict(embed_grid=(8,), rank_grid=(0,), damping_grid=(0.98,),
                           deseason_grid=(True,), detrend_grid=(True,)),
        "fast":       dict(embed_grid=(4, 8, 12), rank_grid=(0, 4), damping_grid=(0.98,),
                           deseason_grid=(True,), detrend_grid=(True,)),
        "normal":     {},
        "slow":       dict(embed_grid=(3, 4, 6, 8, 12, 16, 20, 24, 32),
                           rank_grid=(0, 2, 4, 6, 8, 12),
                           damping_grid=(0.9, 0.95, 0.98, 1.0),
                           deseason_grid=(True, False), detrend_grid=(True, False)),
        "super_slow": dict(embed_grid=(2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40),
                           rank_grid=(0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16),
                           damping_grid=(0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0),
                           deseason_grid=(True, False), detrend_grid=(True, False)),
    },
    "AutoEnsemble": {
        "super_fast": {},
        "fast":       {},
        "normal":     {},
        "slow":       {},
        "super_slow": {},
    },
    "AutoStacked": {
        "super_fast": dict(ridge_grid=(1.0,)),
        "fast":       dict(ridge_grid=(0.1, 1.0)),
        "normal":     {},
        "slow":       dict(ridge_grid=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)),
        "super_slow": dict(ridge_grid=(0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0)),
    },
    "AutoBagged": {
        "super_fast": dict(n_bags=3),
        "fast":       dict(n_bags=5),
        "normal":     {},
        "slow":       dict(n_bags=30, block_frac=0.08),
        "super_slow": dict(n_bags=50, block_frac=0.06),
    },
    "AutoDynamic": {
        "super_fast": dict(smoothing=0.3),
        "fast":       dict(smoothing=0.3),
        "normal":     {},
        "slow":       dict(smoothing=0.3),
        "super_slow": dict(smoothing=0.3),
    },
}


# -----------------------------------------------------------------------
# Decorator: adds `speed` parameter to any Auto* class
# -----------------------------------------------------------------------

def _add_speed(cls):
    """Decorator that adds a ``speed`` parameter to an Auto* class.

    When speed != "normal", the preset grid overrides are merged into
    the kwargs *before* the original __init__ runs.  Explicit kwargs
    always win (so users can still override individual grid params).
    """
    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, speed="normal", **kwargs):
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}, got '{speed}'")
        overrides = _GRIDS.get(cls.__name__, {}).get(speed, {})
        # Preset fills in missing kwargs; explicit kwargs take precedence
        merged = {**overrides, **kwargs}
        self._speed = speed
        original_init(self, *args, **merged)

    cls.__init__ = new_init
    return cls


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def get_grid(name: str, speed: str = "normal") -> dict:
    """Return the kwargs dict for a model at a given speed."""
    return _GRIDS.get(name, {}).get(speed, {})


def list_models():
    """Return sorted list of model names with speed support."""
    return sorted(_GRIDS.keys())


def create_model(name: str, speed: str = "normal"):
    """Create a model instance at the given speed level."""
    from . import models as _m
    cls = getattr(_m, name, None)
    if cls is None:
        raise ValueError(f"Unknown model: {name}")
    return cls(speed=speed)


def get_model_class(name: str):
    """Return the (speed-aware) model class by name."""
    from . import models as _m
    cls = getattr(_m, name, None)
    if cls is None:
        raise ValueError(f"Unknown model: {name}")
    return cls
