"""
Koopman Mode Forecaster via Dynamic Mode Decomposition
========================================================

Uses delay embedding to lift a scalar time series into a higher-dimensional
state space, then discovers the **best-fit linear dynamics operator** via
Dynamic Mode Decomposition (DMD).  Forecasts are produced by propagating
the Koopman eigenvalues forward — giving **exact multi-step prediction**
without iterative error accumulation.

Mathematical foundation
-----------------------
Given a time series y_t, construct delay-embedded state vectors:

    x_t = [y_t, y_{t-1}, …, y_{t-p+1}]^T

Arrange consecutive states into matrices:

    X = [x_1, x_2, …, x_{N-1}]       (p × M)
    Y = [x_2, x_3, …, x_N]           (p × M)

The DMD finds the best-fit linear operator A such that Y ≈ A · X:

    1. SVD:  X = U Σ V*
    2. Project:  Ã = U_r^T Y V_r Σ_r^{-1}   (r × r reduced dynamics)
    3. Eigendecomposition:  Ã W = W Λ
    4. DMD modes:  Φ = Y V_r Σ_r^{-1} W

Each eigenvalue λ_k = |λ_k| · e^{iθ_k} simultaneously encodes:
    - Growth/decay rate:  |λ_k|
    - Oscillation frequency:  θ_k / (2π)

Forecasting (exact, non-iterative):

    x_{N+h} = Φ · Λ^h · b     where b = Φ^+ · x_N

This is the data-driven Koopman operator — cutting-edge applied math
used in fluid dynamics, neuroscience, and control theory, here adapted
for univariate time series forecasting.

Design philosophy
-----------------
The model preprocesses the series in three stages (like the #1-ranked
AutoHybridForecaster's separation-of-concerns approach):

    1. Detect and remove seasonal component (period-averaging)
    2. Remove linear trend
    3. Apply DMD to the stationary residual

Forecasting reverses the pipeline: DMD forecast + trend + season.
"""

from typing import Optional, Dict, Iterable
import numpy as np
from ..metrics import mae, rmse


# -----------------------------------------------------------------------
# Preprocessing utilities
# -----------------------------------------------------------------------

def _detect_period(y: np.ndarray, max_period: int = 60) -> int:
    """Detect dominant period via autocorrelation peak."""
    n = len(y)
    if n < 10:
        return 1
    y_c = y - np.mean(y)
    var = np.dot(y_c, y_c)
    if var < 1e-12:
        return 1
    max_p = min(max_period, n // 3)
    acf = np.zeros(max_p + 1)
    for k in range(1, max_p + 1):
        acf[k] = np.dot(y_c[:n - k], y_c[k:]) / var
    # Find first significant peak
    for k in range(2, max_p):
        if acf[k] > acf[k - 1] and acf[k] > acf[k + 1] and acf[k] > 0.15:
            return k
    return 1


def _seasonal_decompose(y: np.ndarray, period: int):
    """Extract seasonal component via period-averaging. Returns (seasonal, deseasoned)."""
    n = len(y)
    if period <= 1 or period > n // 2:
        return np.zeros(n), y.copy()
    # Detrend for seasonal estimation
    t = np.arange(n, dtype=float)
    p = np.polyfit(t, y, 1)
    detrended = y - np.polyval(p, t)
    seasonal = np.zeros(period)
    counts = np.zeros(period)
    for i in range(n):
        seasonal[i % period] += detrended[i]
        counts[i % period] += 1
    seasonal /= np.maximum(counts, 1)
    seasonal -= np.mean(seasonal)
    full_seasonal = np.tile(seasonal, n // period + 1)[:n]
    return full_seasonal, y - full_seasonal


def _seasonal_forecast(seasonal_pattern: np.ndarray, period: int,
                       n_train: int, h: int) -> np.ndarray:
    """Extrapolate the seasonal pattern h steps beyond training."""
    offset = n_train % period
    result = np.zeros(h)
    for i in range(h):
        result[i] = seasonal_pattern[(offset + i) % period]
    return result


# -----------------------------------------------------------------------
# DMD core
# -----------------------------------------------------------------------

def _optimal_hard_threshold(sv: np.ndarray, m: int, n: int) -> int:
    """
    Optimal hard threshold for singular values (Gavish & Donoho 2014).
    Returns the number of singular values to keep.
    """
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    threshold = omega * np.median(sv)
    r = int(np.sum(sv > threshold))
    return max(1, r)


def _dmd(X: np.ndarray, Y: np.ndarray, rank: int = 0,
         damping: float = 1.0):
    """
    Exact Dynamic Mode Decomposition.

    Parameters
    ----------
    X : (p, M) matrix of current states
    Y : (p, M) matrix of next states
    rank : int
        SVD truncation rank. 0 = auto via optimal hard threshold.
    damping : float
        Clamp eigenvalue magnitudes to this value. 1.0 = neutral.

    Returns
    -------
    eigenvalues : (r,) complex array
    modes : (p, r) complex array  (Φ)
    amplitudes : (r,) complex array (b = Φ^+ · x_last)
    """
    p, M = X.shape

    # Step 1: SVD of X
    U, sv, Vh = np.linalg.svd(X, full_matrices=False)

    # Step 2: Determine rank
    if rank <= 0:
        r = _optimal_hard_threshold(sv, p, M)
    else:
        r = min(rank, len(sv))
    r = min(r, p, M)

    U_r = U[:, :r]
    sv_r = sv[:r]
    Vh_r = Vh[:r, :]

    # Step 3: Build reduced dynamics matrix Ã = U_r^T Y V_r Σ_r^{-1}
    S_inv = np.diag(1.0 / np.maximum(sv_r, 1e-12))
    A_tilde = U_r.T @ Y @ Vh_r.T @ S_inv  # (r, r)

    # Step 4: Eigendecomposition of Ã
    eigenvalues, W = np.linalg.eig(A_tilde)

    # Step 5: DMD modes Φ = Y V_r Σ_r^{-1} W
    modes = Y @ Vh_r.T @ S_inv @ W  # (p, r)

    # Step 6: Eigenvalue damping — clamp magnitudes
    if damping < 1.0:
        mags = np.abs(eigenvalues)
        mask = mags > damping
        eigenvalues[mask] = eigenvalues[mask] / mags[mask] * damping

    # Step 7: Compute amplitudes from the last state
    x_last = X[:, -1]
    # b = Φ^+ · x_last  (pseudoinverse projection)
    b, *_ = np.linalg.lstsq(modes, x_last, rcond=None)

    return eigenvalues, modes, b


# -----------------------------------------------------------------------
# Forecaster
# -----------------------------------------------------------------------

class KoopmanForecaster:
    """
    Koopman / DMD forecaster for univariate time series.

    Parameters
    ----------
    embed_dim : int, default=12
        Delay embedding dimension (number of lags in state vector).
    rank : int, default=0
        SVD truncation rank for DMD. 0 = automatic via optimal threshold.
    damping : float, default=0.98
        Maximum allowed eigenvalue magnitude. Eigenvalues with |λ| > damping
        are rescaled to prevent explosive forecasts.
    deseason : bool, default=True
        Whether to auto-detect and remove seasonality before DMD.
    detrend : bool, default=True
        Whether to remove linear trend before DMD.
    """

    def __init__(self, embed_dim: int = 12, rank: int = 0,
                 damping: float = 0.98, deseason: bool = True,
                 detrend: bool = True):
        self.embed_dim = int(embed_dim)
        self.rank = int(rank)
        self.damping = float(damping)
        self.deseason = deseason
        self.detrend = detrend
        self._fitted = False
        self._y: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._modes: Optional[np.ndarray] = None
        self._amplitudes: Optional[np.ndarray] = None
        self._x_last: Optional[np.ndarray] = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self._trend: Optional[np.ndarray] = None  # [slope, intercept]
        self._seasonal_pattern: Optional[np.ndarray] = None
        self._period: int = 1
        self._n_train: int = 0

    def fit(self, y: np.ndarray) -> "KoopmanForecaster":
        y = np.asarray(y, float)
        n = len(y)
        p = self.embed_dim
        if n < p + 4:
            raise ValueError(f"Need at least {p + 4} points, got {n}.")

        self._y = y.copy()
        self._n_train = n

        # --- Preprocessing ---
        work = y.copy()

        # 1. Deseasonalize
        if self.deseason:
            self._period = _detect_period(y)
            if self._period > 1:
                self._seasonal_pattern, work = _seasonal_decompose(work, self._period)
            else:
                self._seasonal_pattern = np.zeros(n)
        else:
            self._period = 1
            self._seasonal_pattern = np.zeros(n)

        # 2. Detrend
        if self.detrend:
            t_idx = np.arange(n, dtype=float)
            self._trend = np.polyfit(t_idx, work, 1)
            work = work - np.polyval(self._trend, t_idx)
        else:
            self._trend = np.array([0.0, 0.0])

        # 3. Standardize
        self._mean = np.mean(work)
        self._std = np.std(work)
        if self._std < 1e-12:
            self._std = 1.0
        work = (work - self._mean) / self._std

        # --- Build delay-embedded state matrices ---
        M = n - p  # number of state vectors
        if M < 2:
            raise ValueError("Series too short for given embed_dim.")

        # X[:, j] = [work[j+p-1], work[j+p-2], ..., work[j]]  (newest first)
        X = np.zeros((p, M))
        Y_mat = np.zeros((p, M))
        for j in range(M):
            X[:, j] = work[j: j + p][::-1]
        for j in range(M):
            Y_mat[:, j] = work[j + 1: j + p + 1][::-1]

        # --- DMD ---
        self._eigenvalues, self._modes, self._amplitudes = _dmd(
            X, Y_mat, rank=self.rank, damping=self.damping
        )

        # Store last state for forecasting
        self._x_last = X[:, -1].copy()

        self._fitted = True
        return self

    def predict(self, h: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")

        n = self._n_train
        eigenvalues = self._eigenvalues
        modes = self._modes
        b = self._amplitudes

        # --- DMD forecast (exact multi-step) ---
        forecasts_std = np.zeros(h)
        for step in range(h):
            # x_{n+step} = Φ · Λ^{step+1} · b
            powers = eigenvalues ** (step + 1)
            x_forecast = modes @ (powers * b)
            # Take real part of the first component (most recent value)
            forecasts_std[step] = np.real(x_forecast[0])

        # --- Reverse preprocessing ---
        # 1. Un-standardize
        forecasts = forecasts_std * self._std + self._mean

        # 2. Add trend
        if self.detrend and self._trend is not None:
            t_forecast = np.arange(n, n + h, dtype=float)
            forecasts += np.polyval(self._trend, t_forecast)

        # 3. Add seasonal component
        if self.deseason and self._period > 1 and self._seasonal_pattern is not None:
            sp = np.zeros(self._period)
            counts = np.zeros(self._period)
            pattern = self._seasonal_pattern
            for i in range(len(pattern)):
                sp[i % self._period] += pattern[i]
                counts[i % self._period] += 1
            sp /= np.maximum(counts, 1)
            forecasts += _seasonal_forecast(sp, self._period, n, h)

        return forecasts


# -----------------------------------------------------------------------
# Auto tuner
# -----------------------------------------------------------------------

class AutoKoopman:
    """
    Auto-tuned Koopman / DMD forecaster.

    Performs grid search over embedding dimension, SVD rank, eigenvalue
    damping, deseasonalization, and detrending options.

    Parameters
    ----------
    embed_grid : iterable of int
        Candidate delay embedding dimensions.
    rank_grid : iterable of int
        Candidate SVD truncation ranks (0 = auto).
    damping_grid : iterable of float
        Candidate eigenvalue damping factors.
    deseason_grid : iterable of bool
        Whether to try deseasonalization on/off.
    detrend_grid : iterable of bool
        Whether to try detrending on/off.
    """

    def __init__(
        self,
        embed_grid: Iterable[int] = (4, 8, 12, 16, 24),
        rank_grid: Iterable[int] = (0, 3, 6, 10),
        damping_grid: Iterable[float] = (0.95, 0.98, 1.0),
        deseason_grid: Iterable[bool] = (True, False),
        detrend_grid: Iterable[bool] = (True, False),
    ):
        self.embed_grid = embed_grid
        self.rank_grid = rank_grid
        self.damping_grid = damping_grid
        self.deseason_grid = deseason_grid
        self.detrend_grid = detrend_grid
        self.model_: Optional[KoopmanForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction: float = 0.25,
            metric: str = "mae") -> "AutoKoopman":
        y = np.asarray(y, float)
        N = len(y)
        n_val = max(6, int(N * val_fraction))
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for p in self.embed_grid:
            if p >= len(y_train) - 3:
                continue
            for r in self.rank_grid:
                for d in self.damping_grid:
                    for ds in self.deseason_grid:
                        for dt in self.detrend_grid:
                            try:
                                m = KoopmanForecaster(
                                    embed_dim=p, rank=r,
                                    damping=d, deseason=ds,
                                    detrend=dt)
                                m.fit(y_train)
                                preds = m.predict(len(y_val))
                                s = score_fn(y_val, preds)
                                if np.isfinite(s) and s < best_score:
                                    best_score = s
                                    best_conf = dict(
                                        embed_dim=p, rank=r,
                                        damping=d, deseason=ds,
                                        detrend=dt)
                            except Exception:
                                continue

        if best_conf is None:
            best_conf = dict(embed_dim=8, rank=0, damping=0.98,
                             deseason=True, detrend=True)
            best_score = float("inf")

        final = KoopmanForecaster(**best_conf)
        final.fit(y)
        self.model_ = final
        self.best_ = dict(config=best_conf, val_score=best_score)
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the model before predicting.")
        return self.model_.predict(h)
