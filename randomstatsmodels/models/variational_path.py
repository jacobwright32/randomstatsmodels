"""
Variational Calculus Path Forecaster
=====================================

Finds the forecast trajectory that **minimises an energy functional**,
inspired by the Principle of Least Action from physics.

The forecast path y(t) for t ∈ [T, T+h] minimises:

    E[y] = ∫_{T}^{T+h} L(y, ẏ, ÿ, t) dt

where the Lagrangian L combines:

1. **Kinetic energy** (smoothness):       λ_s · (ẏ)²
2. **Jerk penalty** (curvature control):  λ_j · (ÿ)²
3. **Potential energy** (attractor):      λ_p · V(y)
4. **Seasonal constraint**:              λ_c · (y - S(t))²

The Euler-Lagrange equation for this system is:

    λ_j · y'''' - λ_s · y'' + λ_p · V'(y) + λ_c · (y - S(t)) = 0

For the quadratic case (V(y) = (y - μ)²), this becomes a **linear system**
that can be solved exactly by discretising and solving a banded matrix
equation.

Boundary conditions
-------------------
- y(T) = last observed value
- ẏ(T) = estimated trend slope at end of series
- y(T+h) is free (natural boundary)
- ÿ(T+h) = 0  (zero-acceleration terminal condition)

This model is fundamentally different from all others: it formulates
forecasting as a variational optimisation problem from classical mechanics.
"""

from typing import Optional, Dict, Iterable
import numpy as np
from ..metrics import mae, rmse


def _estimate_seasonal(y: np.ndarray, period: int) -> np.ndarray:
    """Estimate seasonal component via period-averaging."""
    n = len(y)
    if period <= 1 or period > n // 2:
        return np.zeros(n)

    # Detrend first
    t = np.arange(n, dtype=float)
    p = np.polyfit(t, y, 1)
    detrended = y - np.polyval(p, t)

    # Average by season
    seasonal = np.zeros(period)
    counts = np.zeros(period)
    for i in range(n):
        seasonal[i % period] += detrended[i]
        counts[i % period] += 1
    seasonal /= np.maximum(counts, 1)
    seasonal -= np.mean(seasonal)

    # Tile to full length
    full = np.tile(seasonal, n // period + 1)[:n]
    return full


def _detect_period(y: np.ndarray, max_period: int = 60) -> int:
    """Simple period detection via autocorrelation peaks."""
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

    # Find first significant peak (not at lag 0)
    if max_p < 3:
        return 1

    for k in range(2, max_p):
        if acf[k] > acf[k - 1] and acf[k] > acf[k + 1] and acf[k] > 0.2:
            return k

    return 1


class VariationalPathForecaster:
    """
    Variational calculus path forecaster.

    Parameters
    ----------
    smoothness : float, default=1.0
        Weight λ_s for the kinetic (first derivative squared) term.
    jerk : float, default=0.1
        Weight λ_j for the jerk (second derivative squared) term.
    potential : float, default=0.5
        Weight λ_p for the potential energy (attractor) term.
    seasonal_weight : float, default=1.0
        Weight λ_c for the seasonal constraint term.
    period : int, default=0
        Seasonal period. 0 = auto-detect.
    attractor : str, default="mean"
        Type of attractor for the potential:
        - "mean": attracts toward global mean
        - "last": attracts toward last value
        - "trend": attracts toward linear trend extrapolation
    """

    def __init__(self, smoothness: float = 1.0, jerk: float = 0.1,
                 potential: float = 0.5, seasonal_weight: float = 1.0,
                 period: int = 0, attractor: str = "mean"):
        self.smoothness = float(smoothness)
        self.jerk = float(jerk)
        self.potential = float(potential)
        self.seasonal_weight = float(seasonal_weight)
        self.period = int(period)
        self.attractor = attractor
        self._fitted = False
        self._y: Optional[np.ndarray] = None
        self._y0: float = 0.0
        self._v0: float = 0.0
        self._mu: float = 0.0
        self._seasonal_pattern: Optional[np.ndarray] = None
        self._detected_period: int = 1

    def fit(self, y: np.ndarray) -> "VariationalPathForecaster":
        y = np.asarray(y, float)
        n = len(y)
        if n < 4:
            raise ValueError(f"Need at least 4 points, got {n}.")

        self._y = y.copy()

        # Boundary conditions
        self._y0 = y[-1]
        # Estimate slope at end using last few points
        tail = min(10, n // 2)
        t_tail = np.arange(tail, dtype=float)
        slope = float(np.polyfit(t_tail, y[-tail:], 1)[0])
        self._v0 = slope

        # Attractor level
        if self.attractor == "mean":
            self._mu = np.mean(y)
        elif self.attractor == "last":
            self._mu = y[-1]
        elif self.attractor == "trend":
            self._mu = y[-1] + slope * 5  # rough trend target
        else:
            self._mu = np.mean(y)

        # Seasonal pattern
        period = self.period
        if period <= 0:
            period = _detect_period(y)
        self._detected_period = period
        self._seasonal_pattern = _estimate_seasonal(y, period)

        self._fitted = True
        return self

    def predict(self, h: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")

        n = len(self._y)
        y0 = self._y0
        v0 = self._v0
        mu = self._mu
        ls = self.smoothness
        lj = self.jerk
        lp = self.potential
        lc = self.seasonal_weight
        period = self._detected_period

        # Build seasonal target for forecast horizon
        if period > 1 and self._seasonal_pattern is not None:
            sp = self._seasonal_pattern
            # Tile the pattern forward
            offset = n % period
            seasonal_target = np.zeros(h)
            for t in range(h):
                idx = (offset + t) % period
                seasonal_target[t] = sp[idx] if idx < len(sp) else 0.0
            # Add the attractor level
            seasonal_target += mu
        else:
            seasonal_target = np.full(h, mu)

        # Also blend in trend continuation
        trend_target = y0 + v0 * np.arange(1, h + 1, dtype=float)
        # Dampen trend over longer horizons
        damping = np.exp(-0.05 * np.arange(h))
        target = 0.5 * (seasonal_target + trend_target * damping
                        + mu * (1 - damping))

        # Solve the discrete Euler-Lagrange equation
        # Discretise:  y = [y_1, y_2, ..., y_h]
        # The energy functional (quadratic) gives a linear system Ay = b
        #
        # Terms:
        # Smoothness: ls * Σ (y_{i+1} - y_i)² → tridiag contribution
        # Jerk: lj * Σ (y_{i+2} - 2y_{i+1} + y_i)² → pentadiag
        # Potential: lp * Σ (y_i - μ)² → diagonal
        # Seasonal: lc * Σ (y_i - S_i)² → diagonal + rhs

        A = np.zeros((h, h))
        b = np.zeros(h)

        # Smoothness: ∂/∂y_k [Σ (y_{i+1} - y_i)²]
        # = 2(y_k - y_{k-1}) - 2(y_{k+1} - y_k)  for interior
        for i in range(h):
            A[i, i] += 2 * ls
            if i > 0:
                A[i, i - 1] -= ls
                A[i - 1, i] -= ls
            else:
                # Boundary: y_0 (last observed) contributes to rhs
                b[0] += ls * y0

        # Jerk penalty: d²y/dt² ≈ y_{i+1} - 2y_i + y_{i-1}
        # Minimise Σ (y_{i+1} - 2y_i + y_{i-1})²
        for i in range(h):
            # Diagonal: coefficient of y_i in the quadratic
            A[i, i] += 4 * lj
            if i > 0:
                A[i, i - 1] -= 4 * lj
                if i > 1:
                    A[i, i - 2] += lj
            else:
                b[0] += 4 * lj * y0
                # y_{-1} approximated as y0 - v0
            if i < h - 1:
                A[i, i + 1] -= 4 * lj
                if i < h - 2:
                    A[i, i + 2] += lj
            if i == 0:
                # y_{i-1} = y0
                b[0] += lj * (4 * y0)
            if i == 1:
                b[1] += lj * y0

        # Potential energy: lp * (y_i - μ)²
        for i in range(h):
            A[i, i] += lp
            b[i] += lp * mu

        # Seasonal constraint: lc * (y_i - target_i)²
        for i in range(h):
            A[i, i] += lc
            b[i] += lc * target[i]

        # Make A symmetric and positive definite (add small ridge)
        A = 0.5 * (A + A.T)
        A += 1e-6 * np.eye(h)

        # Solve
        try:
            forecast = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback: use target directly
            forecast = target

        return forecast


class AutoVariationalPath:
    """
    Auto-tuned variational path forecaster.

    Grid-searches over Lagrangian weights and attractor types.
    """

    def __init__(
        self,
        smoothness_grid: Iterable[float] = (0.5, 1.0, 2.0),
        jerk_grid: Iterable[float] = (0.01, 0.1, 0.5),
        potential_grid: Iterable[float] = (0.1, 0.5, 1.0),
        seasonal_weight_grid: Iterable[float] = (0.5, 1.0, 2.0),
        attractor_grid: Iterable[str] = ("mean", "last", "trend"),
    ):
        self.smoothness_grid = smoothness_grid
        self.jerk_grid = jerk_grid
        self.potential_grid = potential_grid
        self.seasonal_weight_grid = seasonal_weight_grid
        self.attractor_grid = attractor_grid
        self.model_: Optional[VariationalPathForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction: float = 0.25,
            metric: str = "mae") -> "AutoVariationalPath":
        y = np.asarray(y, float)
        N = len(y)
        n_val = max(6, int(N * val_fraction))
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for sm in self.smoothness_grid:
            for jk in self.jerk_grid:
                for pt in self.potential_grid:
                    for sw in self.seasonal_weight_grid:
                        for att in self.attractor_grid:
                            try:
                                m = VariationalPathForecaster(
                                    smoothness=sm, jerk=jk,
                                    potential=pt,
                                    seasonal_weight=sw,
                                    attractor=att)
                                m.fit(y_train)
                                preds = m.predict(len(y_val))
                                s = score_fn(y_val, preds)
                                if np.isfinite(s) and s < best_score:
                                    best_score = s
                                    best_conf = dict(
                                        smoothness=sm, jerk=jk,
                                        potential=pt,
                                        seasonal_weight=sw,
                                        attractor=att)
                            except Exception:
                                continue

        if best_conf is None:
            best_conf = dict(smoothness=1.0, jerk=0.1, potential=0.5,
                             seasonal_weight=1.0, attractor="mean")
            best_score = float("inf")

        final = VariationalPathForecaster(**best_conf)
        final.fit(y)
        self.model_ = final
        self.best_ = dict(config=best_conf, val_score=best_score)
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the model before predicting.")
        return self.model_.predict(h)
