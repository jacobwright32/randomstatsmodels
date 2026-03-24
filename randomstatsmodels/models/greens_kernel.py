"""
Green's Function Integral Kernel Forecaster
============================================

Treats the time series as the output of a linear time-invariant system
driven by a latent forcing function.  The forecast is computed as a
**numerical integral convolution** of a learned Green's function kernel
with the observed data.

Mathematical foundation
-----------------------
A Green's function G(t, τ) is the impulse response of a differential
operator L:

    L[y](t) = f(t)   ⟹   y(t) = ∫₀ᵗ G(t - τ) · f(τ) dτ

We model G as a superposition of exponentially-damped oscillators:

    G(τ) = Σᵢ aᵢ · exp(-λᵢ · τ) · cos(ωᵢ · τ + φᵢ)

The kernel parameters {aᵢ, λᵢ, ωᵢ, φᵢ} are estimated from the
empirical autocorrelation function of the series.

Forecast via numerical quadrature
----------------------------------
    ŷ(T + h) = ∫₀ᵀ G(T + h - τ) · y(τ) dτ
             ≈ Σⱼ G(T + h - τⱼ) · y(τⱼ) · Δτ     (trapezoidal rule)

The integral sums up contributions from all past observations, weighted
by the Green's function evaluated at the time lag.
"""

from typing import Optional, Dict, Iterable
import numpy as np
from ..metrics import mae, rmse


def _estimate_acf(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocorrelation function."""
    y = y - np.mean(y)
    n = len(y)
    var = np.dot(y, y) / n
    if var < 1e-12:
        return np.ones(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        acf[k] = np.dot(y[:n - k], y[k:]) / (n * var)
    return acf


def _fit_kernel_params(acf: np.ndarray, n_components: int):
    """
    Fit Green's function kernel parameters from the ACF.

    We decompose the ACF into damped-cosine components using a
    sequential fitting strategy (Prony-like but simpler).
    """
    max_lag = len(acf) - 1
    tau = np.arange(max_lag + 1, dtype=float)
    residual = acf.copy()
    components = []

    for _ in range(n_components):
        if np.max(np.abs(residual[1:])) < 0.01:
            break

        # Find dominant frequency via zero crossings of residual
        signs = np.sign(residual[1:])
        crossings = np.where(np.diff(signs) != 0)[0]

        if len(crossings) >= 2:
            # Estimate half-period from average zero-crossing interval
            half_period = np.mean(np.diff(crossings))
            omega = np.pi / max(half_period, 1.0)
        elif len(crossings) == 1:
            omega = np.pi / max(crossings[0] + 1, 1.0)
        else:
            omega = 0.0  # pure decay, no oscillation

        # Estimate decay rate from envelope
        env = np.abs(residual)
        # Find where envelope drops to 1/e of its peak
        peak = np.max(env)
        if peak > 1e-10:
            below = np.where(env < peak / np.e)[0]
            if len(below) > 0 and below[0] > 0:
                lam = 1.0 / max(float(below[0]), 0.5)
            else:
                lam = 0.1 / max_lag
        else:
            lam = 1.0

        # Estimate amplitude and phase via least squares
        # G_i(τ) = a * exp(-λτ) * cos(ωτ + φ)
        #        = exp(-λτ) * [a*cos(φ)*cos(ωτ) - a*sin(φ)*sin(ωτ)]
        decay = np.exp(-lam * tau)
        c_part = decay * np.cos(omega * tau)
        s_part = decay * np.sin(omega * tau)
        A_mat = np.column_stack([c_part, s_part])
        try:
            coeffs, *_ = np.linalg.lstsq(A_mat, residual, rcond=None)
            alpha, beta = coeffs
        except Exception:
            alpha, beta = residual[0], 0.0

        a = np.sqrt(alpha ** 2 + beta ** 2)
        phi = np.arctan2(-beta, alpha)

        components.append((a, lam, omega, phi))
        fitted = a * np.exp(-lam * tau) * np.cos(omega * tau + phi)
        residual = residual - fitted

    return components


def _eval_kernel(tau: np.ndarray, components: list) -> np.ndarray:
    """Evaluate the Green's function kernel at given lag values."""
    G = np.zeros_like(tau, dtype=float)
    for (a, lam, omega, phi) in components:
        G += a * np.exp(-lam * tau) * np.cos(omega * tau + phi)
    return G


class GreensKernelForecaster:
    """
    Green's function integral kernel forecaster.

    Parameters
    ----------
    n_components : int, default=3
        Number of damped-cosine components in the kernel.
    max_lag : int, default=50
        Maximum autocorrelation lag for kernel estimation.
    kernel_window : int, default=0
        If > 0, only use the last `kernel_window` observations in the
        integral (truncated kernel). 0 = use all.
    ridge : float, default=0.01
        Regularisation for the integral weights.
    """

    def __init__(self, n_components: int = 3, max_lag: int = 50,
                 kernel_window: int = 0, ridge: float = 0.01):
        self.n_components = int(n_components)
        self.max_lag = int(max_lag)
        self.kernel_window = int(kernel_window)
        self.ridge = float(ridge)
        self._fitted = False
        self._y: Optional[np.ndarray] = None
        self._components: list = []
        self._mean: float = 0.0
        self._scale: float = 1.0

    def fit(self, y: np.ndarray) -> "GreensKernelForecaster":
        y = np.asarray(y, float)
        n = len(y)
        if n < 8:
            raise ValueError(f"Need at least 8 points, got {n}.")

        self._mean = np.mean(y)
        self._scale = np.std(y) if np.std(y) > 1e-12 else 1.0
        self._y = y.copy()

        # Standardise for ACF estimation
        y_std = (y - self._mean) / self._scale
        ml = min(self.max_lag, n // 2)
        acf = _estimate_acf(y_std, ml)

        self._components = _fit_kernel_params(acf, self.n_components)
        if not self._components:
            # Fallback: simple exponential decay
            self._components = [(1.0, 0.1, 0.0, 0.0)]

        self._fitted = True
        return self

    def predict(self, h: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")

        y = self._y
        n = len(y)
        mean = self._mean
        scale = self._scale
        y_std = (y - mean) / scale

        # Determine effective window
        kw = self.kernel_window if self.kernel_window > 0 else n
        kw = min(kw, n)

        # Use the last kw observations for the integral
        y_window = y_std[-kw:]
        nw = len(y_window)

        forecasts = np.zeros(h)
        for step in range(h):
            # Compute lags from the forecast point to each observation
            # tau_j = (n + step) - (n - kw + j) = kw - j + step
            tau_vals = np.arange(kw, 0, -1, dtype=float) + step
            G_vals = _eval_kernel(tau_vals, self._components)

            # Normalise kernel weights (trapezoidal rule approximation)
            G_abs = np.abs(G_vals)
            total = np.sum(G_abs) + self.ridge * kw
            if total > 1e-12:
                weights = G_vals / total
            else:
                weights = np.ones(nw) / nw

            # Integral convolution: ŷ = Σ w_j · y_j
            forecasts[step] = np.dot(weights, y_window)

        # Un-standardise
        return forecasts * scale + mean


class AutoGreensKernel:
    """
    Auto-tuned Green's function integral kernel forecaster.

    Grid-searches over kernel parameters.
    """

    def __init__(
        self,
        n_components_grid: Iterable[int] = (2, 3, 5),
        max_lag_grid: Iterable[int] = (20, 40, 60),
        kernel_window_grid: Iterable[int] = (0, 30, 60),
    ):
        self.n_components_grid = n_components_grid
        self.max_lag_grid = max_lag_grid
        self.kernel_window_grid = kernel_window_grid
        self.model_: Optional[GreensKernelForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction: float = 0.25,
            metric: str = "mae") -> "AutoGreensKernel":
        y = np.asarray(y, float)
        N = len(y)
        n_val = max(6, int(N * val_fraction))
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for nc in self.n_components_grid:
            for ml in self.max_lag_grid:
                if ml > len(y_train) // 2:
                    continue
                for kw in self.kernel_window_grid:
                    try:
                        m = GreensKernelForecaster(
                            n_components=nc, max_lag=ml,
                            kernel_window=kw)
                        m.fit(y_train)
                        preds = m.predict(len(y_val))
                        s = score_fn(y_val, preds)
                        if np.isfinite(s) and s < best_score:
                            best_score = s
                            best_conf = dict(
                                n_components=nc, max_lag=ml,
                                kernel_window=kw)
                    except Exception:
                        continue

        if best_conf is None:
            best_conf = dict(n_components=3, max_lag=40, kernel_window=0)
            best_score = float("inf")

        final = GreensKernelForecaster(**best_conf)
        final.fit(y)
        self.model_ = final
        self.best_ = dict(config=best_conf, val_score=best_score)
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the model before predicting.")
        return self.model_.predict(h)
