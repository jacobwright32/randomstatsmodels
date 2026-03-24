"""
Spectral Gradient Flow Forecaster
==================================

Exploits the fundamental calculus-in-frequency-domain identity:

    d^n y / dt^n  ↔  (iω)^n · Y(ω)

Rather than working with derivatives in the time domain, this model tracks
how the spectral content of the series **evolves** across overlapping time
windows (Short-Time Fourier Transform style).  It computes first- and
second-order *time-derivatives of the spectral magnitudes and phases*,
extrapolates them forward, and reconstructs the forecast via inverse FFT.

This creates a "spectral velocity field" — a gradient flow in frequency
space — that captures how energy moves between frequencies over time.

Mathematical foundation
-----------------------
Given overlapping windows w_1, w_2, …, w_K of the series, compute:

    S_k(ω) = FFT(w_k)   for each window k
    A_k(ω) = |S_k(ω)|   (amplitude envelope)
    φ_k(ω) = arg(S_k(ω)) (instantaneous phase)

Spectral velocity:     dA/dk ≈ (A_{k} - A_{k-1}) / Δ
Spectral acceleration: d²A/dk² ≈ (A_{k} - 2A_{k-1} + A_{k-2}) / Δ²

Extrapolate: A_{K+j} = A_K + j·dA/dk + 0.5·j²·d²A/dk²  (damped)
Similarly for phase: φ_{K+j} = φ_K + j·dφ/dk

Reconstruct: y_hat = IFFT(A_extrap · exp(i·φ_extrap))
"""

from typing import Optional, Dict, Iterable
import numpy as np
from ..metrics import mae, rmse


class SpectralGradientForecaster:
    """
    Forecaster based on spectral derivative flow.

    Parameters
    ----------
    window_size : int, default=32
        Length of each STFT window.
    hop : int, default=8
        Hop size between successive windows.
    n_components : int, default=0
        Number of Fourier components to track (0 = all up to Nyquist).
    damping : float, default=0.9
        Damping factor applied to spectral acceleration extrapolation
        to prevent divergence.
    trend_weight : float, default=0.5
        Weight for linear trend continuation in the final forecast blend.
    """

    def __init__(self, window_size: int = 32, hop: int = 8,
                 n_components: int = 0, damping: float = 0.9,
                 trend_weight: float = 0.5):
        self.window_size = int(window_size)
        self.hop = int(hop)
        self.n_components = int(n_components)
        self.damping = float(damping)
        self.trend_weight = float(trend_weight)
        self._fitted = False
        self._y: Optional[np.ndarray] = None
        self._A_last: Optional[np.ndarray] = None
        self._phi_last: Optional[np.ndarray] = None
        self._dA: Optional[np.ndarray] = None
        self._ddA: Optional[np.ndarray] = None
        self._dphi: Optional[np.ndarray] = None
        self._mean: float = 0.0
        self._trend_slope: float = 0.0

    def fit(self, y: np.ndarray) -> "SpectralGradientForecaster":
        y = np.asarray(y, float)
        n = len(y)
        ws = min(self.window_size, n // 2)
        if ws < 4:
            ws = 4
        hop = max(1, min(self.hop, ws // 2))

        self._y = y.copy()
        self._mean = np.mean(y)

        # Estimate linear trend
        t_idx = np.arange(n, dtype=float)
        self._trend_slope = float(np.polyfit(t_idx, y, 1)[0])

        # Detrend for spectral analysis
        y_detrend = y - (self._mean + self._trend_slope * (t_idx - n / 2))

        # Build STFT frames
        starts = list(range(0, n - ws + 1, hop))
        if len(starts) < 3:
            starts = [max(0, n - ws * 3), max(0, n - ws * 2), n - ws]
            starts = [s for s in starts if s >= 0]

        hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(ws) / ws))
        n_freq = ws // 2 + 1
        nc = self.n_components if self.n_components > 0 else n_freq
        nc = min(nc, n_freq)

        # Compute amplitude and phase trajectories
        A_traj = []
        phi_traj = []
        for s in starts:
            frame = y_detrend[s: s + ws] * hann
            spec = np.fft.rfft(frame)
            A_traj.append(np.abs(spec[:nc]))
            phi_traj.append(np.angle(spec[:nc]))

        A_traj = np.array(A_traj)   # shape (K, nc)
        phi_traj = np.array(phi_traj)
        K = len(A_traj)

        # Unwrap phase for smoother derivatives
        phi_traj = np.unwrap(phi_traj, axis=0)

        # Spectral velocity (first derivative w.r.t. window index)
        if K >= 2:
            dA = A_traj[-1] - A_traj[-2]
            dphi = phi_traj[-1] - phi_traj[-2]
        else:
            dA = np.zeros(nc)
            dphi = np.zeros(nc)

        # Spectral acceleration (second derivative)
        if K >= 3:
            ddA = A_traj[-1] - 2 * A_traj[-2] + A_traj[-3]
        else:
            ddA = np.zeros(nc)

        self._A_last = A_traj[-1].copy()
        self._phi_last = phi_traj[-1].copy()
        self._dA = dA
        self._ddA = ddA
        self._dphi = dphi
        self._nc = nc
        self._ws = ws
        self._fitted = True
        return self

    def predict(self, h: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")

        y = self._y
        n = len(y)
        nc = self._nc
        ws = self._ws
        damping = self.damping

        # We produce one reconstructed window per "step" and overlap-add
        # For simplicity, produce a single extrapolated spectrum that
        # covers h steps, using the spectral derivatives.

        # Number of extrapolation steps (in window-index space)
        # We map h time steps to fractional window steps
        steps_per_window = max(1, self.hop)

        forecasts = np.zeros(h)

        for t in range(h):
            # How far ahead in "window steps" this forecast point is
            j = (t + 1) / steps_per_window

            # Extrapolate amplitude (with damping on acceleration)
            A_ext = self._A_last + j * self._dA + \
                    0.5 * j * j * self._ddA * (damping ** j)
            A_ext = np.maximum(A_ext, 0.0)  # amplitudes must be non-negative

            # Extrapolate phase (linear)
            phi_ext = self._phi_last + j * self._dphi

            # Reconstruct spectrum
            spec_ext = A_ext * np.exp(1j * phi_ext)

            # Pad to full rfft size if needed
            spec_full = np.zeros(ws // 2 + 1, dtype=complex)
            spec_full[:nc] = spec_ext

            # Inverse FFT to get time-domain window
            window_ext = np.fft.irfft(spec_full, n=ws)

            # Take the centre point of the window as our forecast value
            # (this represents the "most confident" point in the window)
            centre = ws // 2
            forecasts[t] = window_ext[centre % ws]

        # Add back the trend
        t_forecast = np.arange(h, dtype=float)
        trend = self._mean + self._trend_slope * (n + t_forecast - n / 2)

        # Blend spectral reconstruction with trend
        tw = self.trend_weight
        result = tw * trend + (1 - tw) * (trend + forecasts)

        return result


class AutoSpectralGradient:
    """
    Auto-tuned spectral gradient flow forecaster.

    Grid-searches over window size, hop, damping, and trend weight.
    """

    def __init__(
        self,
        window_grid: Iterable[int] = (16, 32, 48),
        hop_grid: Iterable[int] = (4, 8),
        damping_grid: Iterable[float] = (0.7, 0.85, 0.95),
        trend_weight_grid: Iterable[float] = (0.3, 0.5, 0.7, 0.9),
    ):
        self.window_grid = window_grid
        self.hop_grid = hop_grid
        self.damping_grid = damping_grid
        self.trend_weight_grid = trend_weight_grid
        self.model_: Optional[SpectralGradientForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction: float = 0.25,
            metric: str = "mae") -> "AutoSpectralGradient":
        y = np.asarray(y, float)
        N = len(y)
        n_val = max(6, int(N * val_fraction))
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for ws in self.window_grid:
            if ws > len(y_train) // 2:
                continue
            for hop in self.hop_grid:
                for damp in self.damping_grid:
                    for tw in self.trend_weight_grid:
                        try:
                            m = SpectralGradientForecaster(
                                window_size=ws, hop=hop,
                                damping=damp, trend_weight=tw)
                            m.fit(y_train)
                            preds = m.predict(len(y_val))
                            s = score_fn(y_val, preds)
                            if np.isfinite(s) and s < best_score:
                                best_score = s
                                best_conf = dict(
                                    window_size=ws, hop=hop,
                                    damping=damp, trend_weight=tw)
                        except Exception:
                            continue

        if best_conf is None:
            best_conf = dict(window_size=32, hop=8,
                             damping=0.9, trend_weight=0.5)
            best_score = float("inf")

        final = SpectralGradientForecaster(**best_conf)
        final.fit(y)
        self.model_ = final
        self.best_ = dict(config=best_conf, val_score=best_score)
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the model before predicting.")
        return self.model_.predict(h)
