"""
PDE-Based Spatiotemporal Field Forecaster
==========================================

Embeds a 1-D time series into a 2-D field **u(t, s)** where *s* is a
"scale" axis (constructed via multi-scale Gaussian smoothing).  The field
evolution is then modelled with a **partial differential equation** that
mixes advection, diffusion, and reaction terms:

    ∂u/∂t  =  α · ∂²u/∂s²  +  β · ∂u/∂s  +  γ · u  +  δ

The PDE coefficients (α, β, γ, δ) are estimated from the data via least
squares on the interior grid points.  Forecasting propagates the field
forward in time using an explicit Euler scheme; the prediction is read
from the finest scale (s = 0).

Partial derivatives are computed using second-order central finite
differences:

    ∂u/∂t|_{i,j} ≈ (u_{i+1,j} - u_{i-1,j}) / (2·Δt)
    ∂u/∂s|_{i,j} ≈ (u_{i,j+1} - u_{i,j-1}) / (2·Δs)
    ∂²u/∂s²|_{i,j} ≈ (u_{i,j+1} - 2·u_{i,j} + u_{i,j-1}) / Δs²

This model genuinely uses **partial derivatives** in two dimensions
(time × scale) — something fundamentally different from all other
models in this package.
"""

from typing import Optional, Dict, Iterable
import numpy as np
from ..metrics import mae, rmse


def _gaussian_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    """
    1-D Gaussian smoothing with a truncated kernel (NumPy only).
    """
    if sigma < 0.5:
        return y.copy()
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # Pad with edge values
    padded = np.pad(y, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


def _build_field(y: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Build 2-D field u[t, s] by smoothing y at each scale.

    Returns array of shape (len(y), len(scales)).
    """
    n = len(y)
    ns = len(scales)
    field = np.zeros((n, ns))
    for j, sigma in enumerate(scales):
        field[:, j] = _gaussian_smooth(y, sigma)
    return field


class PDEFieldForecaster:
    """
    PDE-based spatiotemporal field forecaster.

    Parameters
    ----------
    n_scales : int, default=8
        Number of scale levels in the field.
    max_sigma : float, default=10.0
        Largest Gaussian smoothing bandwidth.
    ridge : float, default=1e-3
        Ridge regularisation for PDE coefficient estimation.
    dt_steps : int, default=2
        Number of Euler sub-steps per forecast step (for stability).
    """

    def __init__(self, n_scales: int = 8, max_sigma: float = 10.0,
                 ridge: float = 1e-3, dt_steps: int = 2):
        self.n_scales = int(n_scales)
        self.max_sigma = float(max_sigma)
        self.ridge = float(ridge)
        self.dt_steps = max(1, int(dt_steps))
        self._fitted = False
        self._y: Optional[np.ndarray] = None
        self._scales: Optional[np.ndarray] = None
        self._pde_coef: Optional[np.ndarray] = None  # [α, β, γ, δ]
        self._last_field_row: Optional[np.ndarray] = None
        self._mean: float = 0.0

    def fit(self, y: np.ndarray) -> "PDEFieldForecaster":
        y = np.asarray(y, float)
        n = len(y)
        if n < 10:
            raise ValueError(f"Need at least 10 points, got {n}.")

        self._y = y.copy()
        self._mean = np.mean(y)

        # Build scale axis
        ns = min(self.n_scales, n // 3)
        ns = max(ns, 3)
        scales = np.linspace(0.0, self.max_sigma, ns)
        self._scales = scales
        ds = scales[1] - scales[0] if ns > 1 else 1.0

        # Build the 2-D field u[t, s]
        field = _build_field(y, scales)

        # Compute partial derivatives on interior grid points
        # We need t ∈ [1, n-2] and s ∈ [1, ns-2]
        t_interior = slice(1, n - 1)
        s_interior = slice(1, ns - 1)
        dt = 1.0  # time step = 1 observation

        # ∂u/∂t  (central difference in time)
        du_dt = (field[2:, s_interior] - field[:-2, s_interior]) / (2 * dt)

        # ∂u/∂s  (central difference in scale)
        du_ds = (field[t_interior, 2:] - field[t_interior, :-2]) / (2 * ds)

        # ∂²u/∂s² (second central difference in scale)
        d2u_ds2 = (field[t_interior, 2:] - 2 * field[t_interior, s_interior]
                   + field[t_interior, :-2]) / (ds ** 2)

        # u values at interior points
        u_int = field[t_interior, s_interior]

        # Flatten for least-squares:  du/dt = α·d²u/ds² + β·du/ds + γ·u + δ
        lhs = du_dt.ravel()
        rhs = np.column_stack([
            d2u_ds2.ravel(),
            du_ds.ravel(),
            u_int.ravel(),
            np.ones(lhs.size),
        ])

        # Ridge regression
        XtX = rhs.T @ rhs + self.ridge * np.eye(4)
        XtY = rhs.T @ lhs
        self._pde_coef = np.linalg.solve(XtX, XtY)

        # Store last two field rows (needed for time-stepping)
        self._last_field_row = field[-1, :].copy()
        self._prev_field_row = field[-2, :].copy()

        self._fitted = True
        return self

    def predict(self, h: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")

        alpha, beta, gamma, delta = self._pde_coef
        scales = self._scales
        ns = len(scales)
        ds = scales[1] - scales[0] if ns > 1 else 1.0
        dt_sub = 1.0 / self.dt_steps

        # Stability: clamp diffusion coefficient
        max_alpha = 0.4 * ds ** 2 / dt_sub if ds > 0 else 0.0
        alpha = np.clip(alpha, -max_alpha, max_alpha)

        # Current and previous field rows
        u_curr = self._last_field_row.copy()
        u_prev = self._prev_field_row.copy()

        forecasts = np.zeros(h)

        for step in range(h):
            for _ in range(self.dt_steps):
                u_new = u_curr.copy()

                for j in range(1, ns - 1):
                    d2u = (u_curr[j + 1] - 2 * u_curr[j] + u_curr[j - 1]) / (ds ** 2)
                    du = (u_curr[j + 1] - u_curr[j - 1]) / (2 * ds)
                    rhs = alpha * d2u + beta * du + gamma * u_curr[j] + delta
                    u_new[j] = u_curr[j] + dt_sub * rhs

                # Boundary conditions: Neumann (zero gradient) at edges
                u_new[0] = u_new[1]
                u_new[-1] = u_new[-2]

                u_prev = u_curr.copy()
                u_curr = u_new

            # Read forecast from finest scale (s=0)
            forecasts[step] = u_curr[0]

        return forecasts


class AutoPDEField:
    """
    Auto-tuned PDE field forecaster.

    Grid-searches over field resolution and PDE estimation parameters.
    """

    def __init__(
        self,
        n_scales_grid: Iterable[int] = (5, 8, 12),
        max_sigma_grid: Iterable[float] = (5.0, 10.0, 20.0),
        dt_steps_grid: Iterable[int] = (1, 2, 4),
    ):
        self.n_scales_grid = n_scales_grid
        self.max_sigma_grid = max_sigma_grid
        self.dt_steps_grid = dt_steps_grid
        self.model_: Optional[PDEFieldForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction: float = 0.25,
            metric: str = "mae") -> "AutoPDEField":
        y = np.asarray(y, float)
        N = len(y)
        n_val = max(6, int(N * val_fraction))
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for ns in self.n_scales_grid:
            for ms in self.max_sigma_grid:
                for dts in self.dt_steps_grid:
                    try:
                        m = PDEFieldForecaster(
                            n_scales=ns, max_sigma=ms, dt_steps=dts)
                        m.fit(y_train)
                        preds = m.predict(len(y_val))
                        s = score_fn(y_val, preds)
                        if np.isfinite(s) and s < best_score:
                            best_score = s
                            best_conf = dict(
                                n_scales=ns, max_sigma=ms, dt_steps=dts)
                    except Exception:
                        continue

        if best_conf is None:
            best_conf = dict(n_scales=8, max_sigma=10.0, dt_steps=2)
            best_score = float("inf")

        final = PDEFieldForecaster(**best_conf)
        final.fit(y)
        self.model_ = final
        self.best_ = dict(config=best_conf, val_score=best_score)
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the model before predicting.")
        return self.model_.predict(h)
