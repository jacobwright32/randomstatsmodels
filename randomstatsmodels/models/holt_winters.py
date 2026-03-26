import numpy as np
from typing import Optional, Union, Sequence, Dict, Tuple

from ..metrics import mae, rmse

ArrayLike = Union[Sequence[float], np.ndarray]


# ================= HoltWintersForecaster =================
class HoltWintersForecaster:
    """
    Holt-Winters Exponential Smoothing forecaster.

    Implements triple exponential smoothing with:
    - Level component (controlled by alpha)
    - Trend component (controlled by beta, optional damping with phi)
    - Seasonal component (controlled by gamma)

    Parameters
    ----------
    seasonal_period : int, default=12
        Length of the seasonal cycle.
    trend : {"add", "none", "damped"}, default="add"
        Type of trend component.
    seasonal : {"add", "none"}, default="add"
        Type of seasonal component ("add" for additive, "none" for no seasonality).
    alpha : float, default=0.3
        Smoothing parameter for level (0 < alpha <= 1).
    beta : float, default=0.1
        Smoothing parameter for trend (0 <= beta <= 1).
    gamma : float, default=0.1
        Smoothing parameter for seasonality (0 <= gamma <= 1).
    phi : float, default=0.98
        Damping parameter for damped trend (0 < phi <= 1).

    Attributes
    ----------
    level_ : float
        Final level after fitting.
    trend_ : float
        Final trend after fitting.
    seasonal_ : ndarray
        Final seasonal indices after fitting.
    fitted_ : bool
        Whether the model has been fitted.
    """

    def __init__(
        self,
        seasonal_period: int = 12,
        trend: str = "add",
        seasonal: str = "add",
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.1,
        phi: float = 0.98,
    ) -> None:
        self.seasonal_period = int(seasonal_period)
        self.trend = trend.lower()
        self.seasonal = seasonal.lower()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.phi = float(phi)

        if self.trend not in ("add", "none", "damped"):
            raise ValueError("trend must be 'add', 'none', or 'damped'")
        if self.seasonal not in ("add", "none"):
            raise ValueError("seasonal must be 'add' or 'none'")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 < self.phi <= 1:
            raise ValueError("phi must be in (0, 1]")

        self.level_: Optional[float] = None
        self.trend_: Optional[float] = None
        self.seasonal_: Optional[np.ndarray] = None
        self.fitted_: bool = False

    def _initialize(self, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Initialize level, trend, and seasonal components."""
        m = self.seasonal_period
        n = len(y)

        # Initialize level: average of first season
        if n >= m:
            level = np.mean(y[:m])
        else:
            level = np.mean(y)

        # Initialize trend
        if self.trend in ("add", "damped") and n >= 2 * m:
            # Average of differences between corresponding points in first two seasons
            trend = np.mean((y[m : 2 * m] - y[:m]) / m)
        elif self.trend in ("add", "damped") and n >= 2:
            trend = (y[-1] - y[0]) / (n - 1)
        else:
            trend = 0.0

        # Initialize seasonal indices
        if self.seasonal == "add" and n >= m:
            seasonal = np.zeros(m)
            for i in range(m):
                # Average deviation from level for each seasonal position
                indices = range(i, n, m)
                seasonal[i] = np.mean([y[j] for j in indices]) - level
            # Center the seasonal components
            seasonal = seasonal - np.mean(seasonal)
        else:
            seasonal = np.zeros(m)

        return level, trend, seasonal

    def fit(self, y: ArrayLike) -> "HoltWintersForecaster":
        """
        Fit the Holt-Winters model to the time series.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Training time series.

        Returns
        -------
        self : HoltWintersForecaster
            Fitted instance.

        Raises
        ------
        ValueError
            If series is too short.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        m = self.seasonal_period

        min_len = max(2, m) if self.seasonal == "add" else 2
        if n < min_len:
            raise ValueError(f"Need at least {min_len} data points.")

        # Initialize components
        level, trend, seasonal = self._initialize(y)

        # Run through the series updating components
        for t in range(n):
            y_t = y[t]
            s_idx = t % m  # seasonal index

            # Previous values
            level_prev = level
            trend_prev = trend if self.trend != "none" else 0.0
            seasonal_t = seasonal[s_idx] if self.seasonal == "add" else 0.0

            # Update level
            if self.seasonal == "add":
                level = self.alpha * (y_t - seasonal_t) + (1 - self.alpha) * (
                    level_prev + trend_prev
                )
            else:
                level = self.alpha * y_t + (1 - self.alpha) * (level_prev + trend_prev)

            # Update trend
            if self.trend == "add":
                trend = self.beta * (level - level_prev) + (1 - self.beta) * trend_prev
            elif self.trend == "damped":
                trend = (
                    self.beta * (level - level_prev)
                    + (1 - self.beta) * self.phi * trend_prev
                )
            # else trend stays 0

            # Update seasonal
            if self.seasonal == "add":
                seasonal[s_idx] = (
                    self.gamma * (y_t - level) + (1 - self.gamma) * seasonal_t
                )

        self.level_ = level
        self.trend_ = trend
        self.seasonal_ = seasonal.copy()
        self.fitted_ = True
        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast h steps ahead.

        Parameters
        ----------
        h : int
            Forecast horizon.

        Returns
        -------
        preds : ndarray of shape (h,)
            Forecasted values.

        Raises
        ------
        RuntimeError
            If called before `fit()`.
        """
        if not self.fitted_:
            raise RuntimeError("Fit the model before calling predict().")

        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        m = self.seasonal_period
        preds = np.empty(h, dtype=float)

        for i in range(h):
            step = i + 1
            s_idx = i % m

            # Compute trend contribution
            if self.trend == "add":
                trend_contrib = self.trend_ * step
            elif self.trend == "damped":
                # Sum of phi^1 + phi^2 + ... + phi^step
                trend_contrib = self.trend_ * self.phi * (1 - self.phi**step) / (
                    1 - self.phi + 1e-10
                )
            else:
                trend_contrib = 0.0

            # Compute seasonal contribution
            if self.seasonal == "add":
                seasonal_contrib = self.seasonal_[s_idx]
            else:
                seasonal_contrib = 0.0

            preds[i] = self.level_ + trend_contrib + seasonal_contrib

        return preds


# ================= AutoHoltWinters =================
class AutoHoltWinters:
    """
    Automatic tuner for HoltWintersForecaster.

    Searches over trend types, seasonal types, and smoothing parameters
    to find the best configuration using validation performance.

    Parameters
    ----------
    seasonal_periods : iterable of int, default=(1, 7, 12, 24)
        Candidate seasonal periods.
    trend_options : iterable of str, default=("add", "none", "damped")
        Candidate trend types.
    seasonal_options : iterable of str, default=("add", "none")
        Candidate seasonal types.
    alpha_grid : iterable of float, default=(0.1, 0.3, 0.5, 0.7)
        Candidate alpha values.
    beta_grid : iterable of float, default=(0.0, 0.1, 0.3)
        Candidate beta values.
    gamma_grid : iterable of float, default=(0.0, 0.1, 0.3)
        Candidate gamma values.
    phi_grid : iterable of float, default=(0.9, 0.95, 0.98)
        Candidate phi values for damped trend.
    metric : {"mae", "rmse"}, default="mae"
        Validation error metric.

    Attributes
    ----------
    model_ : HoltWintersForecaster or None
        Best fitted forecaster.
    best_ : dict or None
        Dictionary with best configuration and validation score.
    """

    def __init__(
        self,
        seasonal_periods: Sequence[int] = (1, 7, 12, 24),
        trend_options: Sequence[str] = ("add", "none", "damped"),
        seasonal_options: Sequence[str] = ("add", "none"),
        alpha_grid: Sequence[float] = (0.1, 0.3, 0.5, 0.7),
        beta_grid: Sequence[float] = (0.0, 0.1, 0.3),
        gamma_grid: Sequence[float] = (0.0, 0.1, 0.3),
        phi_grid: Sequence[float] = (0.9, 0.95, 0.98),
        metric: str = "mae",
    ) -> None:
        self.seasonal_periods = list(seasonal_periods)
        self.trend_options = list(trend_options)
        self.seasonal_options = list(seasonal_options)
        self.alpha_grid = list(alpha_grid)
        self.beta_grid = list(beta_grid)
        self.gamma_grid = list(gamma_grid)
        self.phi_grid = list(phi_grid)
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'")
        self.model_: Optional[HoltWintersForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: ArrayLike, val_fraction: float = 0.25) -> "AutoHoltWinters":
        """
        Grid search tuner for HoltWintersForecaster.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time series data.
        val_fraction : float, default=0.25
            Fraction of series reserved for validation.

        Returns
        -------
        self : AutoHoltWinters
            Instance with best model fitted on full dataset.

        Raises
        ------
        ValueError
            If insufficient data for validation split.
        RuntimeError
            If no valid configuration is found.
        """
        y = np.asarray(y, dtype=float)
        N = len(y)
        n_val = max(8, int(N * float(val_fraction)))
        if N - n_val < 2:
            raise ValueError("Not enough data for validation split.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if self.metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for sp in self.seasonal_periods:
            for trend in self.trend_options:
                for seasonal in self.seasonal_options:
                    # Skip invalid combinations
                    if seasonal == "add" and sp > len(y_train):
                        continue

                    for alpha in self.alpha_grid:
                        # Simplify grid search: only iterate beta if trend is used
                        beta_iter = self.beta_grid if trend != "none" else [0.0]
                        for beta in beta_iter:
                            # Only iterate gamma if seasonal is used
                            gamma_iter = (
                                self.gamma_grid if seasonal != "none" else [0.0]
                            )
                            for gamma in gamma_iter:
                                # Only iterate phi if trend is damped
                                phi_iter = (
                                    self.phi_grid if trend == "damped" else [0.98]
                                )
                                for phi in phi_iter:
                                    try:
                                        model = HoltWintersForecaster(
                                            seasonal_period=sp,
                                            trend=trend,
                                            seasonal=seasonal,
                                            alpha=alpha,
                                            beta=beta,
                                            gamma=gamma,
                                            phi=phi,
                                        ).fit(y_train)
                                    except Exception:
                                        continue

                                    # Multi-step validation (single fit)
                                    preds = model.predict(len(y_val))
                                    score = score_fn(y_val, preds)

                                    if score < best_score:
                                        best_score = score
                                        best_conf = {
                                            "seasonal_period": sp,
                                            "trend": trend,
                                            "seasonal": seasonal,
                                            "alpha": alpha,
                                            "beta": beta,
                                            "gamma": gamma,
                                            "phi": phi,
                                        }

        if best_conf is None:
            raise RuntimeError("AutoHoltWinters failed to find a valid configuration.")

        # Refit best model on full data
        self.model_ = HoltWintersForecaster(**best_conf).fit(y)
        self.best_ = {
            "config": best_conf,
            "val_score": best_score,
            "metric": self.metric,
        }
        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast using the best-found HoltWintersForecaster.

        Parameters
        ----------
        h : int
            Forecast horizon.

        Returns
        -------
        preds : ndarray of shape (h,)
            Forecasted values.

        Raises
        ------
        RuntimeError
            If called before `fit()`.
        """
        if self.model_ is None:
            raise RuntimeError("AutoHoltWinters is not fitted yet.")
        return self.model_.predict(h)
