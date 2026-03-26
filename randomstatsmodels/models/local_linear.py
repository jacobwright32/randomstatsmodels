import numpy as np
from typing import Optional, Union, Sequence, Dict

from ..metrics import mae, rmse

ArrayLike = Union[Sequence[float], np.ndarray]


# ================= LocalLinearForecaster =================
class LocalLinearForecaster:
    """
    Local Linear (Weighted Least Squares) Forecaster.

    Fits a polynomial trend using weighted least squares with exponential
    decay for older observations. Combines trend extrapolation with
    recency focus.

    Parameters
    ----------
    decay : float, default=0.95
        Exponential decay factor for weights. Recent observations get higher
        weight. Use 1.0 for uniform weights (standard OLS).
    degree : int, default=1
        Polynomial degree (1=linear, 2=quadratic).
    seasonal_period : int or None, default=None
        If specified, adds seasonal dummy variables.

    Attributes
    ----------
    coef_ : ndarray
        Fitted polynomial coefficients.
    seasonal_coef_ : ndarray or None
        Fitted seasonal coefficients (if seasonal_period is specified).
    n_ : int
        Length of fitted series.
    """

    def __init__(
        self,
        decay: float = 0.95,
        degree: int = 1,
        seasonal_period: Optional[int] = None,
    ) -> None:
        self.decay = float(decay)
        self.degree = int(degree)
        self.seasonal_period = (
            None if seasonal_period is None else int(seasonal_period)
        )

        if not 0 < self.decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        if self.degree < 1:
            raise ValueError("degree must be at least 1")

        self.coef_: Optional[np.ndarray] = None
        self.seasonal_coef_: Optional[np.ndarray] = None
        self.n_: Optional[int] = None

    def _build_design_matrix(self, t: np.ndarray) -> np.ndarray:
        """Build design matrix with polynomial and optional seasonal terms."""
        n = len(t)
        cols = []

        # Polynomial terms: 1, t, t^2, ...
        for d in range(self.degree + 1):
            cols.append(t**d)

        # Seasonal dummies (if specified)
        if self.seasonal_period is not None and self.seasonal_period > 1:
            for s in range(self.seasonal_period - 1):
                dummy = (t.astype(int) % self.seasonal_period == s).astype(float)
                cols.append(dummy)

        return np.column_stack(cols)

    def fit(self, y: ArrayLike) -> "LocalLinearForecaster":
        """
        Fit the local linear model using weighted least squares.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Training time series.

        Returns
        -------
        self : LocalLinearForecaster
            Fitted instance.

        Raises
        ------
        ValueError
            If series is too short.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)

        min_len = self.degree + 2
        if self.seasonal_period is not None:
            min_len = max(min_len, self.seasonal_period + 1)

        if n < min_len:
            raise ValueError(f"Need at least {min_len} data points.")

        self.n_ = n
        t = np.arange(n, dtype=float)

        # Build design matrix
        X = self._build_design_matrix(t)

        # Compute exponential weights: w_i = decay^(n-1-i)
        # Most recent observation (i=n-1) has weight 1, oldest has weight decay^(n-1)
        weights = self.decay ** (n - 1 - t)

        # Weighted least squares: (X^T W X)^-1 X^T W y
        W = np.diag(weights)
        XtW = X.T @ W
        XtWX = XtW @ X
        XtWy = XtW @ y

        # Solve with regularization for numerical stability
        try:
            self.coef_ = np.linalg.solve(
                XtWX + 1e-8 * np.eye(XtWX.shape[0]), XtWy
            )
        except np.linalg.LinAlgError:
            # Fallback to lstsq
            self.coef_, *_ = np.linalg.lstsq(X * weights[:, np.newaxis], y * weights, rcond=None)

        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast h steps ahead by extrapolating the fitted trend.

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
        if self.coef_ is None:
            raise RuntimeError("Fit the model before calling predict().")

        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        # Future time indices
        t_future = np.arange(self.n_, self.n_ + h, dtype=float)

        # Build design matrix for future
        X_future = self._build_design_matrix(t_future)

        return X_future @ self.coef_


# ================= AutoLocalLinear =================
class AutoLocalLinear:
    """
    Automatic tuner for LocalLinearForecaster.

    Searches over decay rates, polynomial degrees, and seasonal periods
    to find the best configuration using validation performance.

    Parameters
    ----------
    decay_grid : iterable of float, default=(0.9, 0.95, 0.98, 1.0)
        Candidate decay rates.
    degree_grid : iterable of int, default=(1, 2)
        Candidate polynomial degrees.
    seasonal_periods : iterable of int or None, default=(None, 7, 12, 24)
        Candidate seasonal periods.
    metric : {"mae", "rmse"}, default="mae"
        Validation error metric.

    Attributes
    ----------
    model_ : LocalLinearForecaster or None
        Best fitted forecaster.
    best_ : dict or None
        Dictionary with best configuration and validation score.
    """

    def __init__(
        self,
        decay_grid: Sequence[float] = (0.9, 0.95, 0.98, 1.0),
        degree_grid: Sequence[int] = (1, 2),
        seasonal_periods: Sequence[Optional[int]] = (None, 7, 12, 24),
        metric: str = "mae",
    ) -> None:
        self.decay_grid = list(decay_grid)
        self.degree_grid = list(degree_grid)
        self.seasonal_periods = list(seasonal_periods)
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'")
        self.model_: Optional[LocalLinearForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: ArrayLike, val_fraction: float = 0.25) -> "AutoLocalLinear":
        """
        Grid search tuner for LocalLinearForecaster.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time series data.
        val_fraction : float, default=0.25
            Fraction of series reserved for validation.

        Returns
        -------
        self : AutoLocalLinear
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
        if N - n_val < 3:
            raise ValueError("Not enough data for validation split.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if self.metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for decay in self.decay_grid:
            for degree in self.degree_grid:
                for sp in self.seasonal_periods:
                    # Skip if seasonal period is too large for training data
                    if sp is not None and sp >= len(y_train):
                        continue

                    try:
                        model = LocalLinearForecaster(
                            decay=decay,
                            degree=degree,
                            seasonal_period=sp,
                        ).fit(y_train)
                    except Exception:
                        continue

                    # Multi-step validation (single fit)
                    preds = model.predict(len(y_val))
                    score = score_fn(y_val, preds)

                    if score < best_score:
                        best_score = score
                        best_conf = {
                            "decay": decay,
                            "degree": degree,
                            "seasonal_period": sp,
                        }

        if best_conf is None:
            raise RuntimeError("AutoLocalLinear failed to find a valid configuration.")

        # Refit best model on full data
        self.model_ = LocalLinearForecaster(**best_conf).fit(y)
        self.best_ = {
            "config": best_conf,
            "val_score": best_score,
            "metric": self.metric,
        }
        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast using the best-found LocalLinearForecaster.

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
            raise RuntimeError("AutoLocalLinear is not fitted yet.")
        return self.model_.predict(h)
