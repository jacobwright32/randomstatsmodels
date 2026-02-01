import numpy as np
from typing import Optional, Union, Sequence, Dict

from ..metrics import mae, rmse

ArrayLike = Union[Sequence[float], np.ndarray]


# ================= NaiveForecaster =================
class NaiveForecaster:
    """
    Naive baseline forecaster with multiple strategies.

    Provides essential baselines for proper model evaluation:
    - "last": repeat the last observed value
    - "seasonal": repeat values from one seasonal period ago
    - "drift": linear extrapolation from first to last value
    - "mean": rolling or global mean

    Parameters
    ----------
    method : {"last", "seasonal", "drift", "mean"}, default="last"
        Forecasting method to use.
    seasonal_period : int, default=1
        Seasonal period for "seasonal" method.
    window : int or None, default=None
        Window size for rolling mean. If None, uses global mean.

    Attributes
    ----------
    data_ : ndarray
        Stored training series.
    last_value_ : float
        Last value of the training series.
    drift_slope_ : float
        Drift slope (for drift method).
    mean_ : float
        Mean value (for mean method).
    """

    def __init__(
        self,
        method: str = "last",
        seasonal_period: int = 1,
        window: Optional[int] = None,
    ) -> None:
        self.method = method.lower()
        if self.method not in ("last", "seasonal", "drift", "mean"):
            raise ValueError("method must be 'last', 'seasonal', 'drift', or 'mean'")
        self.seasonal_period = int(seasonal_period)
        self.window = window if window is None else int(window)

        self.data_: Optional[np.ndarray] = None
        self.last_value_: Optional[float] = None
        self.drift_slope_: Optional[float] = None
        self.mean_: Optional[float] = None
        self.n_: Optional[int] = None

    def fit(self, y: ArrayLike) -> "NaiveForecaster":
        """
        Fit the naive forecaster by storing necessary statistics.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Training time series.

        Returns
        -------
        self : NaiveForecaster
            Fitted instance.

        Raises
        ------
        ValueError
            If series is too short for the selected method.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)

        if n < 1:
            raise ValueError("Need at least 1 data point.")

        if self.method == "seasonal" and n < self.seasonal_period:
            raise ValueError(
                f"Need at least {self.seasonal_period} points for seasonal method."
            )

        self.data_ = y.copy()
        self.n_ = n
        self.last_value_ = float(y[-1])

        # Compute drift slope: (y_n - y_1) / (n - 1)
        if n > 1:
            self.drift_slope_ = (y[-1] - y[0]) / (n - 1)
        else:
            self.drift_slope_ = 0.0

        # Compute mean
        if self.window is not None and self.window > 0:
            # Rolling mean of last 'window' values
            self.mean_ = float(np.mean(y[-min(self.window, n) :]))
        else:
            # Global mean
            self.mean_ = float(np.mean(y))

        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast h steps ahead using the fitted naive method.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps ahead).

        Returns
        -------
        preds : ndarray of shape (h,)
            Forecasted values.

        Raises
        ------
        RuntimeError
            If called before `fit()`.
        """
        if self.data_ is None:
            raise RuntimeError("Fit the model before calling predict().")

        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        if self.method == "last":
            # Repeat the last value
            return np.full(h, self.last_value_, dtype=float)

        elif self.method == "seasonal":
            # Repeat values from one seasonal period ago
            preds = np.empty(h, dtype=float)
            for i in range(h):
                # Index into the last seasonal_period values
                idx = (self.n_ - self.seasonal_period + i) % self.seasonal_period
                # Get from the last seasonal_period values of training data
                preds[i] = self.data_[-(self.seasonal_period - idx % self.seasonal_period)]
            # More direct approach: cycle through last seasonal_period values
            last_season = self.data_[-self.seasonal_period :]
            preds = np.array([last_season[i % self.seasonal_period] for i in range(h)])
            return preds

        elif self.method == "drift":
            # Linear extrapolation: y_n + slope * (1, 2, ..., h)
            steps = np.arange(1, h + 1)
            return self.last_value_ + self.drift_slope_ * steps

        elif self.method == "mean":
            # Repeat the mean value
            return np.full(h, self.mean_, dtype=float)

        else:
            raise ValueError(f"Unknown method: {self.method}")


# ================= AutoNaive =================
class AutoNaive:
    """
    Automatic tuner for NaiveForecaster.

    Searches over naive methods and their parameters to find the
    best baseline model using validation performance.

    Parameters
    ----------
    method_options : iterable of str, default=("last", "seasonal", "drift", "mean")
        Candidate forecasting methods.
    seasonal_periods : iterable of int, default=(1, 7, 12, 24)
        Candidate seasonal periods (used when method="seasonal").
    window_options : iterable of int or None, default=(None, 5, 10, 20)
        Candidate window sizes for rolling mean.
    metric : {"mae", "rmse"}, default="mae"
        Validation error metric.

    Attributes
    ----------
    model_ : NaiveForecaster or None
        Best fitted forecaster.
    best_ : dict or None
        Dictionary with best configuration and validation score.
    """

    def __init__(
        self,
        method_options: Sequence[str] = ("last", "seasonal", "drift", "mean"),
        seasonal_periods: Sequence[int] = (1, 7, 12, 24),
        window_options: Sequence[Optional[int]] = (None, 5, 10, 20),
        metric: str = "mae",
    ) -> None:
        self.method_options = list(method_options)
        self.seasonal_periods = list(seasonal_periods)
        self.window_options = list(window_options)
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'")
        self.model_: Optional[NaiveForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: ArrayLike, val_fraction: float = 0.25) -> "AutoNaive":
        """
        Grid search tuner for NaiveForecaster.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time series data.
        val_fraction : float, default=0.25
            Fraction of series reserved for validation.

        Returns
        -------
        self : AutoNaive
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
        if N - n_val < 1:
            raise ValueError("Not enough data for validation split.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if self.metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for method in self.method_options:
            # Determine which parameters to iterate over
            if method == "seasonal":
                param_iter = [(sp, None) for sp in self.seasonal_periods]
            elif method == "mean":
                param_iter = [(1, w) for w in self.window_options]
            else:
                param_iter = [(1, None)]  # method-only, no extra params

            for seasonal_period, window in param_iter:
                try:
                    model = NaiveForecaster(
                        method=method,
                        seasonal_period=seasonal_period,
                        window=window,
                    ).fit(y_train)
                except Exception:
                    continue

                # One-step rolling forecast through validation
                preds = []
                current_data = y_train.copy()
                for t in range(split, N):
                    # Refit on current data for rolling update
                    model = NaiveForecaster(
                        method=method,
                        seasonal_period=seasonal_period,
                        window=window,
                    ).fit(current_data)
                    yhat = model.predict(1)[0]
                    preds.append(yhat)
                    current_data = np.append(current_data, y[t])

                preds = np.array(preds)
                score = score_fn(y_val, preds)

                if score < best_score:
                    best_score = score
                    best_conf = {
                        "method": method,
                        "seasonal_period": seasonal_period,
                        "window": window,
                    }

        if best_conf is None:
            raise RuntimeError("AutoNaive failed to find a valid configuration.")

        # Refit best model on full data
        self.model_ = NaiveForecaster(**best_conf).fit(y)
        self.best_ = {
            "config": best_conf,
            "val_score": best_score,
            "metric": self.metric,
        }
        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast using the best-found NaiveForecaster.

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
            raise RuntimeError("AutoNaive is not fitted yet.")
        return self.model_.predict(h)
