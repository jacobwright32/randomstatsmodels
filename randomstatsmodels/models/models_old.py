from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Iterable, List

import numpy as np

from .model_utils import _weighted_quantile, _penalty_value, _golden_section_minimize
from ..metrics import mae, rmse, smape, mape


class SeasonalARForecaster:
    """
    Seasonal Autoregressive Forecaster with polynomial lag features and Fourier seasonality.

    This model fits an additive regression using recent lags (and their nonlinear products)
    plus sine/cosine seasonal terms. It supports multiple seasonal periods and (optionally)
    uses a rolling window of recent data.
    """

    def __init__(
        self,
        lags=5,
        degree=2,
        seasonal_periods: Optional[list] = None,
        fourier_orders: Optional[Dict[int, int]] = None,
        window_size: Optional[int] = None,
    ):
        self.lags = int(lags)
        self.degree = int(degree)
        # Seasonal periods (in number of timesteps) to include (e.g. [7, 365]); default none
        self.seasonal_periods = list(seasonal_periods) if seasonal_periods is not None else []
        # Fourier terms: mapping period -> number of harmonics to include (each adds sin/cos)
        self.fourier_orders = dict(fourier_orders) if fourier_orders is not None else {}
        # Ensure default one harmonic if period given without explicit order
        for p in self.seasonal_periods:
            self.fourier_orders.setdefault(p, 1)
        self.window_size = window_size
        # Fitted model coefficients
        self.coef_ = None
        self.last_window_ = None  # stores the final window of training data
        self.last_index_ = None  # index of the last training point

    def _features(self, state: np.ndarray, t_index: int) -> np.ndarray:
        """
        Build feature vector for a given state (recent lags) and time index for seasonality.
        `state` is a length-m array [x_{t}, x_{t-1}, ..., x_{t-m+1}] (newest first).
        `t_index` is the current time index (0-based) corresponding to x_t.
        """
        feats = [1.0]  # intercept
        m = len(state)
        # Linear terms (lags)
        feats.extend(state.tolist())
        # Higher-order (polynomial) interaction terms
        if self.degree >= 2:
            for i in range(m):
                for j in range(i, m):
                    feats.append(state[i] * state[j])
        if self.degree >= 3:
            for i in range(m):
                for j in range(i, m):
                    for k in range(j, m):
                        feats.append(state[i] * state[j] * state[k])
        # Fourier seasonal features for each specified period
        for period in self.seasonal_periods:
            M = self.fourier_orders.get(period, 1)
            for k in range(1, M + 1):
                angle = 2 * np.pi * k * t_index / period
                feats.append(np.sin(angle))
                feats.append(np.cos(angle))
        return np.array(feats, dtype=float)

    def fit(self, y: np.ndarray):
        """
        Fit the SeasonalARForecaster to the univariate series y.
        Splits a hold-out window if window_size is set, then solves linear regression.
        """
        y = np.asarray(y, float)
        N = len(y)
        m = self.lags
        if N < m + 1:
            raise ValueError(f"Need at least {m+1} points, got {N}.")

        # Determine training start index if using a rolling window
        start = 0
        if self.window_size is not None and self.window_size < N:
            start = N - self.window_size
        # Ensure we have at least lags-1 before first target
        start = max(start, m - 1)

        X_rows = []
        targets = []
        # Construct training design matrix
        for t in range(start, N - 1):
            # state = [y_t, y_{t-1}, ..., y_{t-m+1}]
            state = y[t - m + 1 : t + 1][::-1]
            feats = self._features(state, t)  # include time index for seasonality
            X_rows.append(feats)
            targets.append(y[t + 1])

        X = np.vstack(X_rows)
        Y = np.array(targets, dtype=float)
        # Solve for coefficients (least squares)
        w, *_ = np.linalg.lstsq(X, Y, rcond=None)
        self.coef_ = w
        self.last_window_ = y[-m:].copy()
        self.last_index_ = N - 1
        return self

    def predict(self, h: int, start_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate out-of-sample forecasts for horizon h.
        If start_values are given, use them as the initial state; otherwise use last training window.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit the model before predicting.")
        m = self.lags
        if start_values is None:
            if self.last_window_ is None or self.last_index_ is None:
                raise RuntimeError("No starting values available for prediction.")
            window = self.last_window_.copy()
            curr_index = self.last_index_
        else:
            start_values = np.asarray(start_values, float)
            if len(start_values) != m:
                raise ValueError(f"start_values must have length {m}.")
            window = start_values.copy()
            # Assume continuation from last training index
            curr_index = self.last_index_

        forecasts = []
        for _ in range(h):
            # Advance time index
            next_index = curr_index + 1
            # Compute features for this step
            phi = self._features(window[::-1], next_index)
            yhat = float(np.dot(self.coef_, phi))
            forecasts.append(yhat)
            # Slide window
            window[:-1] = window[1:]
            window[-1] = yhat
            curr_index = next_index

        return np.array(forecasts, dtype=float)


class AutoSeasonalAR:
    """
    Automatic tuner for SeasonalARForecaster. Performs grid search over lags, degree,
    seasonal periods, Fourier orders, and window sizes using a train/validation split.
    """

    def __init__(
        self,
        lags_grid=(4, 8, 12),
        degree_grid=(1, 2, 3),
        seasonal_periods_grid: Optional[List[list]] = (None, [7], [365], [7, 365]),
        fourier_orders=(1, 2, 3),
        window_fracs=(None, 0.5, 0.75),
    ):
        self.lags_grid = lags_grid
        self.degree_grid = degree_grid
        # seasonal_periods_grid is a list of lists (or None) to try
        self.seasonal_periods_grid = seasonal_periods_grid
        self.fourier_orders = fourier_orders
        self.window_fracs = window_fracs
        self.model_: Optional[SeasonalARForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction=0.25, metric="mae"):
        """
        Fit the auto-tuner on a univariate series y.
        Splits into training and validation blocks, evaluates each candidate,
        and picks the best configuration (minimizing MAE or RMSE).
        """
        y = np.asarray(y, float)
        N = len(y)
        n_val = max(16, int(N * val_fraction))
        split = N - n_val
        y_train = y[:split]
        y_val = y[split:]

        best_score = np.inf
        best_conf = None
        best_model = None

        # Iterate over hyperparameter grid
        for lags in self.lags_grid:
            for degree in self.degree_grid:
                for sp in self.seasonal_periods_grid:
                    # Normalize seasonal setting
                    if sp is None:
                        seasonal_periods = []
                    else:
                        seasonal_periods = list(sp)
                    for fo in self.fourier_orders:
                        # Map each period to the same Fourier order
                        fourier_orders = {p: fo for p in seasonal_periods}
                        for wfrac in self.window_fracs:
                            window_size = None
                            if wfrac is not None:
                                window_size = max(lags + 8, int(len(y_train) * float(wfrac)))
                            try:
                                model = SeasonalARForecaster(
                                    lags=lags,
                                    degree=degree,
                                    seasonal_periods=seasonal_periods,
                                    fourier_orders=fourier_orders,
                                    window_size=window_size,
                                )
                                model.fit(y_train)
                                # Forecast validation horizon
                                start_vals = y_train[-lags:]
                                preds = model.predict(len(y_val), start_values=start_vals)
                                score = mae(y_val, preds) if metric == "mae" else rmse(y_val, preds)
                                if score < best_score:
                                    best_score = score
                                    best_model = model
                                    best_conf = dict(
                                        lags=lags,
                                        degree=degree,
                                        seasonal_periods=seasonal_periods,
                                        fourier_orders=fourier_orders,
                                        window_size=window_size,
                                    )
                            except Exception:
                                # Skip invalid combinations quietly
                                continue

        # Refit best model on full series
        if best_conf is None:
            raise RuntimeError("No valid model configuration found.")
        final_model = SeasonalARForecaster(**best_conf)
        final_model.fit(y)
        self.model_ = final_model
        self.best_ = {"config": best_conf, "val_score": best_score}
        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Generate forecasts using the best-fitted SeasonalARForecaster.
        """
        if self.model_ is None:
            raise RuntimeError("AutoSeasonalAR not fitted yet.")
        return self.model_.predict(h)


class WindowAverageForecaster:
    """
    Rolling (Moving) Average Forecaster.
    Predicts future values as the average of the most recent `window` observations.
    """

    def __init__(self, window=3):
        """
        Parameters
        ----------
        window : int
            Number of recent points to average for forecasting.
        """
        self.window = int(window)
        self.data = None  # will hold the fitted time series

    def fit(self, y):
        """
        Store the time series.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < self.window:
            raise ValueError(f"Need at least window={self.window} points, got {n}.")
        self.data = y.copy()
        return self

    def predict(self, h, start_values=None):
        """
        Forecast `h` future steps using the moving average.

        Parameters
        ----------
        h : int
            Number of future points to forecast.
        start_values : array-like (optional)
            Starting window of length `self.window`. If None, uses the last
            `self.window` points from the fitted data.

        Returns
        -------
        preds : np.ndarray
            Array of length h with the forecasted values.
        """
        if self.data is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        # Initialize the window for forecasting
        if start_values is None:
            current_window = self.data[-self.window :].astype(float).copy()
        else:
            start_values = np.asarray(start_values, dtype=float)
            if len(start_values) != self.window:
                raise ValueError(f"start_values must have length {self.window}.")
            current_window = start_values.copy()

        preds = []
        for _ in range(h):
            # Predict as mean of current window
            yhat = float(np.mean(current_window))
            preds.append(yhat)
            # Update the window: drop oldest, append forecast
            current_window[:-1] = current_window[1:]
            current_window[-1] = yhat
        return np.array(preds)


class AutoWindow:
    """
    Automatic tuner for WindowAverageForecaster window length.
    """

    def __init__(self, window_grid=(3, 5, 7, 14, 24), metric="mae"):
        """
        Parameters
        ----------
        window_grid : iterable of int
            Candidate values for the moving average window size.
        metric : str, "mae" or "rmse"
            Error metric to minimize on validation set.
        """
        self.window_grid = list(window_grid)
        self.metric = metric.lower()
        self.model_ = None  # final fitted WindowAverageForecaster
        self.best_ = None  # dict with best config and validation score

    def fit(self, y, val_fraction=0.25):
        """
        Tune window size on validation set and fit final model on full data.

        Parameters
        ----------
        y : array-like
            The full time series to fit.
        val_fraction : float
            Fraction of data to reserve for validation (at least 16 points).

        Returns
        -------
        self : AutoWindow
            Fitted auto model with attributes `model_` and `best_`.
        """
        y = np.asarray(y, dtype=float)
        N = len(y)
        n_val = max(16, int(N * float(val_fraction)))
        if N - n_val < self.window_grid[0]:
            raise ValueError("Not enough data for the smallest window size.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        best_score = np.inf
        best_window = None

        # Define error functions
        def mae(a, b):
            return np.mean(np.abs(a - b))

        def rmse(a, b):
            return np.sqrt(np.mean((a - b) ** 2))

        # Grid search over window sizes
        for w in self.window_grid:
            if len(y_train) < w:
                continue  # skip if training too short
            model = WindowAverageForecaster(window=w).fit(y_train)
            preds = []
            # Rolling one-step forecasts on validation
            for t in range(split, N):
                yhat = model.predict(1)[0]
                preds.append(yhat)
                # Update model with the actual next value
                model.data = np.append(model.data, y[t])
            preds = np.array(preds)
            score = mae(y_val, preds) if self.metric == "mae" else rmse(y_val, preds)
            if score < best_score:
                best_score = score
                best_window = w

        if best_window is None:
            raise RuntimeError("AutoWindow failed to find a valid configuration.")

        # Refit on full data with the best window
        best_model = WindowAverageForecaster(window=best_window).fit(y)
        self.model_ = best_model
        self.best_ = {"config": {"window": best_window}, "val_score": best_score}
        return self

    def predict(self, h):
        """
        Forecast using the best-found WindowAverageForecaster. Must call fit() first.
        """
        if self.model_ is None:
            raise RuntimeError("AutoWindow is not fitted yet.")
        return self.model_.predict(h)


# ================= RollingMedianForecaster =================
class RollingMedianForecaster:
    """
    Rolling Median Forecaster.
    Predicts future values as the median of the most recent `window` observations.
    Robust to outliers compared to a moving average.
    """

    def __init__(self, window=5):
        """
        Parameters
        ----------
        window : int
            Number of recent points to take median over for forecasting.
        """
        self.window = int(window)
        self.data = None  # stores fitted series

    def fit(self, y):
        """
        Store the time series.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < self.window:
            raise ValueError(f"Need at least window={self.window} points, got {n}.")
        self.data = y.copy()
        return self

    def predict(self, h, start_values=None):
        """
        Iteratively predict h steps ahead using rolling median.

        Parameters
        ----------
        h : int
            Forecast horizon.
        start_values : array-like (optional)
            Starting window of length = self.window. If None, uses last window points.

        Returns
        -------
        preds : np.ndarray of shape (h,)
        """
        if self.data is None:
            raise RuntimeError("Fit the model before calling predict().")
        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        # Initialize window
        if start_values is None:
            current_window = self.data[-self.window :].astype(float).copy()
        else:
            start_values = np.asarray(start_values, dtype=float)
            if len(start_values) != self.window:
                raise ValueError(f"start_values must have length {self.window}.")
            current_window = start_values.copy()

        preds = []
        for _ in range(h):
            yhat = float(np.median(current_window))
            preds.append(yhat)
            # roll window forward with forecast
            current_window[:-1] = current_window[1:]
            current_window[-1] = yhat
        return np.array(preds)


# ================= AutoRollingMedian =================
class AutoRollingMedian:
    """
    Automatic tuner for RollingMedianForecaster window length.
    Searches over window_grid and selects best via validation error.
    """

    def __init__(self, window_grid=(3, 5, 7, nine := 9), metric="mae"):
        """
        Parameters
        ----------
        window_grid : iterable of int
            Candidate window sizes to try (odd sizes often work well for medians).
        metric : {"mae","rmse"}
            Validation metric to minimize.
        """
        # allow numbers OR 'nine' walrus; coerce to list of ints
        self.window_grid = [int(w) for w in window_grid]
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'")
        self.model_ = None
        self.best_ = None

    def fit(self, y, val_fraction=0.25):
        """
        Tune window using a hold-out tail set via rolling one-step evaluation,
        then refit best model on full data.
        """
        y = np.asarray(y, dtype=float)
        N = len(y)
        n_val = max(16, int(N * float(val_fraction)))
        if N - n_val < min(self.window_grid):
            raise ValueError("Not enough data for validation given smallest window.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        def mae(a, b):
            return float(np.mean(np.abs(a - b)))

        def rmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)))

        score_fn = mae if self.metric == "mae" else rmse

        best_score = np.inf
        best_w = None

        for w in self.window_grid:
            if len(y_train) < w:
                continue
            model = RollingMedianForecaster(window=w).fit(y_train)
            preds = []
            # rolling one-step through validation, updating with truths
            for t in range(split, N):
                preds.append(model.predict(1)[0])
                model.data = np.append(model.data, y[t])
            preds = np.asarray(preds)
            score = score_fn(y_val, preds)
            if score < best_score:
                best_score = score
                best_w = w

        if best_w is None:
            raise RuntimeError("AutoRollingMedian failed to find a valid configuration.")

        self.model_ = RollingMedianForecaster(window=best_w).fit(y)
        self.best_ = {
            "config": {"window": best_w},
            "val_score": best_score,
            "metric": self.metric,
        }
        return self

    def predict(self, h):
        """
        Forecast using the best-found RollingMedianForecaster.
        """
        if self.model_ is None:
            raise RuntimeError("AutoRollingMedian is not fitted yet.")
        return self.model_.predict(h)


# ================= RollingMedianForecaster =================
class RollingMedianForecaster:
    """
    Rolling Median Forecaster.
    Predicts future values as the median of the most recent `window` observations.
    Robust to outliers compared to a moving average.
    """

    def __init__(self, window=5):
        """
        Parameters
        ----------
        window : int
            Number of recent points to take median over for forecasting.
        """
        self.window = int(window)
        self.data = None  # stores fitted series

    def fit(self, y):
        """
        Store the time series.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < self.window:
            raise ValueError(f"Need at least window={self.window} points, got {n}.")
        self.data = y.copy()
        return self

    def predict(self, h, start_values=None):
        """
        Iteratively predict h steps ahead using rolling median.

        Parameters
        ----------
        h : int
            Forecast horizon.
        start_values : array-like (optional)
            Starting window of length = self.window. If None, uses last window points.

        Returns
        -------
        preds : np.ndarray of shape (h,)
        """
        if self.data is None:
            raise RuntimeError("Fit the model before calling predict().")
        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        # Initialize window
        if start_values is None:
            current_window = self.data[-self.window :].astype(float).copy()
        else:
            start_values = np.asarray(start_values, dtype=float)
            if len(start_values) != self.window:
                raise ValueError(f"start_values must have length {self.window}.")
            current_window = start_values.copy()

        preds = []
        for _ in range(h):
            yhat = float(np.median(current_window))
            preds.append(yhat)
            # roll window forward with forecast
            current_window[:-1] = current_window[1:]
            current_window[-1] = yhat
        return np.array(preds)


# ================= AutoRollingMedian =================
class AutoRollingMedian:
    """
    Automatic tuner for RollingMedianForecaster window length.
    Searches over window_grid and selects best via validation error.
    """

    def __init__(self, window_grid=(3, 5, 7, nine := 9), metric="mae"):
        """
        Parameters
        ----------
        window_grid : iterable of int
            Candidate window sizes to try (odd sizes often work well for medians).
        metric : {"mae","rmse"}
            Validation metric to minimize.
        """
        # allow numbers OR 'nine' walrus; coerce to list of ints
        self.window_grid = [int(w) for w in window_grid]
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'")
        self.model_ = None
        self.best_ = None

    def fit(self, y, val_fraction=0.25):
        """
        Tune window using a hold-out tail set via rolling one-step evaluation,
        then refit best model on full data.
        """
        y = np.asarray(y, dtype=float)
        N = len(y)
        n_val = max(16, int(N * float(val_fraction)))
        if N - n_val < min(self.window_grid):
            raise ValueError("Not enough data for validation given smallest window.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        def mae(a, b):
            return float(np.mean(np.abs(a - b)))

        def rmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)))

        score_fn = mae if self.metric == "mae" else rmse

        best_score = np.inf
        best_w = None

        for w in self.window_grid:
            if len(y_train) < w:
                continue
            model = RollingMedianForecaster(window=w).fit(y_train)
            preds = []
            # rolling one-step through validation, updating with truths
            for t in range(split, N):
                preds.append(model.predict(1)[0])
                model.data = np.append(model.data, y[t])
            preds = np.asarray(preds)
            score = score_fn(y_val, preds)
            if score < best_score:
                best_score = score
                best_w = w

        if best_w is None:
            raise RuntimeError("AutoRollingMedian failed to find a valid configuration.")

        self.model_ = RollingMedianForecaster(window=best_w).fit(y)
        self.best_ = {
            "config": {"window": best_w},
            "val_score": best_score,
            "metric": self.metric,
        }
        return self

    def predict(self, h):
        """
        Forecast using the best-found RollingMedianForecaster.
        """
        if self.model_ is None:
            raise RuntimeError("AutoRollingMedian is not fitted yet.")
        return self.model_.predict(h)


# ================= TrimmedMeanForecaster =================
class TrimmedMeanForecaster:
    """
    Rolling Trimmed-Mean Forecaster.
    Predicts future values as the mean of the most recent `window` observations
    after trimming a fraction `alpha` from each tail.
    """

    def __init__(self, window=7, alpha=0.2):
        """
        Parameters
        ----------
        window : int
            Number of recent points to aggregate for forecasting.
        alpha : float in [0, 0.5)
            Fraction to trim from each tail (e.g., 0.2 trims lowest 20% and highest 20%).
        """
        self.window = int(window)
        self.alpha = float(alpha)
        if not (0.0 <= self.alpha < 0.5):
            raise ValueError("alpha must be in [0, 0.5).")
        self.data = None

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < self.window:
            raise ValueError(f"Need at least window={self.window} points, got {n}.")
        # ensure at least one value survives trimming
        n_trim_each = int(self.alpha * self.window)
        if self.window - 2 * n_trim_each <= 0:
            raise ValueError("window too small for the chosen alpha (nothing left after trimming).")
        self.data = y.copy()
        return self

    def _trimmed_mean(self, arr):
        arr = np.sort(np.asarray(arr, dtype=float))
        n = len(arr)
        k = int(self.alpha * n)
        core = arr[k : n - k] if (n - 2 * k) > 0 else arr  # safety
        return float(np.mean(core))

    def predict(self, h, start_values=None):
        """
        Iteratively predict h steps ahead using rolling trimmed mean.

        Parameters
        ----------
        h : int
            Forecast horizon.
        start_values : array-like (optional)
            Starting window of length = self.window. If None, uses last window points.

        Returns
        -------
        preds : np.ndarray of shape (h,)
        """
        if self.data is None:
            raise RuntimeError("Fit the model before calling predict().")
        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        # Initialize window
        if start_values is None:
            current_window = self.data[-self.window :].astype(float).copy()
        else:
            start_values = np.asarray(start_values, dtype=float)
            if len(start_values) != self.window:
                raise ValueError(f"start_values must have length {self.window}.")
            current_window = start_values.copy()

        preds = []
        for _ in range(h):
            yhat = self._trimmed_mean(current_window)
            preds.append(yhat)
            # roll window forward with forecast
            current_window[:-1] = current_window[1:]
            current_window[-1] = yhat
        return np.array(preds)


# ================= AutoTrimmedMean =================
class AutoTrimmedMean:
    """
    Automatic tuner for TrimmedMeanForecaster.
    Searches over window_grid and alpha_grid and selects best via validation error.
    """

    def __init__(
        self,
        window_grid=(5, 7, 9, 11, 15),
        alpha_grid=(0.0, 0.1, 0.2, 0.3),
        metric="mae",
    ):
        """
        Parameters
        ----------
        window_grid : iterable of int
            Candidate window sizes to try.
        alpha_grid : iterable of float in [0, 0.5)
            Candidate trim fractions per tail.
        metric : {"mae","rmse"}
            Validation metric to minimize.
        """
        self.window_grid = [int(w) for w in window_grid]
        self.alpha_grid = [float(a) for a in alpha_grid]
        for a in self.alpha_grid:
            if not (0.0 <= a < 0.5):
                raise ValueError("All alpha values must be in [0, 0.5).")
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'")
        self.model_ = None
        self.best_ = None

    def fit(self, y, val_fraction=0.25):
        """
        Tune (window, alpha) using a hold-out tail via rolling one-step evaluation,
        then refit best model on full data.
        """
        y = np.asarray(y, dtype=float)
        N = len(y)
        n_val = max(16, int(N * float(val_fraction)))
        if N - n_val < min(self.window_grid):
            raise ValueError("Not enough data for validation given smallest window.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        def mae(a, b):
            return float(np.mean(np.abs(a - b)))

        def rmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)))

        score_fn = mae if self.metric == "mae" else rmse

        best_score = np.inf
        best_conf = None

        for w in self.window_grid:
            if len(y_train) < w:
                continue
            for a in self.alpha_grid:
                # ensure trimming leaves at least one value
                if w - 2 * int(a * w) <= 0:
                    continue
                try:
                    model = TrimmedMeanForecaster(window=w, alpha=a).fit(y_train)
                except Exception:
                    continue
                preds = []
                # rolling one-step through validation, updating with truths
                for t in range(split, N):
                    preds.append(model.predict(1)[0])
                    model.data = np.append(model.data, y[t])
                preds = np.asarray(preds)
                score = score_fn(y_val, preds)
                if score < best_score:
                    best_score = score
                    best_conf = {"window": w, "alpha": a}

        if best_conf is None:
            raise RuntimeError("AutoTrimmedMean failed to find a valid configuration.")

        self.model_ = TrimmedMeanForecaster(**best_conf).fit(y)
        self.best_ = {
            "config": best_conf,
            "val_score": best_score,
            "metric": self.metric,
        }
        return self

    def predict(self, h):
        """
        Forecast using the best-found TrimmedMeanForecaster.
        """
        if self.model_ is None:
            raise RuntimeError("AutoTrimmedMean is not fitted yet.")
        return self.model_.predict(h)


# ================ RankInsertionForecaster ================
class RankInsertionForecaster:
    """
    Rank-Insertion Forecaster (Order-Statistics Based).

    Learns the empirical distribution of the insertion rank of the next value
    relative to the sorted last `window` observations.

    At prediction time, chooses a target rank (mode or expected) and returns
    the corresponding quantile of the current window.
    """

    def __init__(self, window=8, rank_strategy="mode"):
        """
        Parameters
        ----------
        window : int
            Size of the rolling context window.
        rank_strategy : {"mode","mean"}
            - "mode": use the most frequent insertion rank observed in training.
            - "mean": use the expected (average) insertion rank -> percentile = mean_rank / window.
        """
        self.window = int(window)
        if rank_strategy not in ("mode", "mean"):
            raise ValueError("rank_strategy must be 'mode' or 'mean'.")
        self.rank_strategy = rank_strategy

        self.data = None
        self.hist_ = None  # counts for ranks 0..window
        self.total_ = 0  # total number of rank observations
        self._target_rank_ = None  # chosen rank (float for 'mean', int for 'mode')

    @staticmethod
    def _insertion_rank(window_vals, next_val):
        """
        Return the insertion index k in [0..w] for next_val relative to sorted(window_vals),
        i.e., number of elements <= next_val (ties go to the right).
        """
        w_sorted = np.sort(window_vals)
        # searchsorted with side='right' counts how many values <= next_val
        k = int(np.searchsorted(w_sorted, next_val, side="right"))
        return k  # 0..w inclusive

    def fit(self, y):
        """
        Build the insertion-rank histogram over the training series.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < self.window + 1:
            raise ValueError(f"Need at least window+1={self.window+1} points, got {n}.")

        self.data = y.copy()
        w = self.window
        hist = np.zeros(w + 1, dtype=float)  # ranks 0..w

        # Slide windows and collect insertion ranks
        # For t from w .. n-1: window is y[t-w : t], next is y[t]
        for t in range(w, n):
            win = y[t - w : t]
            nxt = y[t]
            k = self._insertion_rank(win, nxt)
            hist[k] += 1.0

        self.hist_ = hist
        self.total_ = int(hist.sum())
        if self.total_ == 0:
            # degenerate case (shouldn't happen with n>=w+1)
            self._target_rank_ = w / 2.0
        else:
            if self.rank_strategy == "mode":
                # Most frequent rank; tie-break by choosing the middle-most among ties
                maxc = hist.max()
                candidates = np.flatnonzero(hist == maxc)
                self._target_rank_ = int(candidates[len(candidates) // 2])
            else:
                # Expected rank
                ks = np.arange(w + 1, dtype=float)
                self._target_rank_ = float((ks * hist).sum() / hist.sum())

        return self

    def predict(self, h, start_values=None):
        """
        Iteratively predict h steps ahead using the learned target rank mapped
        to a quantile of the rolling window.

        Parameters
        ----------
        h : int
            Forecast horizon.
        start_values : array-like (optional)
            Starting window of length = self.window. If None, use the last window from self.data.
        """
        if self.data is None or self.hist_ is None:
            raise RuntimeError("Fit the model before calling predict().")
        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        w = self.window

        # Initialize the forecasting window
        if start_values is None:
            current_window = self.data[-w:].astype(float).copy()
        else:
            start_values = np.asarray(start_values, dtype=float)
            if len(start_values) != w:
                raise ValueError(f"start_values must have length {w}.")
            current_window = start_values.copy()

        preds = []
        # Map target rank to percentile q in [0,1]
        target_rank = self._target_rank_
        q = float(target_rank) / float(w)  # if 'mode' it’s integer; if 'mean' it can be fractional

        for _ in range(h):
            # Use the q-quantile of the current window as the forecast
            # np.quantile handles interpolation between order stats (within min..max)
            yhat = float(np.quantile(current_window, q))
            preds.append(yhat)
            # Roll window forward by appending forecast
            current_window[:-1] = current_window[1:]
            current_window[-1] = yhat

        return np.asarray(preds)


# ================ AutoRankInsertion ================
class AutoRankInsertion:
    """
    Automatic tuner for RankInsertionForecaster over window size and rank strategy.
    Uses rolling one-step validation to pick the best combo.
    """

    def __init__(self, window_grid=(4, 6, 8, 12), rank_strategies=("mode", "mean"), metric="mae"):
        """
        Parameters
        ----------
        window_grid : iterable of int
            Candidate window sizes.
        rank_strategies : iterable of {"mode","mean"}
            Candidate rank-targeting strategies.
        metric : {"mae","rmse"}
            Validation metric to minimize.
        """
        self.window_grid = [int(w) for w in window_grid]
        self.rank_strategies = list(rank_strategies)
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'.")

        self.model_ = None
        self.best_ = None

    def fit(self, y, val_fraction=0.25):
        """
        Split off a validation tail; for each (window, strategy), fit on train
        and evaluate via rolling one-step forecasts through the validation region.
        """
        y = np.asarray(y, dtype=float)
        N = len(y)
        n_val = max(16, int(N * float(val_fraction)))
        if N - n_val < min(self.window_grid) + 1:
            raise ValueError("Not enough data for validation with the smallest window.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        def mae(a, b):
            return float(np.mean(np.abs(a - b)))

        def rmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)))

        score_fn = mae if self.metric == "mae" else rmse

        best_score = np.inf
        best_conf = None

        for w in self.window_grid:
            if len(y_train) < w + 1:
                continue
            for strat in self.rank_strategies:
                try:
                    model = RankInsertionForecaster(window=w, rank_strategy=strat).fit(y_train)
                except Exception:
                    continue

                preds = []
                # Roll forward one step at a time through validation;
                # incrementally update the histogram using the newly observed truth
                # to keep the rank distribution current.
                # (We update using the window just before each true y[t].)
                # For t = split .. N-1, the context window is y[t-w:t]
                # We update the model's histogram with the new insertion rank each step.
                train_hist = model.hist_.copy()
                train_total = model.total_
                # We'll maintain a simple copy of the growing data for consistent windows
                grow_data = y[:split].copy()

                for t in range(split, N):
                    # Predict from the current window at the end of grow_data
                    yhat = model.predict(1, start_values=grow_data[-w:])[0]
                    preds.append(yhat)

                    # Update grow_data with the truth
                    grow_data = np.append(grow_data, y[t])

                    # Update histogram with the new insertion rank based on the *previous* window
                    prev_window = grow_data[-(w + 1) : -1]  # the w values before y[t]
                    k = RankInsertionForecaster._insertion_rank(prev_window, y[t])
                    train_hist[k] += 1.0
                    train_total += 1

                    # Refresh model's target rank from updated histogram
                    if strat == "mode":
                        maxc = train_hist.max()
                        candidates = np.flatnonzero(train_hist == maxc)
                        model._target_rank_ = int(candidates[len(candidates) // 2])
                    else:
                        ks = np.arange(w + 1, dtype=float)
                        model._target_rank_ = float((ks * train_hist).sum() / train_hist.sum())

                preds = np.asarray(preds, dtype=float)
                score = score_fn(y_val, preds)
                if score < best_score:
                    best_score = score
                    best_conf = {"window": w, "rank_strategy": strat}

        if best_conf is None:
            raise RuntimeError("AutoRankInsertion failed to find a valid configuration.")

        # Refit best model on the full series
        self.model_ = RankInsertionForecaster(**best_conf).fit(y)
        self.best_ = {
            "config": best_conf,
            "val_score": best_score,
            "metric": self.metric,
        }
        return self

    def predict(self, h):
        if self.model_ is None:
            raise RuntimeError("AutoRankInsertion is not fitted yet.")
        return self.model_.predict(h)
