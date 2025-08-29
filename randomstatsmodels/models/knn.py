import numpy as np
from typing import Optional, Union, Sequence, Dict, List

from ..metrics import mae, rmse

ArrayLike = Union[Sequence[float], np.ndarray]


# ================= KNNForecaster =================
class KNNForecaster:
    """
    K-Nearest Neighbors Analog Forecaster.

    Uses a fixed-length history window (embedding) to find similar
    past subsequences in the training series and forecasts by
    averaging the next values of the k nearest analogs.

    Parameters
    ----------
    window : int, default=8
        Length of the history window (embedding dimension).
    k : int, default=3
        Number of nearest neighbors to use for forecasting.

    Attributes
    ----------
    data : ndarray of shape (n_samples,)
        Stored training series.
    _X : ndarray of shape (n_windows, window)
        Matrix of past subsequences (windowed embeddings).
    _y_next : ndarray of shape (n_windows,)
        Next values following each subsequence in `_X`.
    """

    def __init__(self, window: int = 8, k: int = 3) -> None:
        self.window = int(window)
        self.k = int(k)
        self.data: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._y_next: Optional[np.ndarray] = None

    def fit(self, y: ArrayLike) -> "KNNForecaster":
        """
        Fit the forecaster by storing the series and building the analog library.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Training time series.

        Returns
        -------
        self : KNNForecaster
            Fitted instance.

        Raises
        ------
        ValueError
            If the series is shorter than `window + 1`.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < self.window + 1:
            raise ValueError(f"Need at least window+1={self.window+1} points, got {n}.")
        # Store the full series
        self.data = y.copy()
        # Build the library of subsequences (length = window) and their next values
        n_windows = n - self.window
        # Matrix of shape (n_windows, window)
        self._X = np.array([y[i : i + self.window] for i in range(n_windows)], dtype=float)
        # Next values for each window (the value right after each subsequence)
        self._y_next = y[self.window :]  # length n_windows
        return self

    def _predict_one(self, current_window: np.ndarray) -> float:
        """
        Predict the next value given a current history window.

        Parameters
        ----------
        current_window : ndarray of shape (window,)
            Most recent subsequence.

        Returns
        -------
        float
            Forecasted next value.
        """
        # Compute distances from current_window to each stored window in the library
        # Using Euclidean distance (L2 norm)
        diff = self._X - current_window  # broadcasting: _X is (n_windows, window), current_window is (window,)
        dists = np.sqrt(np.sum(diff**2, axis=1))
        # Find indices of the k smallest distances
        if self.k == 1:
            idx = np.argmin(dists)
            min_dist = dists[idx]
            # If an exact match (distance 0), return its next value
            if min_dist < 1e-12:
                return float(self._y_next[idx])
            # Otherwise return that neighbor's next value (with k=1, no averaging needed)
            return float(self._y_next[idx])
        else:
            # For k > 1, get k nearest indices (argpartition for efficiency)
            if self.k <= len(dists):
                nn_idx = np.argpartition(dists, self.k)[: self.k]
            else:
                # If k larger than available windows (should not usually happen), use all
                nn_idx = np.arange(len(dists))
            nn_dists = dists[nn_idx]
            # If any perfect match found, use its next value directly
            min_idx = np.argmin(nn_dists)
            if nn_dists[min_idx] < 1e-12:
                return float(self._y_next[nn_idx[min_idx]])
            # Compute inverse-distance weights
            inv_w = 1.0 / (nn_dists + 1e-8)
            weights = inv_w / np.sum(inv_w)
            # Weighted average of the corresponding next values
            next_vals = self._y_next[nn_idx]
            return float(np.dot(weights, next_vals))

    def predict(self, h: int, start_values: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Iteratively forecast h future steps.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps).
        start_values : array-like of shape (window,), optional
            Optional starting window. If None, uses the last
            `window` values from the training series.

        Returns
        -------
        preds : ndarray of shape (h,)
            Forecasted values.

        Raises
        ------
        RuntimeError
            If called before `fit()`.
        ValueError
            If `start_values` length != `window`.
        """
        if self.data is None:
            raise RuntimeError("Fit the model before calling predict().")
        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)
        # Determine the initial window for forecasting
        if start_values is None:
            current_window = self.data[-self.window :].copy()
        else:
            start_values = np.asarray(start_values, dtype=float)
            if len(start_values) != self.window:
                raise ValueError(f"start_values must have length {self.window}.")
            current_window = start_values.copy()
        preds = []
        for _ in range(h):
            # Predict next value from current window
            yhat = self._predict_one(current_window)
            preds.append(yhat)
            # Roll the window: drop the oldest and append the new forecast
            current_window[:-1] = current_window[1:]
            current_window[-1] = yhat
        return np.array(preds)


class AutoKNN:
    """
    Automatic tuner for :class:`KNNForecaster`.

    Performs grid search over `window` and `k` using a holdout
    validation split, selects the best configuration, and refits
    on the full series.

    Parameters
    ----------
    window_grid : iterable of int, default=(4, 8, 12)
        Candidate window lengths.
    k_grid : iterable of int, default=(1, 3, 5)
        Candidate neighbor counts.
    metric : {"mae", "rmse"}, default="mae"
        Validation error metric.

    Attributes
    ----------
    model_ : KNNForecaster or None
        Best fitted forecaster.
    best_ : dict or None
        Dictionary with best configuration and validation score.
    """

    def __init__(
        self,
        window_grid: Sequence[int] = (4, 8, 12),
        k_grid: Sequence[int] = (1, 3, 5),
        metric: str = "mae",
    ) -> None:
        self.window_grid = list(window_grid)
        self.k_grid = list(k_grid)
        self.metric = metric.lower()
        self.model_: Optional[KNNForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: ArrayLike, val_fraction: float = 0.25) -> "AutoKNN":
        """
        Grid search tuner for KNNForecaster.

        Splits the series into training and validation,
        evaluates candidate configs, and refits the best model
        on the full series.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time series.
        val_fraction : float, default=0.25
            Fraction of samples reserved for validation.

        Returns
        -------
        self : AutoKNN
            Fitted tuner.

        Raises
        ------
        ValueError
            If too few samples for validation split.
        RuntimeError
            If no valid configuration is found.
        """
        y = np.asarray(y, dtype=float)
        N = len(y)
        # Determine validation size (at least 16 points, as a safety floor)
        n_val = max(16, int(N * float(val_fraction)))
        if N - n_val < self.window_grid[0] + 1:
            raise ValueError("Not enough data to split for validation given smallest window.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]
        best_score = np.inf
        best_conf = None

        # Grid search over window and k
        for L in self.window_grid:
            for k in self.k_grid:
                # If training data is too short for this window, skip
                if len(y_train) < L + 1:
                    continue
                try:
                    model = KNNForecaster(window=L, k=k).fit(y_train)
                except Exception:
                    # Skip invalid configurations
                    continue
                # Evaluate on validation set with one-step rolling forecasts
                preds = []
                # Start with the model as fitted on y_train, then update through val
                for t in range(split, N):
                    # Predict one step ahead from current model state
                    # (We use the last L points from model.data to forecast next)
                    yhat = model.predict(1)[0]
                    preds.append(yhat)
                    # Update model with the actual value at time t (y[t]) before next step
                    model.data = np.append(model.data, y[t])
                    # Also update the precomputed library with the new data point
                    # (append new window ending at t to _X and its next value y[t] to _y_next if possible)
                    # Note: When moving through validation, we only use one-step forecast then update,
                    # so the library grows, but we won't use beyond one-step ahead at any time.
                    if len(model.data) > model.window:
                        new_window = model.data[-model.window - 1 : -1]  # the window ending at time t-1
                        # Actually, since we've just appended y[t] (truth) to model.data,
                        # the last 'window' points from model.data (excluding the just appended one)
                        # form the window ending at t-1, and y[t] is the next value for that window.
                        model._X = np.vstack([model._X, new_window])
                        model._y_next = np.append(model._y_next, model.data[-1])
                preds = np.array(preds)
                # Calculate error
                score = mae(y_val, preds) if self.metric == "mae" else rmse(y_val, preds)
                if score < best_score:
                    best_score = score
                    best_conf = {"window": L, "k": k}
        if best_conf is None:
            raise RuntimeError("AutoKNN failed to find a valid configuration.")
        # Refit best model on the full dataset
        best_model = KNNForecaster(**best_conf).fit(y)
        self.model_ = best_model
        self.best_ = {"config": best_conf, "val_score": best_score}
        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast with the best tuned model.

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
            raise RuntimeError("AutoKNN is not fitted yet.")
        return self.model_.predict(h)
