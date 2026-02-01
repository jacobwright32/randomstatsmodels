import numpy as np
from typing import Optional, Union, Sequence, Dict, List

from ..metrics import mae, rmse

ArrayLike = Union[Sequence[float], np.ndarray]


# ================= SSAForecaster =================
class SSAForecaster:
    """
    Singular Spectrum Analysis (SSA) forecaster.

    Decomposes time series using SVD on a trajectory matrix to discover
    adaptive oscillatory modes (unlike fixed Fourier frequencies).

    Algorithm:
    1. Embed the series into a trajectory matrix (Hankel matrix)
    2. Compute SVD to extract principal components
    3. Group and reconstruct the signal from selected components
    4. Forecast using the reconstructed components

    Parameters
    ----------
    window_length : int or None, default=None
        Window length for the trajectory matrix. If None, uses N//3.
    n_components : int or None, default=None
        Number of SVD components to use. If None, uses all significant components.
    grouping : list of list of int or None, default=None
        Component groupings for reconstruction. If None, uses components 0..n_components-1.

    Attributes
    ----------
    window_length_ : int
        Actual window length used.
    n_components_ : int
        Actual number of components used.
    U_ : ndarray
        Left singular vectors.
    s_ : ndarray
        Singular values.
    Vt_ : ndarray
        Right singular vectors (transposed).
    last_values_ : ndarray
        Last window_length values for forecasting.
    n_ : int
        Length of fitted series.
    """

    def __init__(
        self,
        window_length: Optional[int] = None,
        n_components: Optional[int] = None,
        grouping: Optional[List[List[int]]] = None,
    ) -> None:
        self.window_length = window_length
        self.n_components = n_components
        self.grouping = grouping

        self.window_length_: Optional[int] = None
        self.n_components_: Optional[int] = None
        self.U_: Optional[np.ndarray] = None
        self.s_: Optional[np.ndarray] = None
        self.Vt_: Optional[np.ndarray] = None
        self.last_values_: Optional[np.ndarray] = None
        self.n_: Optional[int] = None
        self._R: Optional[np.ndarray] = None  # For forecasting recurrence

    def _embed(self, y: np.ndarray, L: int) -> np.ndarray:
        """Create trajectory (Hankel) matrix from time series."""
        n = len(y)
        K = n - L + 1
        X = np.zeros((L, K))
        for i in range(L):
            X[i, :] = y[i : i + K]
        return X

    def _diagonal_averaging(self, X: np.ndarray, n: int) -> np.ndarray:
        """Reconstruct time series from matrix via diagonal averaging."""
        L, K = X.shape
        y = np.zeros(n)
        counts = np.zeros(n)

        for i in range(L):
            for j in range(K):
                y[i + j] += X[i, j]
                counts[i + j] += 1

        return y / counts

    def fit(self, y: ArrayLike) -> "SSAForecaster":
        """
        Fit the SSA model to the time series.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Training time series.

        Returns
        -------
        self : SSAForecaster
            Fitted instance.

        Raises
        ------
        ValueError
            If series is too short.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)

        if n < 4:
            raise ValueError("Need at least 4 data points for SSA.")

        # Determine window length
        if self.window_length is None:
            L = max(2, n // 3)
        else:
            L = min(self.window_length, n - 1)
        L = max(2, L)

        self.window_length_ = L
        self.n_ = n
        K = n - L + 1

        # Embed the series
        X = self._embed(y, L)

        # Compute SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        # Determine number of components
        if self.n_components is None:
            # Use components explaining 95% of variance
            total_var = np.sum(s**2)
            cumvar = np.cumsum(s**2) / total_var
            n_comp = int(np.searchsorted(cumvar, 0.95) + 1)
            n_comp = max(1, min(n_comp, len(s)))
        else:
            n_comp = min(self.n_components, len(s))

        self.n_components_ = n_comp
        self.U_ = U[:, :n_comp]
        self.s_ = s[:n_comp]
        self.Vt_ = Vt[:n_comp, :]

        # Store last L values for forecasting
        self.last_values_ = y[-L:].copy()

        # Compute the recurrence coefficients for forecasting
        # R = sum of (U_i[-1] * U_i[:-1]) / (1 - sum(U_i[-1]^2))
        U_last = self.U_[-1, :]
        nu_sq = np.sum(U_last**2)
        if nu_sq < 1 - 1e-10:
            self._R = np.sum(
                self.U_[:-1, :] * U_last[np.newaxis, :], axis=1
            ) / (1 - nu_sq)
        else:
            # Degenerate case: use simple average
            self._R = np.ones(L - 1) / (L - 1)

        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast h steps ahead using SSA recurrence.

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
        if self.U_ is None:
            raise RuntimeError("Fit the model before calling predict().")

        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        L = self.window_length_
        preds = np.zeros(h)

        # Initialize with last values
        window = self.last_values_.copy()

        for i in range(h):
            # Forecast next value using recurrence relation
            # y_{n+1} = R^T @ y_{n-L+2:n+1}
            y_next = np.dot(self._R, window[1:])
            preds[i] = y_next

            # Update window
            window[:-1] = window[1:]
            window[-1] = y_next

        return preds


# ================= AutoSSA =================
class AutoSSA:
    """
    Automatic tuner for SSAForecaster.

    Searches over window lengths and component counts to find the
    best configuration using validation performance.

    Parameters
    ----------
    window_fracs : iterable of float, default=(0.25, 0.33, 0.5)
        Candidate window lengths as fractions of series length.
    n_components_grid : iterable of int or None, default=(None, 2, 4, 8)
        Candidate number of components. None means auto-select.
    metric : {"mae", "rmse"}, default="mae"
        Validation error metric.

    Attributes
    ----------
    model_ : SSAForecaster or None
        Best fitted forecaster.
    best_ : dict or None
        Dictionary with best configuration and validation score.
    """

    def __init__(
        self,
        window_fracs: Sequence[float] = (0.25, 0.33, 0.5),
        n_components_grid: Sequence[Optional[int]] = (None, 2, 4, 8),
        metric: str = "mae",
    ) -> None:
        self.window_fracs = list(window_fracs)
        self.n_components_grid = list(n_components_grid)
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'")
        self.model_: Optional[SSAForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: ArrayLike, val_fraction: float = 0.25) -> "AutoSSA":
        """
        Grid search tuner for SSAForecaster.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time series data.
        val_fraction : float, default=0.25
            Fraction of series reserved for validation.

        Returns
        -------
        self : AutoSSA
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
        if N - n_val < 4:
            raise ValueError("Not enough data for validation split.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if self.metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for wfrac in self.window_fracs:
            window_length = max(2, int(len(y_train) * wfrac))

            for n_comp in self.n_components_grid:
                try:
                    model = SSAForecaster(
                        window_length=window_length,
                        n_components=n_comp,
                    ).fit(y_train)
                except Exception:
                    continue

                # One-step rolling forecast
                preds = []
                current_data = y_train.copy()
                for t in range(split, N):
                    model = SSAForecaster(
                        window_length=window_length,
                        n_components=n_comp,
                    ).fit(current_data)
                    yhat = model.predict(1)[0]
                    preds.append(yhat)
                    current_data = np.append(current_data, y[t])

                preds = np.array(preds)
                score = score_fn(y_val, preds)

                if score < best_score:
                    best_score = score
                    best_conf = {
                        "window_length": window_length,
                        "n_components": n_comp,
                    }

        if best_conf is None:
            raise RuntimeError("AutoSSA failed to find a valid configuration.")

        # Refit best model on full data (recalculate window_length for full data)
        final_window = max(2, int(N * self.window_fracs[0]))
        for wfrac in self.window_fracs:
            if int(len(y_train) * wfrac) == best_conf["window_length"]:
                final_window = max(2, int(N * wfrac))
                break

        self.model_ = SSAForecaster(
            window_length=final_window,
            n_components=best_conf["n_components"],
        ).fit(y)
        self.best_ = {
            "config": best_conf,
            "val_score": best_score,
            "metric": self.metric,
        }
        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast using the best-found SSAForecaster.

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
            raise RuntimeError("AutoSSA is not fitted yet.")
        return self.model_.predict(h)
