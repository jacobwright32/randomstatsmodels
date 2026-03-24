"""
Fractional Calculus Forecaster
==============================

Uses Grünwald-Letnikov fractional derivatives to transform the series into
a memory-preserving stationary representation.  An autoregressive model is
fitted in the fractionally-differenced space and forecasts are mapped back
via the inverse operator (fractional cumulative summation).

Key idea
--------
Standard integer differencing (d=1) removes ALL long-range memory.
Fractional differencing with d ∈ (0, 1) removes just enough persistence
to achieve near-stationarity while retaining the maximum possible memory —
critical for forecasting accuracy.

Mathematical foundation
-----------------------
Grünwald-Letnikov fractional difference operator:

    (1 - B)^d y_t = Σ_{k=0}^{∞} w_k · y_{t-k}

where B is the backshift operator and the weights are:

    w_0 = 1
    w_k = -w_{k-1} · (d - k + 1) / k     for k ≥ 1

The inverse (fractional cumulative sum) uses weights from d → -d.
"""

from typing import Optional, Dict, Iterable
import numpy as np
from ..metrics import mae, rmse


# -----------------------------------------------------------------------
# Fractional differencing utilities
# -----------------------------------------------------------------------

def _gl_weights(d: float, n: int, threshold: float = 1e-6) -> np.ndarray:
    """Grünwald-Letnikov weights for fractional order *d*, up to *n* terms."""
    w = np.zeros(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = -w[k - 1] * (d - k + 1) / k
        if abs(w[k]) < threshold and k > 10:
            break
    return w


def frac_diff(y: np.ndarray, d: float, window: int = 0) -> np.ndarray:
    """
    Apply fractional differencing of order *d* to series *y*.

    Parameters
    ----------
    y : ndarray
        Input time series.
    d : float
        Differencing order, typically in (0, 1).
    window : int
        If > 0, truncate GL weights to this many terms (fixed-width window).
        If 0, use the full series length.

    Returns
    -------
    y_d : ndarray of same length as y.
        Fractionally differenced series (first few values may be inaccurate
        due to boundary; we leave them so indices stay aligned).
    """
    y = np.asarray(y, float)
    n = len(y)
    k = window if window > 0 else n
    w = _gl_weights(d, k)

    y_d = np.zeros(n)
    for t in range(n):
        # convolve: y_d[t] = Σ w[j] * y[t-j]  for j=0..min(t, k-1)
        upper = min(t + 1, len(w))
        y_d[t] = np.dot(w[:upper], y[t::-1][:upper])
    return y_d


def frac_cumsum(y_d: np.ndarray, d: float, y_prefix: np.ndarray,
                window: int = 0) -> np.ndarray:
    """
    Inverse of frac_diff: reconstruct original-scale values from a
    fractionally differenced series.

    Parameters
    ----------
    y_d : ndarray
        Values in the differenced space to invert.
    d : float
        The differencing order used.
    y_prefix : ndarray
        The last `window` (or all) original-scale values before the
        forecast period, needed to "undo" the convolution.
    window : int
        GL weight truncation window (same as used in frac_diff).
    """
    # Inverse weights: just negate d
    n_out = len(y_d)
    prefix = np.asarray(y_prefix, float)
    n_prefix = len(prefix)
    k = window if window > 0 else n_prefix + n_out
    # weights for order -d (cumulative sum)
    w_inv = _gl_weights(-d, k)

    # Build full buffer: prefix + space for output
    buf = np.concatenate([prefix, np.zeros(n_out)])
    start = n_prefix

    for t in range(n_out):
        # y_d[t] = Σ w_diff[j] * buf[start+t - j]  for j=0..
        # => buf[start+t] = (y_d[t] - Σ_{j=1} w_diff[j] * buf[start+t-j]) / w_diff[0]
        # But it's simpler to use the inverse-weight convolution directly:
        idx = start + t
        s = y_d[t]
        upper = min(idx, len(w_inv) - 1)
        for j in range(1, upper + 1):
            s -= w_inv[j] * buf[idx - j]
        buf[idx] = s  # w_inv[0] = 1 always

    return buf[start:]


# -----------------------------------------------------------------------
# Forecaster
# -----------------------------------------------------------------------

class FracDiffForecaster:
    """
    Fractional-differencing + autoregressive forecaster.

    Parameters
    ----------
    d : float, default=0.4
        Fractional differencing order in (0, 1).
    ar_order : int, default=5
        Number of autoregressive lags to fit in the differenced space.
    gl_window : int, default=64
        Truncation window for GL weights (0 = full length).
    ridge : float, default=1e-4
        Ridge regularisation for the AR coefficient estimation.
    """

    def __init__(self, d: float = 0.4, ar_order: int = 5,
                 gl_window: int = 64, ridge: float = 1e-4):
        self.d = float(d)
        self.ar_order = int(ar_order)
        self.gl_window = int(gl_window)
        self.ridge = float(ridge)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self._y: Optional[np.ndarray] = None
        self._y_d: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "FracDiffForecaster":
        y = np.asarray(y, float)
        n = len(y)
        p = self.ar_order
        if n < p + 4:
            raise ValueError(f"Need at least {p + 4} points, got {n}.")

        self._y = y.copy()
        self._y_d = frac_diff(y, self.d, self.gl_window)

        # Build AR design matrix in the differenced space
        # Skip the first max(p, gl_window) points for warm-up
        warmup = max(p, min(self.gl_window, n // 4))
        X_rows, targets = [], []
        for t in range(warmup, n):
            X_rows.append(self._y_d[t - p: t][::-1])
            targets.append(self._y_d[t])
        X = np.vstack(X_rows)
        Y = np.array(targets)

        # Ridge regression
        XtX = X.T @ X + self.ridge * np.eye(p)
        XtY = X.T @ Y
        self.coef_ = np.linalg.solve(XtX, XtY)
        self.intercept_ = np.mean(Y - X @ self.coef_)
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Fit the model before predicting.")

        p = self.ar_order
        y_d = self._y_d

        # AR forecast in differenced space
        window = y_d[-p:].copy()
        fcast_d = []
        for _ in range(h):
            yhat_d = float(np.dot(self.coef_, window[::-1])) + self.intercept_
            fcast_d.append(yhat_d)
            window[:-1] = window[1:]
            window[-1] = yhat_d
        fcast_d = np.array(fcast_d)

        # Invert fractional differencing to get original-scale forecasts
        fcast = frac_cumsum(fcast_d, self.d, self._y, self.gl_window)
        return fcast


# -----------------------------------------------------------------------
# Auto tuner
# -----------------------------------------------------------------------

class AutoFracDiff:
    """
    Auto-tuned fractional differencing forecaster.

    Grid-searches over differencing order *d* and AR lag order *p* to
    find the combination that minimises validation error.

    Parameters
    ----------
    d_grid : iterable of float
        Candidate fractional differencing orders.
    ar_grid : iterable of int
        Candidate AR orders.
    gl_window_grid : iterable of int
        Candidate GL weight window sizes.
    """

    def __init__(
        self,
        d_grid: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        ar_grid: Iterable[int] = (3, 5, 8, 12),
        gl_window_grid: Iterable[int] = (32, 64),
    ):
        self.d_grid = d_grid
        self.ar_grid = ar_grid
        self.gl_window_grid = gl_window_grid
        self.model_: Optional[FracDiffForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction: float = 0.25,
            metric: str = "mae") -> "AutoFracDiff":
        y = np.asarray(y, float)
        N = len(y)
        n_val = max(6, int(N * val_fraction))
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for d in self.d_grid:
            for p in self.ar_grid:
                for glw in self.gl_window_grid:
                    try:
                        m = FracDiffForecaster(d=d, ar_order=p, gl_window=glw)
                        m.fit(y_train)
                        preds = m.predict(len(y_val))
                        s = score_fn(y_val, preds)
                        if np.isfinite(s) and s < best_score:
                            best_score = s
                            best_conf = dict(d=d, ar_order=p, gl_window=glw)
                    except Exception:
                        continue

        if best_conf is None:
            best_conf = dict(d=0.4, ar_order=5, gl_window=64)
            best_score = float("inf")

        final = FracDiffForecaster(**best_conf)
        final.fit(y)
        self.model_ = final
        self.best_ = dict(config=best_conf, val_score=best_score)
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the model before predicting.")
        return self.model_.predict(h)
