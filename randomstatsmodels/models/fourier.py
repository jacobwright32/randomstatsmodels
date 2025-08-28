import numpy as np


# ================= FourierForecaster =================
class FourierForecaster:
    """
    Fourier (harmonic) forecaster with optional linear trend.
    Fits y_t ≈ c0 + c1*t + Σ_k (a_k cos(2π k t / N) + b_k sin(2π k t / N)),
    using up to `n_harmonics` harmonics from the sample length N.
    """

    def __init__(self, n_harmonics=3, trend="linear"):
        """
        Parameters
        ----------
        n_harmonics : int
            Number of Fourier harmonics to include (k = 1..n_harmonics).
        trend : {"none","linear"}
            Whether to include a linear trend term.
        """
        self.n_harmonics = int(n_harmonics)
        self.trend = str(trend)
        if self.trend not in ("none", "linear"):
            raise ValueError("trend must be 'none' or 'linear'")
        self.coef_ = None
        self.n_ = None  # length of fitted series (defines base frequencies)
        self._Xcols_ = None  # cache columns builder for predict

    def _design_matrix(self, n, t_idx):
        """
        Build design matrix for times in t_idx (array of ints), using base length n.
        Columns: [1], optional t, then for k=1..H: cos(2π k t / n), sin(2π k t / n)
        """
        t = np.asarray(t_idx, dtype=float)
        cols = [np.ones_like(t)]
        if self.trend == "linear":
            cols.append(t)
        H = min(self.n_harmonics, max(0, n // 2 - 1))  # Nyquist-safe cap
        for k in range(1, H + 1):
            w = 2.0 * np.pi * k / float(n)
            cols.append(np.cos(w * t))
            cols.append(np.sin(w * t))
        if len(cols) == 0:
            raise RuntimeError("Empty design matrix.")
        X = np.vstack(cols).T
        return X, H

    def fit(self, y):
        """
        Fit coefficients by least squares on the provided series y.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < 3:
            raise ValueError("Need at least 3 points to fit Fourier forecaster.")
        self.n_ = n
        X, H = self._design_matrix(n, np.arange(n))
        # Solve least squares
        self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        # keep meta for predict
        self._Xcols_ = {"H": H}
        return self

    def predict(self, h):
        """
        Forecast h steps ahead by extrapolating the fitted harmonic + trend structure.
        """
        if self.coef_ is None or self.n_ is None:
            raise RuntimeError("Fit the model before calling predict().")
        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)
        t_future = np.arange(self.n_, self.n_ + h)
        Xf, _ = self._design_matrix(self.n_, t_future)
        return Xf @ self.coef_


# ================= AutoFourier =================
class AutoFourier:
    """
    Automatic tuner for FourierForecaster: searches over number of harmonics and trend option.
    """

    def __init__(
        self,
        n_harmonics_grid=(0, 1, 2, 3, 5, 8),
        trend_options=("none", "linear"),
        metric="mae",
    ):
        """
        Parameters
        ----------
        n_harmonics_grid : iterable of int
            Candidate counts of harmonics to try.
        trend_options : iterable of {"none","linear"}
            Whether to include a trend term.
        metric : {"mae","rmse"}
            Validation metric to minimize.
        """
        self.n_harmonics_grid = [int(h) for h in n_harmonics_grid]
        self.trend_options = list(trend_options)
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'")
        self.model_ = None
        self.best_ = None

    def fit(self, y, val_fraction=0.25):
        """
        Split series into train/validation tail, run rolling one-step evaluation,
        pick the best configuration, then refit on the full series.
        """
        y = np.asarray(y, dtype=float)
        N = len(y)
        n_val = max(16, int(N * float(val_fraction)))
        if N - n_val < 3:
            raise ValueError("Not enough data to train before validation.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        def mae(a, b):
            return float(np.mean(np.abs(a - b)))

        def rmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)))

        score_fn = mae if self.metric == "mae" else rmse

        best_score = np.inf
        best_conf = None

        for H in self.n_harmonics_grid:
            for trend in self.trend_options:
                # Safety: ensure feasible with current train length
                if len(y_train) < 3:
                    continue
                try:
                    model = FourierForecaster(n_harmonics=H, trend=trend).fit(y_train)
                except Exception:
                    continue
                preds = []
                # Rolling one-step ahead through validation (refit each step or update?)
                # For Fourier with fixed base length, we refit quickly each step on the growing data
                # to keep frequencies aligned with current sample size.
                for t in range(split, N):
                    # Predict next value from current fit
                    yhat = model.predict(1)[0]
                    preds.append(yhat)
                    # Update by refitting on data up to t (fast LS on small design)
                    model = FourierForecaster(n_harmonics=H, trend=trend).fit(y[: t + 1])
                preds = np.asarray(preds)
                score = score_fn(y_val, preds)
                if score < best_score:
                    best_score = score
                    best_conf = {"n_harmonics": H, "trend": trend}

        if best_conf is None:
            raise RuntimeError("AutoFourier failed to find a valid configuration.")

        # Refit best model on full data
        self.model_ = FourierForecaster(**best_conf).fit(y)
        self.best_ = {
            "config": best_conf,
            "val_score": best_score,
            "metric": self.metric,
        }
        return self

    def predict(self, h):
        """
        Forecast using the best-found FourierForecaster.
        """
        if self.model_ is None:
            raise RuntimeError("AutoFourier is not fitted yet.")
        return self.model_.predict(h)
