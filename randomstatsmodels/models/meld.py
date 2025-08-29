from typing import Optional, Tuple, Dict, Iterable
import numpy as np


# ===================== MELDForecaster =====================
class MELDForecaster:
    """
    Multiscale Embedding with Learned Dynamics (MELD).

    Combines:
      - Multiscale, causal moving-average embeddings (wavelet-inspired, causal).
      - Nonlinear feature lifting via Random Fourier Features (RBF kernel approximation).
      - Linear dynamics in feature space (ridge-regularized regression).
      - Instance-based analog correction (kNN in lifted space), blended adaptively 
        by a confidence weight from neighbor distances.
    """

    def __init__(
        self,
        lags: int = 12,
        scales: Tuple[int, ...] = (1, 3, 7),
        rff_features: int = 128,
        lengthscale: float = 3.0,
        ridge: float = 1e-2,
        knn: int = 5,
        blend_strength: float = 1.0,
        random_state: Optional[int] = 123,
    ):
        """
        Initialize a MELDForecaster.

        Parameters
        ----------
        lags : int, default=12
            Number of recent points used at each scale to define the state.
        scales : tuple of int, default=(1, 3, 7)
            Causal moving-average windows (in time steps) to create multiscale views.
            Must include ``1`` to keep the raw scale.
        rff_features : int, default=128
            Number of random Fourier features (RBF kernel approximation).
        lengthscale : float, default=3.0
            RBF kernel lengthscale for RFF. Larger values → smoother features.
        ridge : float, default=1e-2
            Ridge regularization strength for linear regression in feature space.
        knn : int, default=5
            Number of nearest analogs (in lifted feature space) for residual correction.
            Set to 0 to disable analog blending.
        blend_strength : float, default=1.0
            Controls how strongly analogs influence forecasts. Larger values
            increase analog weight when nearest neighbor distances are small.
        random_state : int or None, default=123
            Seed for reproducibility of random Fourier feature sampling.

        Attributes
        ----------
        _y : ndarray of shape (n_samples,), optional
            Training time series.
        _beta : ndarray of shape (n_features,), optional
            Learned regression coefficients.
        _W : ndarray of shape (rff_features, dim_raw), optional
            Random Fourier feature weight matrix.
        _b : ndarray of shape (rff_features,), optional
            Random Fourier feature phase vector.
        _train_Z : ndarray of shape (n_samples, rff_features), optional
            Lifted training features used for analog search.
        _y_next : ndarray of shape (n_samples,), optional
            Next-step targets for analog correction.
        _sigma_d : float
            Typical nearest-neighbor distance in lifted space, for analog confidence scaling.
        """
        self.lags = int(lags)
        self.scales = tuple(int(s) for s in scales)
        if 1 not in self.scales:
            self.scales = (1,) + self.scales  # ensure raw scale present
        self.rff_features = int(rff_features)
        self.lengthscale = float(lengthscale)
        self.ridge = float(ridge)
        self.knn = int(knn)
        self.blend_strength = float(blend_strength)
        self.random_state = random_state

        # Fitted artifacts
        self._y: Optional[np.ndarray] = None  # training series
        self._W: Optional[np.ndarray] = None  # RFF matrix (D x d)
        self._b: Optional[np.ndarray] = None  # RFF phase (D,)
        self._beta: Optional[np.ndarray] = None  # linear weights over [1, x_raw, z_rff]
        self._dim_raw: Optional[int] = None
        self._train_Z: Optional[np.ndarray] = None  # lifted features for analog search (n_samples x D)
        self._y_next: Optional[np.ndarray] = None  # next values for each training row
        self._sigma_d: float = 1.0  # distance scale for analog confidence

    # ---------- utilities ----------

    @staticmethod
    def _causal_ma(y: np.ndarray, w: int) -> np.ndarray:
        """Causal moving average with window w: at t, mean of y[max(0,t-w+1):t+1]."""
        n = y.size
        cs = np.cumsum(y, dtype=float)
        out = np.empty(n, dtype=float)
        for t in range(n):
            if t < w - 1:
                out[t] = cs[t] / (t + 1)
            else:
                out[t] = (cs[t] - cs[t - w]) / w
        return out

    def _multiscale_series(self, y: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute causal moving-averaged series for each scale."""
        return {s: (y if s == 1 else self._causal_ma(y, s)) for s in self.scales}

    def _build_raw_state(self, y: np.ndarray, t: int, ms: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Build the raw state vector at time t (predicting t+1):
          - For each scale s: take [ms[s][t - i] for i=0..lags-1] (newest first)
          - First differences on the raw scale (lags-1 entries): Δy_t,...,Δy_{t-lags+2}
        Concatenate in order of scales (ascending), then diffs.
        """
        m = self.lags
        if t < m - 1:
            raise ValueError("Not enough history to build state.")
        parts = []
        for s in self.scales:
            seq = ms[s][t - m + 1 : t + 1][::-1]  # newest first
            parts.append(seq)
        # first differences on raw scale (s=1)
        raw_seq = ms[1][t - m + 1 : t + 1]
        diffs = np.diff(raw_seq)[::-1]  # newest diff first, length m-1
        x_raw = np.concatenate(parts + [diffs], axis=0)
        return x_raw

    def _init_rff(self, dim_raw: int):
        """Sample Random Fourier Features for an RBF kernel on x_raw."""
        rng = np.random.RandomState(self.random_state)
        D = self.rff_features
        # RBF kernel ~ exp(-||x-x'||^2 / (2 * ell^2))
        # W ~ N(0, 1/ell^2 * I), b ~ Uniform[0, 2π]
        self._W = rng.normal(loc=0.0, scale=1.0 / max(self.lengthscale, 1e-9), size=(D, dim_raw))
        self._b = rng.uniform(0.0, 2.0 * np.pi, size=D)

    def _lift(self, x_raw: np.ndarray) -> np.ndarray:
        """RFF lift z(x) = sqrt(2/D) * cos(W x + b)."""
        Wx = self._W @ x_raw  # (D,)
        z = np.cos(Wx + self._b)
        return np.sqrt(2.0 / self.rff_features) * z

    def _phi(self, x_raw: np.ndarray) -> np.ndarray:
        """Full feature vector: [1, x_raw, z_rff]."""
        z = self._lift(x_raw)
        return np.concatenate(([1.0], x_raw, z), axis=0)

    def _design_matrix(self, X_raw: np.ndarray) -> np.ndarray:
        """Build design matrix Φ from a matrix of raw states (n x d_raw)."""
        Z = np.array([self._lift(x) for x in X_raw], dtype=float)  # (n x D)
        ones = np.ones((X_raw.shape[0], 1), dtype=float)
        return np.concatenate([ones, X_raw, Z], axis=1), Z

    @staticmethod
    def _estimate_sigma_d(Z: np.ndarray, sample: int = 512) -> float:
        """Estimate a typical nearest-neighbor distance scale in Z-space."""
        n = Z.shape[0]
        if n <= 2:
            return 1.0
        idx = np.arange(n)
        if n > sample:
            rng = np.random.RandomState(123)
            idx = rng.choice(n, size=sample, replace=False)
        dmins = []
        for i in idx:
            d = np.sqrt(np.sum((Z - Z[i]) ** 2, axis=1))
            d[i] = np.inf
            dmins.append(np.min(d))
        sig = float(np.median(dmins))
        return max(sig, 1e-6)

    # ---------- fitting ----------

    def fit(self, y: np.ndarray):
        """
        Fit the MELD model to a univariate time series.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Training time series.

        Returns
        -------
        self : MELDForecaster
            Fitted model instance.

        Raises
        ------
        ValueError
            If not enough data points are provided (requires at least `lags + 1`).
        """
        y = np.asarray(y, dtype=float)
        N = y.size
        m = self.lags
        if N < m + 1:
            raise ValueError(f"Need at least {m+1} points, got {N}.")

        self._y = y.copy()
        ms = self._multiscale_series(y)

        # Build training rows (t = m-1 .. N-2) -> predict y[t+1]
        rows = []
        targets = []
        for t in range(m - 1, N - 1):
            rows.append(self._build_raw_state(y, t, ms))
            targets.append(y[t + 1])
        X_raw = np.vstack(rows)  # (n_samples x dim_raw)
        Y = np.array(targets, dtype=float)  # (n_samples,)

        if self._W is None:
            self._dim_raw = X_raw.shape[1]
            self._init_rff(self._dim_raw)

        Phi, Z = self._design_matrix(X_raw)  # Φ = [1, x_raw, z_rff]
        # Ridge fit (do not penalize intercept):
        lam = self.ridge
        nfeat = Phi.shape[1]
        XtX = Phi.T @ Phi
        reg = lam * np.eye(nfeat, dtype=float)
        reg[0, 0] = 0.0  # no penalty on intercept
        Xty = Phi.T @ Y
        self._beta = np.linalg.solve(XtX + reg, Xty)

        # Store lifted features and next values for analog correction
        self._train_Z = Z.copy()
        self._y_next = Y.copy()
        self._sigma_d = self._estimate_sigma_d(self._train_Z)
        return self

    # ---------- forecasting primitives ----------

    def _predict_one_from_series(self, y_series: np.ndarray) -> float:
        """
        Predict next value given the *full* current series.
        Uses training-time analog library but builds state from y_series's tail.
        """
        y_series = np.asarray(y_series, dtype=float)
        if y_series.size < self.lags:
            raise ValueError("Not enough history in provided series.")
        ms_cur = self._multiscale_series(y_series)
        t = y_series.size - 1
        x_raw = self._build_raw_state(y_series, t, ms_cur)
        phi = self._phi(x_raw)
        y_lin = float(np.dot(self._beta, phi))

        if self.knn <= 0 or self._train_Z is None or self._sigma_d <= 0:
            return y_lin

        # Analog correction in lifted space
        z = self._lift(x_raw)  # (D,)
        d = np.sqrt(np.sum((self._train_Z - z) ** 2, axis=1))  # (n_train,)
        k = min(self.knn, d.size)
        nn_idx = np.argpartition(d, k - 1)[:k]
        nn_d = d[nn_idx]
        # If there is an exact (or near-exact) analog, trust it strongly.
        d_min = float(np.min(nn_d))
        # inverse-distance weights
        w = 1.0 / (nn_d + 1e-8)
        w = w / np.sum(w)
        y_analog = float(np.dot(w, self._y_next[nn_idx]))

        # Adaptive blend: gamma in [0,1], larger when d_min << sigma_d
        # gamma = 1 - exp(- blend_strength * (sigma_d / (d_min + eps)))
        # Alternative (smoother): gamma = exp( - d_min / (sigma_d) )**blend_strength
        gamma = float(np.exp(-d_min / (self._sigma_d + 1e-12)) ** max(self.blend_strength, 0.0))
        return (1.0 - gamma) * y_lin + gamma * y_analog

    # ---------- public API ----------

    def predict(self, h: int, start_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate iterative forecasts for a given horizon.

        Parameters
        ----------
        h : int
            Number of forecast steps.
        start_values : ndarray of shape (>=lags,), optional
            Starting history to condition forecasts. If None, uses training series.

        Returns
        -------
        preds : ndarray of shape (h,)
            Forecasted values.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If `start_values` is provided but shorter than `lags`.
        """
        if self._beta is None or self._y is None:
            raise RuntimeError("Fit the model before predicting.")
        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        if start_values is None:
            y_cur = self._y.copy()
        else:
            y_cur = np.asarray(start_values, dtype=float)
            if y_cur.size < self.lags:
                raise ValueError(f"start_values must have at least {self.lags} points.")

        fcst = []
        for _ in range(h):
            yhat = self._predict_one_from_series(y_cur)
            fcst.append(yhat)
            y_cur = np.append(y_cur, yhat)
        return np.array(fcst, dtype=float)


# ================= AutoMELD =================
class AutoMELD:
    """
    Validation-based tuner for MELDForecaster.

    Performs grid search over hyperparameter combinations and selects the
    configuration with the best validation score.

    Examples
    --------
    >>> mdl = AutoMELD().fit(y)
    >>> preds = mdl.predict(10)
    """

    def __init__(
        self,
        lags_grid: Iterable[int] = (8, 12, 16),
        scales_grid: Iterable[Tuple[int, ...]] = ((1, 3, 7), (1, 2, 4, 8)),
        rff_features_grid: Iterable[int] = (64, 128),
        lengthscales: Iterable[float] = (2.0, 4.0),
        ridges: Iterable[float] = (1e-3, 1e-2),
        knns: Iterable[int] = (0, 5),
        blend_strengths: Iterable[float] = (0.8, 1.5),
        metric: str = "mae",
        random_state: Optional[int] = 123,
    ):
        """
        Initialize AutoMELD grid tuner.

        Parameters
        ----------
        lags_grid : iterable of int, default=(8, 12, 16)
            Candidate lag values.
        scales_grid : iterable of tuple of int, default=((1, 3, 7), (1, 2, 4, 8))
            Candidate multiscale embedding sets.
        rff_features_grid : iterable of int, default=(64, 128)
            Candidate numbers of random Fourier features.
        lengthscales : iterable of float, default=(2.0, 4.0)
            Candidate RBF kernel lengthscales.
        ridges : iterable of float, default=(1e-3, 1e-2)
            Candidate ridge penalties.
        knns : iterable of int, default=(0, 5)
            Candidate nearest-neighbor counts for analog correction.
        blend_strengths : iterable of float, default=(0.8, 1.5)
            Candidate blend strengths for analog correction.
        metric : {"mae", "rmse"}, default="mae"
            Validation scoring metric.
        random_state : int or None, default=123
            Random seed for reproducibility.

        Attributes
        ----------
        model_ : MELDForecaster or None
            Best fitted model after tuning.
        best_ : dict or None
            Dictionary with best configuration and validation score.
        """

        self.lags_grid = list(lags_grid)
        self.scales_grid = [tuple(s) for s in scales_grid]
        self.rff_features_grid = list(rff_features_grid)
        self.lengthscales = list(lengthscales)
        self.ridges = list(ridges)
        self.knns = list(knns)
        self.blend_strengths = list(blend_strengths)
        self.metric = metric.lower()
        self.random_state = random_state

        self.model_: Optional[MELDForecaster] = None
        self.best_: Optional[Dict] = None

    @staticmethod
    def _mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    @staticmethod
    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def fit(self, y: np.ndarray, val_fraction: float = 0.25):
        """
        Fit AutoMELD by tuning hyperparameters on a validation split.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Input time series.
        val_fraction : float, default=0.25
            Fraction of data reserved for validation.

        Returns
        -------
        self : AutoMELD
            Fitted tuner with best model stored.

        Raises
        ------
        ValueError
            If insufficient data exists for the largest lag value.
        RuntimeError
            If no valid configuration can be fitted.
        """
        y = np.asarray(y, dtype=float)
        N = len(y)
        n_val = max(16, int(N * float(val_fraction)))
        split = N - n_val
        if split < max(self.lags_grid) + 1:
            raise ValueError("Not enough data before validation for largest lags.")
        y_train, y_val = y[:split], y[split:]

        best_score = np.inf
        best_conf = None
        best_model = None

        for lags in self.lags_grid:
            for scales in self.scales_grid:
                for D in self.rff_features_grid:
                    for ell in self.lengthscales:
                        for lam in self.ridges:
                            for k in self.knns:
                                for bs in self.blend_strengths:
                                    try:
                                        mdl = MELDForecaster(
                                            lags=lags,
                                            scales=scales,
                                            rff_features=D,
                                            lengthscale=ell,
                                            ridge=lam,
                                            knn=k,
                                            blend_strength=bs,
                                            random_state=self.random_state,
                                        ).fit(y_train)
                                    except Exception:
                                        continue
                                    # One-step rolling over validation (use current true history, model fixed)
                                    preds = []
                                    y_so_far = y[:split].copy()
                                    for t in range(split, N):
                                        yhat = mdl._predict_one_from_series(y_so_far)
                                        preds.append(yhat)
                                        y_so_far = np.append(y_so_far, y[t])
                                    preds = np.array(preds, dtype=float)
                                    score = (
                                        self._mae(y_val, preds) if self.metric == "mae" else self._rmse(y_val, preds)
                                    )
                                    if score < best_score:
                                        best_score = score
                                        best_conf = dict(
                                            lags=lags,
                                            scales=scales,
                                            rff_features=D,
                                            lengthscale=ell,
                                            ridge=lam,
                                            knn=k,
                                            blend_strength=bs,
                                        )
                                        best_model = mdl

        if best_model is None:
            raise RuntimeError("AutoMELD failed to find a valid configuration.")

        # Refit on full data with best config (fresh fit to use all history for training library)
        final = MELDForecaster(**best_conf, random_state=self.random_state).fit(y)
        self.model_ = final
        self.best_ = {"config": best_conf, "val_score": best_score}
        return self

    def predict(self, h: int, start_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate forecasts with the best fitted MELDForecaster.

        Parameters
        ----------
        h : int
            Forecast horizon.
        start_values : ndarray of shape (>=lags,), optional
            Starting history for forecasts. If None, uses the full training series.

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
            raise RuntimeError("Call fit() first.")
        return self.model_.predict(h, start_values=start_values)
