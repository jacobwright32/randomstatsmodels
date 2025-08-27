from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Iterable, List

import numpy as np

from .model_utils import _weighted_quantile, _penalty_value, _golden_section_minimize
from ..metrics import mae, rmse, smape, mape


# ================== HybridForecastNet =================
class HybridForecastNet:
    """
    Hybrid Dynamical Fourier + Trend/AR + GRU-residual forecaster.
    Deterministic point forecasts (extendable to probabilistic).
    """

    def __init__(
        self,
        seasonal_period=24,
        fourier_order=3,
        trend_degree=1,
        ar_order=3,
        hidden_size=16,
        rnn_layers=1,
        epochs=100,
        lr=0.01,
        seed=123,
    ):
        self.seasonal_period = int(seasonal_period)
        self.fourier_order = int(max(0, fourier_order))
        self.trend_degree = int(max(0, trend_degree))
        self.ar_order = int(max(0, ar_order))
        self.hidden_size = int(max(1, hidden_size))
        self.rnn_layers = int(max(1, rnn_layers))
        self.epochs = int(max(1, epochs))
        self.lr = float(lr)
        self.seed = int(seed)

        # learned linear parts
        self._fourier_coefs = None  # (2*fourier_order,)
        self._trend_coefs = None  # (trend_degree+1,)
        self._ar_coefs = None  # (ar_order,)

        # learned GRU params (single-layer GRU + linear head)
        self._rnn = None

        # training artifacts
        self._y_mean = 0.0
        self._y_std = 1.0
        self._y_train_norm = None  # (n,)
        self._residual_norm = None  # (n,)
        self._linear_fitted = None  # (n,)
        self._seq_len = None  # residual window length used to train GRU
        self._res_window_init = None  # (seq_len,) last residual window from training
        self._ar_buffer_init = None  # (max(1, ar_order),) last y_norm lags from training

        self.train_loss_history = []

    # ---------- feature builders ----------

    def _fourier_feats(self, t_idx):
        if self.fourier_order == 0:
            return np.empty(0, dtype=float)
        k = np.arange(1, self.fourier_order + 1, dtype=float)
        ang = 2.0 * np.pi * (t_idx / self.seasonal_period) * k
        return np.concatenate([np.cos(ang), np.sin(ang)], axis=0)  # length 2*order

    def _trend_val(self, t_idx):
        if self.trend_degree < 0:
            return 0.0
        return sum(self._trend_coefs[d] * (t_idx**d) for d in range(self.trend_degree + 1))

    def _prepare_design_matrix(self, y_norm):
        n = len(y_norm)
        t = np.arange(n, dtype=float)

        # Fourier block
        if self.fourier_order > 0:
            k = np.arange(1, self.fourier_order + 1, dtype=float)[None, :]
            ang = 2.0 * np.pi * (t[:, None] / self.seasonal_period) * k
            cos_terms = np.cos(ang)
            sin_terms = np.sin(ang)
            S = np.concatenate([cos_terms, sin_terms], axis=1)  # (n, 2*order)
        else:
            S = np.empty((n, 0), dtype=float)

        # Trend block
        if self.trend_degree >= 0:
            T = np.vstack([t**d for d in range(self.trend_degree + 1)]).T  # (n, deg+1)
        else:
            T = np.empty((n, 0), dtype=float)

        # AR block
        if self.ar_order > 0:
            A = np.zeros((n, self.ar_order), dtype=float)
            A[:] = np.nan
            for lag in range(1, self.ar_order + 1):
                A[lag:, lag - 1] = y_norm[:-lag]
            # fill NaNs with first value to avoid dropping rows
            A[np.isnan(A)] = y_norm[0]
        else:
            A = np.empty((n, 0), dtype=float)

        X = np.concatenate([S, T, A], axis=1)
        return X

    # ---------- GRU helpers ----------

    def _init_gru(self):
        rs = np.random.RandomState(self.seed)
        H = self.hidden_size
        # Single input scalar per step (residual)
        W_z = rs.randn(H, 1) * 0.1
        U_z = rs.randn(H, H) * 0.1
        b_z = np.zeros(H)
        W_r = rs.randn(H, 1) * 0.1
        U_r = rs.randn(H, H) * 0.1
        b_r = np.zeros(H)
        W_h = rs.randn(H, 1) * 0.1
        U_h = rs.randn(H, H) * 0.1
        b_h = np.zeros(H)
        V_o = rs.randn(H) * 0.1
        c_o = 0.0
        return {
            "W_z": W_z,
            "U_z": U_z,
            "b_z": b_z,
            "W_r": W_r,
            "U_r": U_r,
            "b_r": b_r,
            "W_h": W_h,
            "U_h": U_h,
            "b_h": b_h,
            "V_o": V_o,
            "c_o": c_o,
            "H": H,
        }

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    def _gru_forward_once(self, rnn, seq):
        """
        Run GRU over a residual window (seq: shape (L,)), return final hidden h and output y_pred.
        """
        W_z, U_z, b_z = rnn["W_z"], rnn["U_z"], rnn["b_z"]
        W_r, U_r, b_r = rnn["W_r"], rnn["U_r"], rnn["b_r"]
        W_h, U_h, b_h = rnn["W_h"], rnn["U_h"], rnn["b_h"]
        V_o, c_o = rnn["V_o"], rnn["c_o"]
        H = rnn["H"]

        h = np.zeros(H, dtype=float)
        for x in seq:
            x1 = np.array([x], dtype=float)  # (1,)
            z = self._sigmoid(W_z @ x1 + U_z @ h + b_z)  # (H,)
            r = self._sigmoid(W_r @ x1 + U_r @ h + b_r)  # (H,)
            h_tilde = self._tanh(W_h @ x1 + U_h @ (r * h) + b_h)  # (H,)
            h = z * h + (1.0 - z) * h_tilde
        y_pred = V_o.dot(h) + c_o
        return h, float(y_pred)

    # ---------- fit / predict ----------

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < max(3, self.ar_order + 1):
            raise ValueError("Series too short for configuration.")

        # normalize
        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 1e-8 else 1.0
        y_norm = (y - self._y_mean) / self._y_std
        self._y_train_norm = y_norm.copy()

        # linear fit
        X = self._prepare_design_matrix(y_norm)
        beta, *_ = np.linalg.lstsq(X, y_norm, rcond=None)
        p = 2 * self.fourier_order
        q = self.trend_degree + 1
        r = self.ar_order
        self._fourier_coefs = beta[:p] if p > 0 else np.zeros(0)
        self._trend_coefs = beta[p : p + q] if q > 0 else np.zeros(0)
        self._ar_coefs = beta[p + q : p + q + r] if r > 0 else np.zeros(0)

        linear_fitted = X @ beta
        self._linear_fitted = linear_fitted
        residual = y_norm - linear_fitted
        self._residual_norm = residual

        # residual windows for GRU
        seq_len = max(1, self.ar_order)  # tie window to AR depth
        self._seq_len = seq_len
        X_res = []
        Y_res = []
        for t in range(seq_len - 1, n - 1):
            X_res.append(residual[t - seq_len + 1 : t + 1])
            Y_res.append(residual[t + 1])
        X_res = np.array(X_res, dtype=float)  # (N_s, L)
        Y_res = np.array(Y_res, dtype=float)  # (N_s,)

        # initialize GRU
        rnn = self._init_gru()

        # simple full-batch gradient descent through time (coarse but OK)
        lr = self.lr
        H = rnn["H"]
        for epoch in range(self.epochs):
            d = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0 for k, v in rnn.items()}
            total_loss = 0.0

            for i in range(X_res.shape[0]):
                seq = X_res[i]
                target = Y_res[i]

                # forward + store for BPTT
                W_z, U_z, b_z = rnn["W_z"], rnn["U_z"], rnn["b_z"]
                W_r, U_r, b_r = rnn["W_r"], rnn["U_r"], rnn["b_r"]
                W_h, U_h, b_h = rnn["W_h"], rnn["U_h"], rnn["b_h"]
                V_o, c_o = rnn["V_o"], rnn["c_o"]

                hs = [np.zeros(H, dtype=float)]
                zs, rs, hts, xs = [], [], [], []
                h = hs[0]
                for x in seq:
                    x1 = np.array([x], dtype=float)
                    z = self._sigmoid(W_z @ x1 + U_z @ h + b_z)
                    r = self._sigmoid(W_r @ x1 + U_r @ h + b_r)
                    h_tilde = self._tanh(W_h @ x1 + U_h @ (r * h) + b_h)
                    h = z * h + (1.0 - z) * h_tilde

                    xs.append(x1)
                    zs.append(z)
                    rs.append(r)
                    hts.append(h_tilde)
                    hs.append(h)

                y_pred = V_o.dot(h) + c_o
                err = y_pred - target
                total_loss += 0.5 * err * err

                # backprop output
                d["V_o"] += err * h
                d["c_o"] += err
                dh_next = err * V_o

                # BPTT
                for t in reversed(range(len(seq))):
                    h_prev = hs[t]
                    h_curr = hs[t + 1]
                    z = zs[t]
                    r = rs[t]
                    h_tilde = hts[t]
                    x1 = xs[t]

                    # h = z * h_prev + (1 - z) * h_tilde
                    dh = dh_next.copy()
                    dh_prev = z * dh
                    dh_tilde = (1.0 - z) * dh
                    dz = (h_prev - h_tilde) * dh

                    # activations
                    dz_net = dz * z * (1.0 - z)
                    dh_tilde_net = dh_tilde * (1.0 - h_tilde**2)

                    # r gate grad via candidate path
                    dr_from_ht = (rnn["U_h"].T @ dh_tilde_net) * h_prev
                    dr_net = dr_from_ht * r * (1.0 - r)

                    # params grads
                    d["W_z"] += dz_net[:, None] * x1[None, :]
                    d["U_z"] += dz_net[:, None] * h_prev[None, :]
                    d["b_z"] += dz_net

                    d["W_r"] += dr_net[:, None] * x1[None, :]
                    d["U_r"] += dr_net[:, None] * h_prev[None, :]
                    d["b_r"] += dr_net

                    d["W_h"] += dh_tilde_net[:, None] * x1[None, :]
                    d["U_h"] += dh_tilde_net[:, None] * (r * h_prev)[None, :]
                    d["b_h"] += dh_tilde_net

                    # to previous h
                    dh_prev += rnn["U_z"].T @ dz_net
                    dh_prev += rnn["U_r"].T @ dr_net
                    dh_prev += r * (rnn["U_h"].T @ dh_tilde_net)
                    dh_next = dh_prev

            # update step (average)
            N = X_res.shape[0]
            for k in [
                "W_z",
                "U_z",
                "b_z",
                "W_r",
                "U_r",
                "b_r",
                "W_h",
                "U_h",
                "b_h",
                "V_o",
                "c_o",
            ]:
                rnn[k] -= (lr / max(1, N)) * d[k]
            self.train_loss_history.append(float(total_loss) / max(1, N))

        self._rnn = rnn

        # buffers for inference
        # last residual window (length seq_len)
        self._res_window_init = residual[-self._seq_len :].copy()
        # last AR buffer: last max(1, ar_order) normalized values
        L = max(1, self.ar_order)
        self._ar_buffer_init = y_norm[-L:].copy()
        return self

    def _linear_forecast_norm(self, current_index, ar_buffer):
        # Fourier
        fourier = 0.0
        if self.fourier_order > 0 and self._fourier_coefs.size > 0:
            k = np.arange(1, self.fourier_order + 1, dtype=float)
            ang = 2.0 * np.pi * (current_index / self.seasonal_period) * k
            cos_vals = np.cos(ang)
            sin_vals = np.sin(ang)
            fourier = float(
                cos_vals.dot(self._fourier_coefs[: self.fourier_order])
                + sin_vals.dot(self._fourier_coefs[self.fourier_order : 2 * self.fourier_order])
            )

        # Trend
        trend = 0.0
        if self._trend_coefs is not None and self._trend_coefs.size > 0:
            trend = sum(self._trend_coefs[d] * (current_index**d) for d in range(self.trend_degree + 1))

        # AR
        ar_part = 0.0
        if self.ar_order > 0 and self._ar_coefs.size > 0:
            # ar_buffer[-1] is most recent actual y_norm
            for lag in range(1, self.ar_order + 1):
                ar_part += self._ar_coefs[lag - 1] * ar_buffer[-lag]

        return fourier + trend + ar_part

    def predict(self, h):
        if self._y_train_norm is None or self._rnn is None:
            raise RuntimeError("Call fit() before predict().")
        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        n_train = len(self._y_train_norm)
        ar_buffer = self._ar_buffer_init.copy()
        res_window = self._res_window_init.copy()

        preds_norm = []
        for step in range(h):
            t_idx = n_train + step
            lin = self._linear_forecast_norm(t_idx, ar_buffer)
            # GRU over residual window -> next residual prediction
            _, res_pred = self._gru_forward_once(self._rnn, res_window)
            y_pred_norm = lin + res_pred

            preds_norm.append(y_pred_norm)

            # roll buffers
            if self.ar_order > 0:
                ar_buffer = np.append(ar_buffer, y_pred_norm)[-max(1, self.ar_order) :]
            res_window = np.append(res_window, res_pred)[-self._seq_len :]

        preds_norm = np.array(preds_norm, dtype=float)
        return preds_norm * self._y_std + self._y_mean


# ============ AutoHybridForecaster ==============
class AutoHybridForecaster:
    """
    Validation tuner for HybridForecastNet.
    Accepts `season_length` as an alias for `seasonal_period`.
    """

    def __init__(
        self,
        seasonal_period=24,
        season_length=None,  # alias
        candidate_fourier=(0, 3, 6),
        candidate_trend=(0, 1),
        candidate_ar=(0, 3, 5),
        candidate_hidden=(8, 16, 32),
        rnn_layers=1,
        epochs=100,
        lr=0.01,
        val_ratio=0.2,
        seed=123,
    ):
        self.seasonal_period = int(seasonal_period if season_length is None else season_length)
        self.candidate_fourier = tuple(candidate_fourier)
        self.candidate_trend = tuple(candidate_trend)
        self.candidate_ar = tuple(candidate_ar)
        self.candidate_hidden = tuple(candidate_hidden)
        self.rnn_layers = int(rnn_layers)
        self.epochs = int(max(1, epochs))
        self.lr = float(lr)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)

        self.best_model = None
        self.best_config = None
        self.best_val_mse = None

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        n = len(y)
        split = max(1, int(n * (1.0 - self.val_ratio)))
        if split < 5 or n - split < 5:
            raise ValueError("Series too short for validation split.")

        y_train = y[:split]
        y_val = y[split:]

        best_mse = np.inf
        best_model = None
        best_cfg = None

        for fo in self.candidate_fourier:
            for td in self.candidate_trend:
                for ar in self.candidate_ar:
                    for hs in self.candidate_hidden:
                        try:
                            model = HybridForecastNet(
                                seasonal_period=self.seasonal_period,
                                fourier_order=fo,
                                trend_degree=td,
                                ar_order=ar,
                                hidden_size=hs,
                                rnn_layers=self.rnn_layers,
                                epochs=self.epochs,
                                lr=self.lr,
                                seed=self.seed,
                            )
                            model.fit(y_train)
                            preds = model.predict(len(y_val))  # <- fixed: model now has needed state
                            mse = float(np.mean((preds - y_val) ** 2))
                        except Exception:
                            continue
                        if mse < best_mse:
                            best_mse = mse
                            best_model = model
                            best_cfg = dict(
                                fourier_order=fo,
                                trend_degree=td,
                                ar_order=ar,
                                hidden_size=hs,
                            )

        if best_model is None:
            raise RuntimeError("AutoHybridForecaster failed to find a valid configuration.")

        # Refit best on full data to finalize
        final = HybridForecastNet(
            seasonal_period=self.seasonal_period,
            fourier_order=best_cfg["fourier_order"],
            trend_degree=best_cfg["trend_degree"],
            ar_order=best_cfg["ar_order"],
            hidden_size=best_cfg["hidden_size"],
            rnn_layers=self.rnn_layers,
            epochs=self.epochs,
            lr=self.lr,
            seed=self.seed,
        ).fit(y)

        self.best_model = final
        self.best_config = best_cfg
        self.best_val_mse = best_mse
        return self

    def predict(self, h):
        if self.best_model is None:
            raise RuntimeError("Call fit() first.")
        return self.best_model.predict(h)


# ===================== MELDForecaster =====================
class MELDForecaster:
    """
    MELD = Multiscale Embedding with Learned Dynamics (kernelized) + adaptive Analog blend.

    Ideas combined:
      - Multiscale, causal moving-average embeddings (wavelet-ish, but causal & simple).
      - Nonlinear feature lift via Random Fourier Features (RBF kernel approximation).
      - Linear dynamics in feature space (ridge-regularized regression).
      - Instance-based analog correction (kNN on lifted features), blended adaptively by
        a confidence weight derived from neighbor distances.

    Goal: high accuracy on nonlinear series; speed is secondary.
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
        Parameters
        ----------
        lags : int
            Number of recent points used at each scale to define the state.
        scales : tuple of int
            Causal moving-average windows (in time steps) to create multiscale views.
            Always include 1 to keep the raw scale.
        rff_features : int
            Number of random Fourier features (RBF approximation).
        lengthscale : float
            RBF lengthscale for RFF (larger -> smoother features).
        ridge : float
            Ridge regularization strength for the linear model in feature space.
        knn : int
            Number of nearest analogs (in lifted feature space) for residual correction.
            Set 0 to disable analog blending.
        blend_strength : float
            Controls how aggressively we trust analogs when they’re very close (>=0).
            Larger = stronger pull toward analog average when distances are small.
        random_state : int or None
            Seed for reproducibility of RFF sampling.
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
        Fit MELD on y_0..y_{N-1} to predict y_{t+1} from state at t (t >= lags-1).
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
        Iterative forecasting for h steps.

        Parameters
        ----------
        h : int
            Horizon length.
        start_values : optional np.ndarray
            If provided, used as the starting tail (length >= lags). If None, uses the
            fitted series self._y as the starting context.

        Returns
        -------
        np.ndarray of length h
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
    Simple validation-based tuner for MELDForecaster hyperparameters.

    Grid examples (tweak freely for your data/compute budget):
      - lags_grid: (8, 12, 16)
      - scales_grid: ((1,3,7), (1,2,4,8))
      - rff_features_grid: (64, 128, 256)
      - lengthscales: (1.5, 3.0, 6.0)
      - ridges: (1e-3, 1e-2, 1e-1)
      - knns: (0, 3, 7)
      - blend_strengths: (0.5, 1.0, 2.0)
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
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.predict(h, start_values=start_values)


# ================= KNNForecaster =================
class KNNForecaster:
    """
    K-Nearest Neighbors Analog Forecaster.
    Finds past patterns similar to the recent history and predicts using their outcomes.
    """

    def __init__(self, window=8, k=3):
        """
        Parameters
        ----------
        window : int
            Length of the history window (embedding dimension) to use for finding analogs.
        k : int
            Number of nearest neighbors to average for forecasting.
        """
        self.window = int(window)
        self.k = int(k)
        self.data = None  # stores the fitted time series
        self._X = None  # matrix of shape (n_windows, window) of past subsequences
        self._y_next = None  # array of next values corresponding to each subsequence in _X

    def fit(self, y):
        """
        Store the time series and precompute windowed subsequences for neighbor search.
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

    def _predict_one(self, current_window):
        """
        Internal helper to predict the next value given the current window (length = self.window).
        Uses the precomputed library self._X and self._y_next.
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

    def predict(self, h, start_values=None):
        """
        Iteratively predict h time steps into the future.

        Parameters
        ----------
        h : int
            Forecast horizon (number of future points to predict).
        start_values : array-like (optional)
            An optional starting window of length = self.window. If None, uses the last
            'window' points from the fitted data as the starting state.

        Returns
        -------
        preds : np.ndarray
            Array of length h with the forecasted values.
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
    Automatic tuner for KNNForecaster hyperparameters (window length and k neighbors).
    """

    def __init__(self, window_grid=(4, 8, 12), k_grid=(1, 3, 5), metric="mae"):
        """
        Parameters
        ----------
        window_grid : iterable of int
            Candidate values for the window (embedding length) to try.
        k_grid : iterable of int
            Candidate values for number of neighbors to try.
        metric : str, "mae" or "rmse"
            Metric to optimize on validation set: mean absolute error or root mean squared error.
        """
        self.window_grid = list(window_grid)
        self.k_grid = list(k_grid)
        self.metric = metric.lower()
        self.model_ = None  # will hold the final fitted KNNForecaster
        self.best_ = None  # dict with best config and validation score

    def fit(self, y, val_fraction=0.25):
        """
        Fit KNNForecaster with best hyperparameters found via validation.
        Splits the series into training and validation, searches grid, and refits on full data.
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

        # Simple error functions (define internally for convenience)
        def mae(a, b):
            return np.mean(np.abs(a - b))

        def rmse(a, b):
            return np.sqrt(np.mean((a - b) ** 2))

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

    def predict(self, h):
        """
        Forecast using the best-found KNNForecaster. `fit()` must be called first.
        """
        if self.model_ is None:
            raise RuntimeError("AutoKNN is not fitted yet.")
        return self.model_.predict(h)


# ================= PALF =================
class PALF:
    """
    Proximal Aggregation Lag Forecaster (configured model)
    """

    def __init__(
        self,
        p=8,
        penalty="huber",
        decay_param=5.0,
        huber_delta=1.0,
        pinball_tau=0.5,
        level_penalty="l2",
        level_weight=1.0,
        irregular_timestamps=None,
    ):
        self.p = int(p)
        self.penalty = penalty
        self.decay_param = float(decay_param)
        self.huber_delta = float(huber_delta)
        self.pinball_tau = float(pinball_tau)
        self.level_penalty = level_penalty
        self.level_weight = float(level_weight)
        self.irregular_timestamps = irregular_timestamps

        self.mu_ = None
        self.y_ = None

    def _lag_weights(self, t_index):
        if self.irregular_timestamps is None:
            return np.array(
                [np.exp(-(i - 1) / self.decay_param) for i in range(1, self.p + 1)],
                dtype=float,
            )
        ts = self.irregular_timestamps
        t0 = ts[t_index]
        gaps = np.array([t0 - ts[t_index - i] for i in range(1, self.p + 1)], dtype=float)
        return np.exp(-gaps / max(self.decay_param, 1e-9))

    def _objective_factory(self, anchors, weights, level_anchor):
        kind = self.penalty
        delta = self.huber_delta
        tau = self.pinball_tau
        a_vals = np.asarray(anchors, float)
        w = np.asarray(weights, float)

        def J(z):
            r = z - a_vals
            val = np.sum(w * _penalty_value(r, kind, delta, tau))
            if level_anchor is not None and self.level_weight > 0.0:
                val += self.level_weight * _penalty_value(z - level_anchor, self.level_penalty, delta, 0.5)
            return float(val)

        return J

    def _solve_argmin(self, anchors, weights, level_anchor):
        anchors = np.asarray(anchors, float)
        weights = np.asarray(weights, float)

        if self.penalty == "l1" and (self.level_weight == 0 or self.level_penalty == "l1"):
            vals = anchors.copy()
            w = weights.copy()
            if self.level_weight > 0 and level_anchor is not None:
                vals = np.append(vals, level_anchor)
                w = np.append(w, self.level_weight)
            return _weighted_quantile(vals, w, 0.5)

        if self.penalty == "pinball" and (self.level_weight == 0 or self.level_penalty == "pinball"):
            tau = self.pinball_tau
            vals = anchors.copy()
            w = weights.copy()
            if self.level_weight > 0 and level_anchor is not None:
                vals = np.append(vals, level_anchor)
                w = np.append(w, self.level_weight)
            return _weighted_quantile(vals, w, tau)

        vmin = np.min(anchors)
        vmax = np.max(anchors)
        std = np.std(anchors) if anchors.size > 1 else 1.0
        a = vmin - 3 * std - 1.0
        b = vmax + 3 * std + 1.0
        J = self._objective_factory(anchors, weights, level_anchor)
        return _golden_section_minimize(J, a, b, tol=1e-6, max_iter=300)

    def fit(self, y):
        y = np.asarray(y, float)
        self.y_ = y.copy()
        self.mu_ = float(np.median(y[: max(self.p, 5)]))
        return self

    def _one_step(self, t):
        anchors = [self.y_[t - i] for i in range(1, self.p + 1)]
        w = self._lag_weights(t)
        yhat = self._solve_argmin(anchors, w, self.mu_)
        if self.level_penalty == "l2":
            alpha = min(1.0, max(0.0, 1.0 / (1.0 + self.level_weight))) if self.level_weight > 0 else 1.0
            self.mu_ = (1 - alpha) * self.mu_ + alpha * yhat
        elif self.level_penalty == "l1":
            self.mu_ = np.median([self.mu_, yhat])
        else:
            self.mu_ = 0.5 * self.mu_ + 0.5 * yhat
        return float(yhat)

    def predict(self, h):
        preds = []
        saved_y = self.y_.copy()
        saved_mu = self.mu_
        for _ in range(h):
            yhat = self._one_step(len(self.y_) - 1)
            preds.append(yhat)
            self.y_ = np.append(self.y_, yhat)
        self.y_ = saved_y
        self.mu_ = saved_mu
        return np.array(preds)


class AutoPALF:
    def __init__(
        self,
        p_candidates=(4, 8, 12),
        penalties=("huber", "l2", "l1", "pinball"),
        decay_params=(3.0, 5.0, 8.0),
        huber_deltas=(0.5, 1.0, 2.0),
        pinball_taus=(0.5,),
        level_penalty="l2",
        level_weight=1.0,
        irregular_timestamps=None,
    ):
        self.grid = dict(
            p=list(p_candidates),
            penalties=list(penalties),
            decay_params=list(decay_params),
            huber_deltas=list(huber_deltas),
            pinball_taus=list(pinball_taus),
        )
        self.level_penalty = level_penalty
        self.level_weight = level_weight
        self.irregular_timestamps = irregular_timestamps
        self.model_ = None
        self.best_ = None

    def fit(self, y, val_fraction=0.25, metric="mae"):
        y = np.asarray(y, float)
        n = len(y)
        n_val = max(16, int(n * val_fraction))
        split = n - n_val
        best = None
        best_score = np.inf
        for p in self.grid["p"]:
            for penalty in self.grid["penalties"]:
                for decay_param in self.grid["decay_params"]:
                    for delta in self.grid["huber_deltas"]:
                        for tau in self.grid["pinball_taus"]:
                            model = PALF(
                                p=p,
                                penalty=penalty,
                                decay_param=decay_param,
                                huber_delta=delta,
                                pinball_tau=tau,
                                level_penalty=self.level_penalty,
                                level_weight=self.level_weight,
                            )
                            model.fit(y[:split])
                            preds = []
                            truth = y[split:]
                            for t in range(split, n):
                                yhat = model._one_step(t - 1)
                                preds.append(yhat)
                                model.y_ = np.append(model.y_, y[t])
                                model.mu_ = 0.8 * model.mu_ + 0.2 * y[t]
                            preds = np.array(preds)
                            score = mae(truth, preds) if metric == "mae" else rmse(truth, preds)
                            if score < best_score:
                                best_score = score
                                best = model
        self.model_ = best
        self.best_ = {"val_score": best_score}
        return self

    def predict(self, h):
        return self.model_.predict(h)


# ============== NEO (Nonlinear Evolution Operator) ==============


class NEOForecaster:
    """
    Nonlinear Evolution Operator (NEO) forecaster with polynomial features over lags.
    """

    def __init__(self, lags=5, degree=2, window_size: Optional[int] = None):
        self.lags = int(lags)
        self.degree = int(degree)
        self.window_size = window_size
        self.coef_ = None
        self.last_window_ = None

    def _features(self, state: np.ndarray) -> np.ndarray:
        # state length = lags, ordered newest first
        feats = [1.0]
        n = len(state)
        # linear
        feats.extend(state.tolist())
        # quadratic
        if self.degree >= 2:
            for i in range(n):
                for j in range(i, n):
                    feats.append(state[i] * state[j])
        # cubic
        if self.degree >= 3:
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        feats.append(state[i] * state[j] * state[k])
        return np.array(feats, dtype=float)

    def fit(self, y: np.ndarray):
        y = np.asarray(y, float)
        N = len(y)
        m = self.lags
        if N < m + 1:
            raise ValueError(f"Need at least {m+1} points, got {N}.")
        start = 0 if (self.window_size is None or self.window_size >= N) else (N - self.window_size)
        start = max(start, m - 1)

        X_rows = []
        targets = []
        for t in range(start, N - 1):
            # state = [x_t, x_{t-1}, ..., x_{t-m+1}] newest first
            state = y[t - m + 1 : t + 1][::-1]
            X_rows.append(self._features(state))
            targets.append(y[t + 1])
        X = np.vstack(X_rows)
        Y = np.array(targets, dtype=float)
        # Least squares
        w, *_ = np.linalg.lstsq(X, Y, rcond=None)
        self.coef_ = w
        self.last_window_ = y[-m:].copy()
        return self

    def predict(self, h: int, start_values: Optional[np.ndarray] = None) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Fit the model before predicting.")
        m = self.lags
        if start_values is None:
            if self.last_window_ is None:
                raise RuntimeError("No start_values and no training window available.")
            window = self.last_window_.copy()
        else:
            start_values = np.asarray(start_values, float)
            if len(start_values) != m:
                raise ValueError(f"start_values must have length {m}.")
            window = start_values.copy()

        fcst = []
        for _ in range(h):
            state = window[::-1]
            phi = self._features(state)
            yhat = float(np.dot(self.coef_, phi))
            fcst.append(yhat)
            window[:-1] = window[1:]
            window[-1] = yhat
        return np.array(fcst)


class AutoNEO:
    """
    Automatic hyperparameter tuner for :class:`NEOForecaster`.

    This class performs a grid search over lag length, polynomial degree,
    and optional rolling window size fractions. A holdout validation set
    is taken from the end of the input series, candidate models are fit
    on the training segment, and scored on the validation block. The best
    configuration is then refit on the entire series.

    Parameters
    ----------
    lags_grid : iterable of int, default=(4, 8, 12)
        Candidate autoregressive lag lengths to evaluate.
    degree_grid : iterable of int, default=(1, 2, 3)
        Candidate polynomial degrees for the forecaster.
    window_fracs : iterable of {float, None}, default=(None, 0.5, 0.75)
        Fractions of the training set length to use for rolling windows.
        If ``None``, all available training data is used.
        If float ``w`` in (0, 1], window size is computed as
        ``max(lags + 8, int(len(y_train) * w))``.

    Attributes
    ----------
    model_ : NEOForecaster or None
        The final fitted forecaster using the best-found configuration.
    best_ : dict or None
        Dictionary with keys:
            - ``"config"`` : dict of best hyperparameters
            - ``"val_score"`` : float validation score
        Populated after :meth:`fit`.

    Notes
    -----
    - Validation split is a single holdout block; no cross-validation.
    - Candidates that raise exceptions during fitting or prediction
      are skipped silently.
    - Supports ``mae`` and ``rmse`` as validation metrics.
    """

    def __init__(
        self,
        lags_grid=(4, 8, 12),
        degree_grid=(1, 2, 3),
        window_fracs=(None, 0.5, 0.75),
    ):
        self.lags_grid = lags_grid
        self.degree_grid = degree_grid
        self.window_fracs = window_fracs
        self.model_: Optional[NEOForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction=0.25, metric="mae"):
        """
        Fit the auto-tuner on a univariate time series.

        The method splits the series into a training segment and a validation
        block. For each combination of hyperparameters, a candidate
        :class:`NEOForecaster` is fit on the training data, predictions are
        generated iteratively for the validation block, and the chosen metric
        is computed. The best configuration is refit on the full data.

        Parameters
        ----------
        y : array_like of shape (n_samples,)
            Univariate time series data.
        val_fraction : float, default=0.25
            Fraction of samples reserved for validation. The actual size is
            ``max(16, int(n_samples * val_fraction))`` to ensure at least 16
            validation points.
        metric : {"mae", "rmse"}, default="mae"
            Scoring metric used for validation selection.

        Returns
        -------
        self : AutoNEO
            The fitted instance with :attr:`model_` and :attr:`best_` set.
        """

        y = np.asarray(y, float)
        N = len(y)
        n_val = max(16, int(N * val_fraction))
        split = N - n_val
        y_train = y[:split]
        y_val = y[split:]

        best_score = np.inf
        best_model = None
        best_conf = None

        for lags in self.lags_grid:
            for degree in self.degree_grid:
                for wfrac in self.window_fracs:
                    window_size = None
                    if wfrac is not None:
                        window_size = max(lags + 8, int(len(y_train) * float(wfrac)))
                    try:
                        neo = NEOForecaster(lags=lags, degree=degree, window_size=window_size)
                        neo.fit(y_train)
                        # Predict entire validation block iteratively from end of train
                        start_vals = y_train[-lags:]
                        preds = neo.predict(len(y_val), start_values=start_vals)
                        score = mae(y_val, preds) if metric == "mae" else rmse(y_val, preds)
                        if score < best_score:
                            best_score = score
                            best_model = neo
                            best_conf = dict(lags=lags, degree=degree, window_size=window_size)
                    except Exception:
                        continue

        # Refit best model on full data
        final = NEOForecaster(**best_conf)
        final.fit(y)
        self.model_ = final
        self.best_ = dict(config=best_conf, val_score=best_score)
        return self

    def predict(self, h: int):
        """
        Generate forecasts using the best-fitted model.

        Parameters
        ----------
        h : int
            Forecast horizon, i.e., number of future steps to predict.

        Returns
        -------
        y_pred : ndarray of shape (h,)
            Forecasted values.
        """
        return self.model_.predict(h)


class AutoThetaAR:
    """
    Hybrid Theta-AR(1) forecaster:
      1) Optional deseasonalization (additive in log-space if multiplicative)
      2) Theta core: linear trend + SES on theta-line (theta=2), combined 50/50
      3) Residual AR(1) correction
    Fast: O(n) fit, O(h) predict.
    """

    def __init__(
        self,
        season_length=1,
        deseasonalize=True,
        multiplicative=None,
        theta=2.0,
        weight=0.5,
    ):
        """
        Parameters
        ----------
        season_length : int
            Seasonal period m (e.g., 7, 12, 24). Use 1 for non-seasonal.
        deseasonalize : bool
            If True and m>1, estimate/remove seasonality, then add back on predict.
        multiplicative : bool or None
            If None, auto: use multiplicative only if all y>0; we model multiplicative by log-transform.
        theta : float
            Theta coefficient (default 2.0).
        weight : float in [0,1]
            Weight for trend vs level in Theta combination. 0.5 is standard.
        """
        self.season_length = int(max(1, season_length))
        self.deseasonalize = bool(deseasonalize)
        self.multiplicative = multiplicative
        self.theta = float(theta)
        self.weight = float(weight)

        # learned params
        self._use_log = False
        self._n = None
        self.seasonal_indices = None  # length m (additive indices in working space)
        self.a = None  # trend intercept (on deseasonalized scale)
        self.b = None  # trend slope
        self.alpha = None  # SES smoothing on theta-line
        self.last_level = None  # L_T
        self.phi = 0.0  # AR(1) coefficient on SES residuals
        self.last_resid = 0.0  # last residual from SES fit
        self._fitted = False

    # ---------- helpers ----------
    def _estimate_seasonal_indices(self, yw):
        """Return length-m additive seasonal indices (mean 0) in the working space yw."""
        m = self.season_length
        n = len(yw)
        if n < m or m == 1:
            return np.zeros(m, dtype=float)

        # Simple seasonal estimate: deviations from a moving average of length m, padded at edges
        ma = np.convolve(yw, np.ones(m) / m, mode="valid")  # length n-m+1
        # Pad by repeating edge values to length n
        lead = (n - len(ma)) // 2
        lag = n - len(ma) - lead
        trend_est = np.concatenate([np.full(lead, ma[0]), ma, np.full(lag, ma[-1])])

        seasonal_dev = yw - trend_est  # additive in working space (log for multiplicative)
        # Average per seasonal position
        idx = np.arange(n) % m
        seas = np.zeros(m, dtype=float)
        for k in range(m):
            mask = idx == k
            if mask.any():
                seas[k] = seasonal_dev[mask].mean()
        # Normalize to mean 0 so season adds no net level
        seas -= seas.mean()
        return seas

    # ---------- API ----------
    def fit(self, y):
        y = np.asarray(y, dtype=float)
        n = y.size
        if n < 3:
            raise ValueError("Need at least 3 points to fit AutoThetaAR.")
        self._n = n

        # decide working space (log for multiplicative if possible)
        if self.multiplicative is None:
            self.multiplicative = np.all(y > 0)
        self._use_log = bool(self.multiplicative and self.deseasonalize and np.all(y > 0))
        yw = np.log(y) if self._use_log else y.copy()

        # seasonality (additive in working space)
        if self.deseasonalize and self.season_length > 1:
            self.seasonal_indices = self._estimate_seasonal_indices(yw)
            m = self.season_length
            seas = self.seasonal_indices[np.arange(n) % m]
            x = yw - seas
        else:
            self.seasonal_indices = np.zeros(self.season_length, dtype=float)
            x = yw

        # linear trend on deseasonalized x
        t = np.arange(n, dtype=float)
        tm = t.mean()
        xm = x.mean()
        S_tt = np.sum((t - tm) ** 2)
        if S_tt <= 1e-12:
            self.b = 0.0
            self.a = xm
        else:
            self.b = np.sum((t - tm) * (x - xm)) / S_tt
            self.a = xm - self.b * tm

        # theta-line and SES
        trend = self.a + self.b * t
        theta_line = self.theta * x + (1.0 - self.theta) * trend

        # pick alpha by SSE of one-step SES errors
        best_sse, best_alpha = np.inf, 0.2
        for alpha in np.linspace(0.01, 1.0, 100):
            L = theta_line[0]
            sse = 0.0
            for j in range(1, n):
                e = theta_line[j] - L
                sse += e * e
                L += alpha * e
            if sse < best_sse:
                best_sse, best_alpha = sse, alpha
        self.alpha = best_alpha

        # final level with best alpha
        L = theta_line[0]
        for j in range(1, n):
            e = theta_line[j] - L
            L += self.alpha * e
        self.last_level = L

        # residuals of SES and AR(1)
        res = []
        Lh = theta_line[0]
        for j in range(1, n):
            e = theta_line[j] - Lh
            res.append(e)
            Lh += self.alpha * e
        res = np.asarray(res, dtype=float)
        if res.size >= 2:
            num = np.dot(res[1:], res[:-1])
            den = np.dot(res[:-1], res[:-1])
            self.phi = float(num / den) if den > 1e-12 else 0.0
        else:
            self.phi = 0.0
        self.last_resid = float(res[-1]) if res.size else 0.0

        self._fitted = True
        return self

    def predict(self, h):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        n = self._n
        steps = np.arange(1, h + 1, dtype=float)
        t_last = n - 1

        # theta combination on deseasonalized scale
        trend_fc = self.a + self.b * (t_last + steps)  # linear extrapolation
        level_fc = np.full(h, self.last_level, dtype=float)  # SES flat
        base = self.weight * trend_fc + (1.0 - self.weight) * level_fc

        # AR(1) residual correction
        if abs(self.phi) > 1e-12 and abs(self.last_resid) > 0.0:
            base = base + (self.phi**steps) * self.last_resid

        # add back seasonality (additive in working space)
        if self.deseasonalize and self.season_length > 1:
            m = self.season_length
            seas_future = self.seasonal_indices[((t_last + steps).astype(int)) % m]
            base = base + seas_future

        # inverse transform
        if self._use_log:
            return np.exp(base)
        return base


class PolymathForecaster:
    """Polymath Forecaster: A generalized forecasting model with polynomial, Fourier,
    and other basis expansions over lagged states.
    """

    def __init__(
        self,
        lags=12,
        degree=2,
        period_length=None,
        fourier_terms=0,
        ridge=0.0,
        window_size=None,
    ):
        """
        lags: Number of lagged observations to use (the size of the state vector).
        degree: Polynomial degree for non-linear lag interactions.
        period_length: Length of seasonal period to model (e.g., 12 for monthly, 24 for daily/hourly).
        fourier_terms: Number of Fourier harmonics (sine/cosine pairs) to include for the given period.
                       If 0, no Fourier seasonal features are included.
        ridge: Ridge regularization strength (lambda). 0.0 means no ridge penalty (OLS).
        window_size: If set, model is fit only on the last `window_size` observations (rolling window).
                     If None, uses all available data.
        """
        self.lags = int(lags)
        self.degree = int(degree)
        self.period_length = period_length
        self.fourier_terms = int(fourier_terms)
        self.ridge = float(ridge)
        self.window_size = window_size
        # Fitted parameters:
        self.coef_ = None
        self.last_window_ = None
        self.last_time_index_ = None  # store last time index for forecasting if needed

    def _features(self, state: np.ndarray, t_idx: int = None) -> np.ndarray:
        """Construct feature vector from the current state (lagged values) and time index."""
        # state is expected to be length = lags, ordered [y_t, y_{t-1}, ..., y_{t-m+1}]
        m = len(state)
        feats = [1.0]  # intercept
        # Linear terms:
        feats.extend(state.tolist())
        # Polynomial interaction terms (quadratic, cubic, etc):
        if self.degree >= 2:
            for i in range(m):
                for j in range(i, m):
                    feats.append(state[i] * state[j])
        if self.degree >= 3:
            # For cubic terms
            for i in range(m):
                for j in range(i, m):
                    for k in range(j, m):
                        feats.append(state[i] * state[j] * state[k])
        # (Note: In practice, we might generate combinations using itertools for brevity)

        # Fourier seasonal terms:
        if self.period_length is not None and self.fourier_terms > 0:
            # We require a time index to compute sin/cos features.
            # t_idx is the time index for the "current" state (e.g., if state ends at time t_idx).
            # If not provided, we assume last_time_index_ is set and use that.
            if t_idx is None:
                if self.last_time_index_ is None:
                    raise RuntimeError("Time index not provided for Fourier features.")
                t_idx = self.last_time_index_
            # Compute Fourier series terms up to the specified number
            for k in range(1, self.fourier_terms + 1):
                angle = 2 * np.pi * k * (t_idx) / float(self.period_length)
                feats.append(np.cos(angle))
                feats.append(np.sin(angle))
        return np.array(feats, dtype=float)

    def fit(self, y: np.ndarray, time_index: np.ndarray = None):
        """Fit the Polymath model to the time series data."""
        y = np.asarray(y, dtype=float)
        N = len(y)
        m = self.lags
        if N < m + 1:
            raise ValueError(f"Need at least {m+1} data points to fit model (got {N}).")
        # Determine start index for training (if using window)
        start = 0 if (self.window_size is None or self.window_size >= N) else (N - self.window_size)
        start = max(start, m)  # ensure we have at least m lags to form first feature vector
        X_rows = []
        Y_vals = []
        # Loop through each time t where we can form a training pair (t has lag history and t+1 exists)
        for t in range(start, N - 1):
            state = y[t - m : t] if t - m >= 0 else y[0:t]  # last m values up to time t-1 (inclusive)
            if len(state) < m:
                # Pad the beginning if needed (this shouldn't happen after the max start logic)
                state = np.concatenate([np.zeros(m - len(state)), state])
            # state now contains [y_{t-m}, ..., y_{t-1}] but we want newest first for consistency with predict
            state = state[::-1]  # reverse to [y_{t-1}, y_{t-2}, ..., y_{t-m}]
            # Compute time index if provided (for Fourier features)
            t_idx = None
            if time_index is not None:
                t_idx = time_index[t]  # assuming time_index aligns with y array
            X_rows.append(self._features(state, t_idx=t_idx))
            Y_vals.append(
                y[t]
            )  # predict current value y_t from previous state (or we could predict y_{t+1} from state at t)
            # **Note**: We use one-step ahead scheme, so actually Y_vals should be y[t], features from t-1.
            # We may adjust indexing: to predict y[t] (target), use state ending at t-1.
            # For simplicity, this uses state (t-m ... t-1) to predict y_t.
        X = np.vstack(X_rows)
        Y_arr = np.array(Y_vals, dtype=float)
        # Solve regression: (X^T X + λI) w = X^T Y  (ridge) or standard if λ=0.
        if self.ridge is None or self.ridge == 0.0:
            # ordinary least squares via pseudo-inverse or np.linalg.lstsq
            w, *_ = np.linalg.lstsq(X, Y_arr, rcond=None)
        else:
            # Ridge regression closed-form
            n_feats = X.shape[1]
            A = X.T.dot(X) + self.ridge * np.eye(n_feats)
            b = X.T.dot(Y_arr)
            w = np.linalg.solve(A, b)
        self.coef_ = w
        # Store last window of actual data for recursive forecasting:
        self.last_window_ = y[-m:].copy()
        # If time indices given, store the last time index:
        if time_index is not None:
            self.last_time_index_ = time_index[-1]
        else:
            # If not provided, assume last index = N (we'll treat subsequent as N+1, N+2, ...)
            self.last_time_index_ = N - 1  # using 0-based index of array
        return self

    def predict(self, h: int) -> np.ndarray:
        """Generate forecasts for h future steps."""
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        m = self.lags
        # Start with the last observed window
        if self.last_window_ is None:
            raise RuntimeError("No last window available for prediction.")
        window = self.last_window_.copy()  # this is length m, with most recent observed values
        preds = []
        current_time = self.last_time_index_  # last seen time index
        for step in range(1, h + 1):
            current_time += 1  # time index for the step we're predicting
            state = window[::-1]  # reverse to get [y_{t-1}, ..., y_{t-m}] form (newest first)
            # Compute features for the current state and time
            feats = self._features(state, t_idx=current_time)
            # Predict next value
            y_hat = float(np.dot(self.coef_, feats))
            preds.append(y_hat)
            # Update the rolling window with this forecast
            window[:-1] = window[1:]
            window[-1] = y_hat
        return np.array(preds)


class AutoPolymath:
    """Automatic tuner for PolymathForecaster. Explores combinations of lags, degree,
    seasonal Fourier terms, and window sizes to find the best model.
    """

    def __init__(
        self,
        lags_grid=(6, 12, 24),
        degree_grid=(1, 2, 3),
        seasonality_options=(None,),
        fourier_terms_grid=(0, 2, 4),
        window_fracs=(None, 0.5),
    ):
        """
        lags_grid: iterable of lag lengths to try.
        degree_grid: iterable of polynomial degrees to try.
        seasonality_options: iterable of seasonal period lengths to consider (e.g., (None, 12, 24) to try no seasonality or an annual or daily cycle).
        fourier_terms_grid: iterable of how many Fourier pairs to use if seasonality is enabled. Ignored if seasonality is None.
        window_fracs: iterable of fractions of the training set to use as rolling window. None means use full training.
                      e.g., 0.5 means use last 50% of data for training.
        """
        self.lags_grid = lags_grid
        self.degree_grid = degree_grid
        self.seasonality_options = seasonality_options
        self.fourier_terms_grid = fourier_terms_grid
        self.window_fracs = window_fracs
        self.best_model_ = None
        self.best_config_ = None
        self.best_val_score_ = None

    def fit(
        self,
        y: np.ndarray,
        val_fraction=0.2,
        metric="mae",
        time_index: np.ndarray = None,
    ):
        """Find the best PolymathForecaster configuration via grid search on a validation holdout."""
        y = np.asarray(y, dtype=float)
        N = len(y)
        # Determine validation size (at least 16 points or val_fraction)
        n_val = max(16, int(N * val_fraction))
        train_end = N - n_val
        y_train = y[:train_end]
        y_val = y[train_end:]
        # If time indices provided, split them too
        time_idx_train = time_index[:train_end] if time_index is not None else None
        time_idx_val = time_index[train_end:] if time_index is not None else None
        best_score = float("inf")
        best_model = None
        best_conf = None

        # Iterate over all combinations of hyperparameters
        for m in self.lags_grid:
            for deg in self.degree_grid:
                for season in self.seasonality_options:
                    for K in self.fourier_terms_grid:
                        # If no seasonality (None), skip any K > 0
                        if season is None and K > 0:
                            continue
                        # If seasonality is specified and K=0, that's effectively no seasonal terms
                        # (We could allow that as another model, but it's redundant with season=None case)
                        if season is not None and K == 0:
                            continue
                        for wfrac in self.window_fracs:
                            # Determine rolling window size if fraction given
                            if wfrac is None:
                                w_size = None
                            else:
                                w_size = max(m + 8, int(len(y_train) * wfrac))
                            # Initialize model with this configuration
                            model = PolymathForecaster(
                                lags=m,
                                degree=deg,
                                period_length=season,
                                fourier_terms=K,
                                ridge=0.0,
                                window_size=w_size,
                            )
                            try:
                                model.fit(y_train, time_index=time_idx_train)
                                # Forecast the validation period
                                h = len(y_val)
                                model.last_time_index_ = (
                                    time_idx_train[-1] if time_idx_train is not None else (train_end - 1)
                                )
                                preds = model.predict(h)
                                # Compute error on validation
                                if metric == "mae":
                                    score = np.mean(np.abs(y_val - preds))
                                elif metric == "rmse":
                                    score = np.sqrt(np.mean((y_val - preds) ** 2))
                                else:
                                    raise ValueError("Unsupported metric. Use 'mae' or 'rmse'.")
                            except Exception as e:
                                # If model fails to fit or predict (e.g., due to singular matrix or other issues), skip it
                                continue
                            # Check if this is the best so far
                            if score < best_score:
                                best_score = score
                                best_model = model
                                best_conf = {
                                    "lags": m,
                                    "degree": deg,
                                    "period_length": season,
                                    "fourier_terms": K,
                                    "window_size": w_size,
                                }
        # Refit best model on full data (train + val) if found
        if best_model is None:
            raise RuntimeError("AutoPolymath was unable to fit any model on the data.")
        final_model = PolymathForecaster(**best_conf)
        final_model.fit(y, time_index=time_index)
        self.best_model_ = final_model
        self.best_config_ = best_conf
        self.best_val_score_ = best_score
        return self

    def predict(self, h: int) -> np.ndarray:
        """Generate h-step forecast using the best found model."""
        if self.best_model_ is None:
            raise RuntimeError("AutoPolymath has not been fit yet.")
        return self.best_model_.predict(h)


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


import numpy as np


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


import numpy as np


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
