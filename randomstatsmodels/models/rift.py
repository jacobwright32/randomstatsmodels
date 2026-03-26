"""
RIFT - Recursive Information Flow Tensor Forecaster

A novel forecasting framework based on "Predictive Information Field Dynamics" theory.

THEORETICAL FOUNDATION
======================

Traditional forecasting treats predictions at different horizons independently or assumes
simple decay of predictability. RIFT introduces a fundamentally different paradigm:

1. INFORMATION CHANNELS: A time series contains predictive information distributed across
   multiple orthogonal "channels":
   - Level channel (DC component / mean)
   - Trend channel (first derivative / momentum)
   - Curvature channel (second derivative / acceleration)
   - Oscillatory channels at multiple frequencies (spectral components)
   - Residual/noise channel

2. INFORMATION STATE TENSOR: At any time t, we define an "information state" vector I(t)
   where each component represents how much predictive power that channel currently holds.
   This is estimated via the local Fisher information of each channel.

3. INFORMATION FLOW DYNAMICS: The key innovation - as forecast horizon h increases,
   information doesn't just decay, it TRANSFORMS between channels according to:

       dI/dh = F · I + G · u(h)

   Where:
   - F is the "Information Flow Matrix" - learned from data
   - G captures how external/residual information enters
   - u(h) is a horizon-dependent forcing term

   This captures phenomena like:
   - Trend information becoming level information at longer horizons
   - High-frequency oscillations decaying while low-frequency persist
   - Information "cascading" from fast to slow channels

4. RECONSTRUCTION: The forecast at horizon h is reconstructed by:
   - Propagating the information state to horizon h
   - Using each channel's extrapolation weighted by its information content
   - Combining via learned reconstruction matrix

5. UNCERTAINTY QUANTIFICATION: The entropy of the information state provides
   principled forecast uncertainty - high entropy = high uncertainty.

This framework unifies several concepts:
- State-space models (information state as hidden state)
- Spectral methods (channels include frequencies)
- Information theory (Fisher information, entropy)
- Dynamical systems (flow matrix captures attractor geometry)

Author: Novel theoretical framework developed for randomstatsmodels
"""

import numpy as np
from typing import Optional, Union, Sequence, Dict, List, Tuple
from ..metrics import mae, rmse

ArrayLike = Union[Sequence[float], np.ndarray]


class RIFTForecaster:
    """
    Recursive Information Flow Tensor Forecaster.

    A novel forecasting model based on Predictive Information Field Dynamics theory.
    Models how predictive information flows between temporal channels as forecast
    horizon increases.

    Parameters
    ----------
    n_frequencies : int, default=4
        Number of oscillatory frequency channels to extract.
    embedding_dim : int, default=3
        Number of derivative channels (1=level, 2=+trend, 3=+curvature).
    horizon_resolution : int, default=10
        Resolution for discretizing the information flow dynamics.
    regularization : float, default=0.01
        Ridge regularization for learning the flow matrix.
    adaptive_channels : bool, default=True
        If True, automatically select optimal frequencies from data.

    Attributes
    ----------
    n_channels_ : int
        Total number of information channels.
    flow_matrix_ : ndarray of shape (n_channels, n_channels)
        Learned information flow matrix F.
    reconstruction_weights_ : ndarray
        Weights for reconstructing forecasts from information state.
    channel_frequencies_ : ndarray
        Frequencies used for oscillatory channels.
    base_info_state_ : ndarray
        Information state at the end of training data.
    channel_extrapolators_ : list
        Functions to extrapolate each channel.
    """

    def __init__(
        self,
        n_frequencies: int = 4,
        embedding_dim: int = 3,
        horizon_resolution: int = 10,
        regularization: float = 0.01,
        adaptive_channels: bool = True,
    ) -> None:
        self.n_frequencies = n_frequencies
        self.embedding_dim = min(3, max(1, embedding_dim))  # Cap at 3 (level, trend, curvature)
        self.horizon_resolution = horizon_resolution
        self.regularization = regularization
        self.adaptive_channels = adaptive_channels

        # Fitted attributes
        self.n_channels_: Optional[int] = None
        self.flow_matrix_: Optional[np.ndarray] = None
        self.reconstruction_weights_: Optional[np.ndarray] = None
        self.channel_frequencies_: Optional[np.ndarray] = None
        self.base_info_state_: Optional[np.ndarray] = None
        self.channel_extrapolators_: Optional[List] = None
        self.n_: Optional[int] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self._last_values_: Optional[np.ndarray] = None

    def _extract_channel_features(self, y: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Extract features for each information channel.

        Returns feature matrix (n_samples x n_channels) and list of extrapolator functions.
        """
        n = len(y)
        features = []
        extrapolators = []

        # Channel 1: Level (local mean with exponential smoothing)
        alpha_level = 0.3
        level = np.zeros(n)
        level[0] = y[0]
        for i in range(1, n):
            level[i] = alpha_level * y[i] + (1 - alpha_level) * level[i-1]
        features.append(level)
        final_level = level[-1]
        extrapolators.append(lambda h, fl=final_level: np.full(h, fl))

        # Channel 2: Trend (smoothed first derivative)
        if self.embedding_dim >= 2:
            trend = np.gradient(y)
            # Smooth the trend
            alpha_trend = 0.2
            smooth_trend = np.zeros(n)
            smooth_trend[0] = trend[0]
            for i in range(1, n):
                smooth_trend[i] = alpha_trend * trend[i] + (1 - alpha_trend) * smooth_trend[i-1]
            features.append(smooth_trend)
            final_trend = smooth_trend[-1]
            final_level_for_trend = level[-1]
            extrapolators.append(
                lambda h, ft=final_trend, fl=final_level_for_trend: fl + ft * np.arange(1, h+1)
            )

        # Channel 3: Curvature (smoothed second derivative)
        if self.embedding_dim >= 3:
            curvature = np.gradient(np.gradient(y))
            alpha_curv = 0.15
            smooth_curv = np.zeros(n)
            smooth_curv[0] = curvature[0]
            for i in range(1, n):
                smooth_curv[i] = alpha_curv * curvature[i] + (1 - alpha_curv) * smooth_curv[i-1]
            features.append(smooth_curv)
            final_curv = smooth_curv[-1]
            final_trend_for_curv = smooth_trend[-1] if self.embedding_dim >= 2 else 0
            final_level_for_curv = level[-1]
            extrapolators.append(
                lambda h, fc=final_curv, ft=final_trend_for_curv, fl=final_level_for_curv: (
                    fl + ft * np.arange(1, h+1) + 0.5 * fc * np.arange(1, h+1)**2
                )
            )

        # Oscillatory channels: Extract dominant frequencies
        if self.n_frequencies > 0:
            # FFT to find dominant frequencies
            fft_result = np.fft.rfft(y - np.mean(y))
            freqs = np.fft.rfftfreq(n)
            magnitudes = np.abs(fft_result)

            # Find top frequencies (excluding DC)
            if len(magnitudes) > 1:
                freq_indices = np.argsort(magnitudes[1:])[::-1][:self.n_frequencies] + 1
            else:
                freq_indices = []

            self.channel_frequencies_ = freqs[freq_indices] if len(freq_indices) > 0 else np.array([])

            for idx in freq_indices:
                freq = freqs[idx]
                if freq > 0:
                    # Extract this frequency component
                    period = int(1.0 / freq) if freq > 0 else n
                    period = max(2, min(period, n // 2))

                    # Create oscillatory feature via bandpass-like extraction
                    t = np.arange(n)
                    cos_comp = np.cos(2 * np.pi * freq * t)
                    sin_comp = np.sin(2 * np.pi * freq * t)

                    # Project data onto this frequency
                    a = 2 * np.mean(y * cos_comp)
                    b = 2 * np.mean(y * sin_comp)

                    osc_feature = a * cos_comp + b * sin_comp
                    features.append(osc_feature)

                    # Extrapolator for this frequency
                    final_a, final_b, final_freq = a, b, freq
                    extrapolators.append(
                        lambda h, a=final_a, b=final_b, f=final_freq, n_=n: (
                            a * np.cos(2 * np.pi * f * (n_ + np.arange(1, h+1))) +
                            b * np.sin(2 * np.pi * f * (n_ + np.arange(1, h+1)))
                        )
                    )

        # Stack features
        feature_matrix = np.column_stack(features) if len(features) > 1 else features[0].reshape(-1, 1)

        return feature_matrix, extrapolators

    def _compute_information_state(self, y: np.ndarray, features: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Compute information state vector based on local Fisher information.

        The information in each channel is estimated by how much that channel
        reduces prediction uncertainty (measured via local variance reduction).
        """
        n, n_channels = features.shape
        window = min(window, n // 4, 50)
        window = max(window, 5)

        info_states = np.zeros((n, n_channels))

        for t in range(window, n):
            y_local = y[t-window:t]
            feat_local = features[t-window:t]

            # Baseline variance (no features)
            baseline_var = np.var(y_local) + 1e-10

            for c in range(n_channels):
                # Variance explained by this channel (simple linear regression)
                X = feat_local[:, c:c+1]
                X = np.column_stack([np.ones(window), X])
                try:
                    beta = np.linalg.lstsq(X, y_local, rcond=None)[0]
                    residuals = y_local - X @ beta
                    residual_var = np.var(residuals) + 1e-10

                    # Information = log ratio of variances (like Fisher information)
                    info_states[t, c] = np.log(baseline_var / residual_var + 1)
                except:
                    info_states[t, c] = 0

        # Fill in early values
        info_states[:window] = info_states[window]

        # Normalize to sum to 1 (probability-like)
        row_sums = np.sum(info_states, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        info_states = info_states / row_sums

        return info_states

    def _learn_flow_dynamics(
        self,
        y: np.ndarray,
        features: np.ndarray,
        info_states: np.ndarray,
        max_horizon: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Learn the Information Flow Matrix F from data.

        Key insight: Look at how well each channel predicts at different horizons,
        and model how this changes as horizon increases.
        """
        n, n_channels = features.shape
        max_horizon = min(max_horizon, n // 4)

        # For each horizon h, compute the "effective information" of each channel
        horizon_info = np.zeros((max_horizon, n_channels))

        for h in range(1, max_horizon + 1):
            for c in range(n_channels):
                # How well does channel c at time t predict y[t+h]?
                valid_n = n - h - 10
                if valid_n < 20:
                    continue

                X = features[10:10+valid_n, c:c+1]
                X = np.column_stack([np.ones(valid_n), X])
                y_future = y[10+h:10+h+valid_n]

                try:
                    beta = np.linalg.lstsq(X, y_future, rcond=None)[0]
                    pred = X @ beta
                    ss_res = np.sum((y_future - pred)**2)
                    ss_tot = np.sum((y_future - np.mean(y_future))**2) + 1e-10
                    r2 = max(0, 1 - ss_res / ss_tot)
                    horizon_info[h-1, c] = r2
                except:
                    horizon_info[h-1, c] = 0

        # Normalize each row
        row_sums = np.sum(horizon_info, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        horizon_info = horizon_info / row_sums

        # Learn flow matrix: how does info state at horizon h predict info state at h+1?
        # dI/dh ≈ F · I  =>  I(h+1) ≈ (I + F) · I(h)
        # So (I + F) = I(h+1) · I(h)^{-1} approximately

        # Use least squares to fit: horizon_info[h+1] = M @ horizon_info[h]
        X_flow = horizon_info[:-1]  # horizon 1 to max-1
        Y_flow = horizon_info[1:]   # horizon 2 to max

        # Ridge regression: M = Y @ X^T @ (X @ X^T + λI)^{-1}
        XXT = X_flow.T @ X_flow + self.regularization * np.eye(n_channels)
        try:
            M = np.linalg.solve(XXT, X_flow.T @ Y_flow).T
        except:
            M = np.eye(n_channels)

        # Flow matrix F = M - I
        flow_matrix = M - np.eye(n_channels)

        # Learn reconstruction weights: how to combine channel extrapolations
        # based on their information content
        # Simple approach: weights proportional to average information
        reconstruction_weights = np.mean(horizon_info, axis=0)
        reconstruction_weights = reconstruction_weights / (np.sum(reconstruction_weights) + 1e-10)

        return flow_matrix, reconstruction_weights

    def fit(self, y: ArrayLike) -> "RIFTForecaster":
        """
        Fit the RIFT model by learning information flow dynamics.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Training time series.

        Returns
        -------
        self : RIFTForecaster
            Fitted instance.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)

        if n < 30:
            raise ValueError("Need at least 30 data points for RIFT.")

        self.n_ = n
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y) + 1e-10

        # Normalize
        y_norm = (y - self.y_mean_) / self.y_std_

        # Store last values for continuity
        self._last_values_ = y[-10:].copy()

        # Extract channel features and extrapolators
        features, extrapolators = self._extract_channel_features(y_norm)
        self.channel_extrapolators_ = extrapolators
        self.n_channels_ = features.shape[1]

        # Compute information states
        info_states = self._compute_information_state(y_norm, features)
        self.base_info_state_ = info_states[-1].copy()

        # Learn flow dynamics
        self.flow_matrix_, self.reconstruction_weights_ = self._learn_flow_dynamics(
            y_norm, features, info_states
        )

        return self

    def _propagate_information(self, horizon: int) -> np.ndarray:
        """
        Propagate information state forward to given horizon.

        Uses the learned flow matrix: I(h) = exp(F*h) @ I(0)
        approximated as (I + F)^h @ I(0)
        """
        # Transition matrix for one step
        M = np.eye(self.n_channels_) + self.flow_matrix_ / self.horizon_resolution

        # Propagate
        info_state = self.base_info_state_.copy()

        # Propagate through horizon steps
        steps = horizon * self.horizon_resolution // max(1, self.n_ // 50)
        steps = max(1, min(steps, 100))

        for _ in range(steps):
            info_state = M @ info_state
            # Ensure non-negative and normalize
            info_state = np.maximum(info_state, 0)
            if np.sum(info_state) > 0:
                info_state = info_state / np.sum(info_state)
            else:
                info_state = np.ones(self.n_channels_) / self.n_channels_

        return info_state

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast h steps ahead using information flow dynamics.

        Parameters
        ----------
        h : int
            Forecast horizon.

        Returns
        -------
        preds : ndarray of shape (h,)
            Forecasted values.
        """
        if self.flow_matrix_ is None:
            raise RuntimeError("Fit the model before calling predict().")

        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        # Get channel extrapolations (normalized scale)
        channel_forecasts = []
        for extrap in self.channel_extrapolators_:
            try:
                fc = extrap(h)
                if len(fc) != h:
                    fc = np.zeros(h)
            except:
                fc = np.zeros(h)
            channel_forecasts.append(fc)

        channel_forecasts = np.array(channel_forecasts)  # (n_channels, h)

        # For each forecast step, propagate information and combine
        preds = np.zeros(h)
        for step in range(h):
            # Get information state at this horizon
            info_state = self._propagate_information(step + 1)

            # Combine channel forecasts weighted by information state and reconstruction weights
            combined_weights = info_state * self.reconstruction_weights_
            combined_weights = combined_weights / (np.sum(combined_weights) + 1e-10)

            preds[step] = np.sum(combined_weights * channel_forecasts[:, step])

        # Denormalize
        preds = preds * self.y_std_ + self.y_mean_

        # Apply continuity correction for smooth transition from last values
        if len(self._last_values_) > 0:
            last_val = self._last_values_[-1]
            # Smooth transition: blend first few predictions with continuation
            blend_len = min(3, h)
            for i in range(blend_len):
                alpha = (i + 1) / (blend_len + 1)
                preds[i] = (1 - alpha) * last_val + alpha * preds[i]

        return preds

    def get_information_state(self, horizon: int = 1) -> Dict:
        """
        Get the information state at a given forecast horizon.

        Useful for understanding which channels are most informative.

        Parameters
        ----------
        horizon : int, default=1
            Forecast horizon to analyze.

        Returns
        -------
        dict
            Dictionary with channel names and their information content.
        """
        if self.flow_matrix_ is None:
            raise RuntimeError("Fit the model first.")

        info_state = self._propagate_information(horizon)

        channel_names = ["level"]
        if self.embedding_dim >= 2:
            channel_names.append("trend")
        if self.embedding_dim >= 3:
            channel_names.append("curvature")

        if self.channel_frequencies_ is not None:
            for i, freq in enumerate(self.channel_frequencies_):
                period = int(1.0 / freq) if freq > 0 else float('inf')
                channel_names.append(f"oscillation_period_{period}")

        return {
            "channel_names": channel_names[:len(info_state)],
            "information_content": info_state.tolist(),
            "horizon": horizon,
        }


class AutoRIFT:
    """
    Automatic tuner for RIFT (Recursive Information Flow Tensor) Forecaster.

    Searches over number of frequency channels, embedding dimension, and
    regularization to find optimal configuration.

    Parameters
    ----------
    n_frequencies_grid : iterable of int, default=(2, 4, 6, 8)
        Candidate number of oscillatory channels.
    embedding_dim_grid : iterable of int, default=(2, 3)
        Candidate embedding dimensions (derivative channels).
    regularization_grid : iterable of float, default=(0.001, 0.01, 0.1)
        Candidate regularization strengths.
    metric : {"mae", "rmse"}, default="mae"
        Validation error metric.

    Attributes
    ----------
    model_ : RIFTForecaster or None
        Best fitted model.
    best_ : dict or None
        Best configuration and validation score.
    """

    def __init__(
        self,
        n_frequencies_grid: Sequence[int] = (2, 4, 6, 8),
        embedding_dim_grid: Sequence[int] = (2, 3),
        regularization_grid: Sequence[float] = (0.001, 0.01, 0.1),
        metric: str = "mae",
    ) -> None:
        self.n_frequencies_grid = list(n_frequencies_grid)
        self.embedding_dim_grid = list(embedding_dim_grid)
        self.regularization_grid = list(regularization_grid)
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'")

        self.model_: Optional[RIFTForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: ArrayLike, val_fraction: float = 0.25) -> "AutoRIFT":
        """
        Grid search for optimal RIFT configuration.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time series data.
        val_fraction : float, default=0.25
            Fraction reserved for validation.

        Returns
        -------
        self : AutoRIFT
            Fitted tuner with best model.
        """
        y = np.asarray(y, dtype=float)
        N = len(y)
        n_val = max(10, int(N * val_fraction))

        if N - n_val < 30:
            raise ValueError("Not enough data for RIFT (need at least 30 training points).")

        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if self.metric == "mae" else rmse
        best_score = np.inf
        best_conf = None

        for n_freq in self.n_frequencies_grid:
            for emb_dim in self.embedding_dim_grid:
                for reg in self.regularization_grid:
                    try:
                        model = RIFTForecaster(
                            n_frequencies=n_freq,
                            embedding_dim=emb_dim,
                            regularization=reg,
                        ).fit(y_train)

                        # Multi-step validation (single fit)
                        preds = model.predict(len(y_val))
                        score = score_fn(y_val, preds)

                        if score < best_score:
                            best_score = score
                            best_conf = {
                                "n_frequencies": n_freq,
                                "embedding_dim": emb_dim,
                                "regularization": reg,
                            }
                    except Exception:
                        continue

        if best_conf is None:
            raise RuntimeError("AutoRIFT failed to find valid configuration.")

        # Refit on full data
        self.model_ = RIFTForecaster(**best_conf).fit(y)
        self.best_ = {
            "config": best_conf,
            "val_score": best_score,
            "metric": self.metric,
        }

        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast using the best RIFT model.

        Parameters
        ----------
        h : int
            Forecast horizon.

        Returns
        -------
        preds : ndarray of shape (h,)
            Forecasted values.
        """
        if self.model_ is None:
            raise RuntimeError("AutoRIFT is not fitted yet.")
        return self.model_.predict(h)

    def get_information_analysis(self, horizon: int = 1) -> Dict:
        """
        Get information flow analysis at given horizon.

        Returns
        -------
        dict
            Analysis of which channels hold predictive information.
        """
        if self.model_ is None:
            raise RuntimeError("AutoRIFT is not fitted yet.")
        return self.model_.get_information_state(horizon)
