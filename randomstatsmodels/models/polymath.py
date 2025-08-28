import numpy as np


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
