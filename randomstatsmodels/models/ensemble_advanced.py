"""
Advanced Ensemble Forecasters
==============================

Three distinct ensemble strategies that go beyond simple weighted averaging:

1. **StackedForecaster / AutoStacked**
   Meta-learning stacking: base models produce out-of-fold predictions on
   a validation window, then a ridge meta-learner is trained to optimally
   combine them.  Unlike simple weighting, the meta-learner can learn
   interactions (e.g. "trust model A for the first few steps but model B
   for longer horizons") because it sees the full prediction matrix.

2. **BaggedForecaster / AutoBagged**
   Block-bootstrap bagging for time series.  Creates multiple resampled
   training sets (preserving temporal structure via contiguous blocks),
   fits the same Auto* model on each, and averages predictions.  Reduces
   variance without increasing bias — the classic bias-variance trade-off.

3. **DynamicEnsemble / AutoDynamic**
   Horizon-adaptive weighting: model weights change at each forecast step
   based on how each model's rolling validation error evolves over the
   horizon.  Near-term forecasts may be dominated by one model while
   far-out forecasts are dominated by another.
"""

from typing import Optional, Dict, List, Iterable, Any
import numpy as np
from ..metrics import mae, rmse


# -----------------------------------------------------------------------
# Default base model pool (top 5 randomstatsmodels models by benchmark rank)
# -----------------------------------------------------------------------

def _default_pool():
    """Return the top 5 randomstatsmodels models by benchmark ranking."""
    from .hybrid import AutoHybridForecaster
    from .polymath import AutoPolymath
    from .neo import AutoNEO
    from .koopman import AutoKoopman
    from .ssa import AutoSSA

    return [
        (AutoHybridForecaster, {}),
        (AutoPolymath, {}),
        (AutoNEO, {"lags_grid": (4, 8), "degree_grid": (1, 2)}),
        (AutoKoopman, {"embed_grid": (4, 8, 12), "rank_grid": (0, 4),
                       "damping_grid": (0.98,), "deseason_grid": (True,),
                       "detrend_grid": (True,)}),
        (AutoSSA, {}),
    ]


# =====================================================================
# 1. Stacked Ensemble (Meta-Learner)
# =====================================================================

class StackedForecaster:
    """
    Stacking ensemble with a ridge meta-learner.

    Base models produce predictions on a validation fold, then a ridge
    regression learns the optimal linear combination.  The meta-learner
    also includes an intercept, allowing bias correction.

    Parameters
    ----------
    base_models : list of fitted forecaster instances
        Pre-fitted base models that have ``.predict(h)`` methods.
    meta_weights : ndarray of shape (n_models,)
        Learned meta-learner coefficients.
    meta_intercept : float
        Learned meta-learner intercept (bias correction).
    """

    def __init__(self, base_models: List[Any],
                 meta_weights: np.ndarray,
                 meta_intercept: float = 0.0):
        self.base_models = base_models
        self.meta_weights = np.asarray(meta_weights, float)
        self.meta_intercept = float(meta_intercept)
        self._fitted = True

    def predict(self, h: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")
        h = int(h)
        preds_matrix = []
        for m in self.base_models:
            try:
                p = np.asarray(m.predict(h), float)
                if len(p) < h:
                    p = np.pad(p, (0, h - len(p)), constant_values=np.nan)
                preds_matrix.append(p[:h])
            except Exception:
                preds_matrix.append(np.full(h, np.nan))

        preds_matrix = np.array(preds_matrix)  # (n_models, h)
        result = np.zeros(h)
        for t in range(h):
            col = preds_matrix[:, t]
            valid = ~np.isnan(col)
            if np.any(valid):
                result[t] = np.dot(self.meta_weights[valid], col[valid]) + self.meta_intercept
            else:
                result[t] = self.meta_intercept
        return result


class AutoStacked:
    """
    Auto-tuned stacking ensemble with ridge meta-learner.

    Fits each base model on a training fold, collects their multi-step
    predictions on a validation fold, then trains a ridge regression
    meta-learner to combine them.  Refits all base models on the full
    series for final forecasting.

    Parameters
    ----------
    base_pool : list of (AutoClass, kwargs) or None
        Base model pool. None = default pool of 7 fast models.
    ridge_grid : iterable of float
        Ridge regularisation values for the meta-learner.
    """

    def __init__(self, base_pool=None,
                 ridge_grid: Iterable[float] = (0.01, 0.1, 1.0, 10.0)):
        self.base_pool = base_pool
        self.ridge_grid = list(ridge_grid)
        self.model_: Optional[StackedForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction: float = 0.25,
            metric: str = "mae") -> "AutoStacked":
        y = np.asarray(y, float)
        N = len(y)
        n_val = max(6, int(N * val_fraction))
        split = N - n_val
        y_train, y_val = y[:split], y[split:]
        h_val = len(y_val)

        score_fn = mae if metric == "mae" else rmse
        pool = self.base_pool if self.base_pool is not None else _default_pool()

        # Step 1: Fit each base model on training data, predict validation
        val_preds = []  # (n_successful, h_val)
        successful = []

        for ModelClass, kwargs in pool:
            try:
                m = ModelClass(**kwargs)
                m.fit(y_train)
                p = np.asarray(m.predict(h_val), float)[:h_val]
                if len(p) < h_val:
                    p = np.pad(p, (0, h_val - len(p)), constant_values=np.nan)
                if np.any(np.isfinite(p)):
                    val_preds.append(p)
                    successful.append((ModelClass, kwargs))
            except Exception:
                continue

        if len(successful) < 1:
            raise RuntimeError("No base models could be fitted.")

        P = np.array(val_preds).T  # (h_val, n_models)
        # Replace NaN with column mean for ridge fitting
        for j in range(P.shape[1]):
            col = P[:, j]
            mask = np.isnan(col)
            if np.any(mask) and np.any(~mask):
                col[mask] = np.nanmean(col)
            elif np.all(mask):
                col[:] = np.mean(y_val)
            P[:, j] = col

        # Step 2: Ridge meta-learner — find best ridge alpha
        best_score = np.inf
        best_w = None
        best_b = 0.0
        best_ridge = 1.0

        # Add intercept column
        ones = np.ones((h_val, 1))
        P_aug = np.hstack([P, ones])

        for ridge in self.ridge_grid:
            try:
                n_feat = P_aug.shape[1]
                reg = ridge * np.eye(n_feat)
                reg[-1, -1] = 0.0  # don't regularise intercept
                w = np.linalg.solve(P_aug.T @ P_aug + reg, P_aug.T @ y_val)
                pred = P_aug @ w
                s = score_fn(y_val, pred)
                if np.isfinite(s) and s < best_score:
                    best_score = s
                    best_w = w[:-1]
                    best_b = w[-1]
                    best_ridge = ridge
            except Exception:
                continue

        if best_w is None:
            # Fallback: uniform
            n = len(successful)
            best_w = np.ones(n) / n
            best_b = 0.0
            best_score = float("inf")

        # Step 3: Refit all base models on full data
        final_models = []
        for ModelClass, kwargs in successful:
            try:
                m = ModelClass(**kwargs)
                m.fit(y)
                final_models.append(m)
            except Exception:
                final_models.append(None)

        # Remove failed models and adjust weights
        keep = [i for i, m in enumerate(final_models) if m is not None]
        final_models = [final_models[i] for i in keep]
        best_w = best_w[keep]

        self.model_ = StackedForecaster(final_models, best_w, best_b)
        self.best_ = {"config": {"n_models": len(final_models),
                                  "ridge": best_ridge},
                       "val_score": best_score}
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the model before predicting.")
        return self.model_.predict(h)


# =====================================================================
# 2. Bagged Ensemble (Block Bootstrap)
# =====================================================================

class BaggedForecaster:
    """
    Bootstrap-aggregated (bagged) forecaster.

    Averages predictions from multiple models, each trained on a
    different block-bootstrap resample of the series.

    Parameters
    ----------
    fitted_models : list of fitted forecaster instances
        Models trained on bootstrap samples.
    """

    def __init__(self, fitted_models: List[Any]):
        self.fitted_models = fitted_models
        self._fitted = True

    def predict(self, h: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")
        h = int(h)
        all_preds = []
        for m in self.fitted_models:
            try:
                p = np.asarray(m.predict(h), float)[:h]
                if len(p) == h and np.all(np.isfinite(p)):
                    all_preds.append(p)
            except Exception:
                continue
        if not all_preds:
            raise RuntimeError("All bagged models failed to predict.")
        return np.median(np.array(all_preds), axis=0)


def _block_bootstrap(y: np.ndarray, block_size: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Create a block-bootstrap resample of time series y."""
    n = len(y)
    n_blocks = max(1, (n + block_size - 1) // block_size)
    # Sample random block starting positions
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    blocks = [y[s: s + block_size] for s in starts]
    resampled = np.concatenate(blocks)[:n]
    return resampled


class AutoBagged:
    """
    Auto-tuned bagged ensemble via block bootstrap.

    Creates `n_bags` resampled training sets using block bootstrap
    (preserving temporal structure), fits a chosen Auto* model on
    each, and combines predictions via median.

    Parameters
    ----------
    model_class : class or None
        The Auto* model class to bag. None = AutoKoopman.
    model_kwargs : dict or None
        kwargs passed to each model instance.
    n_bags : int
        Number of bootstrap resamples.
    block_frac : float
        Block size as fraction of series length.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, model_class=None, model_kwargs=None,
                 n_bags: int = 15, block_frac: float = 0.1,
                 seed: int = 42):
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.n_bags = int(n_bags)
        self.block_frac = float(block_frac)
        self.seed = int(seed)
        self.model_: Optional[BaggedForecaster] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction: float = 0.25,
            metric: str = "mae") -> "AutoBagged":
        y = np.asarray(y, float)
        n = len(y)

        if self.model_class is None:
            from .koopman import AutoKoopman
            ModelClass = AutoKoopman
            kwargs = {"embed_grid": (4, 8, 12), "rank_grid": (0, 4),
                      "damping_grid": (0.98,), "deseason_grid": (True,),
                      "detrend_grid": (True,)}
        else:
            ModelClass = self.model_class
            kwargs = self.model_kwargs

        block_size = max(3, int(n * self.block_frac))
        rng = np.random.default_rng(self.seed)

        fitted_models = []
        # Always include one model trained on original data
        try:
            m0 = ModelClass(**kwargs)
            m0.fit(y)
            fitted_models.append(m0)
        except Exception:
            pass

        # Bag the rest
        for i in range(self.n_bags - 1):
            y_boot = _block_bootstrap(y, block_size, rng)
            try:
                m = ModelClass(**kwargs)
                m.fit(y_boot)
                fitted_models.append(m)
            except Exception:
                continue

        if not fitted_models:
            raise RuntimeError("No bagged models could be fitted.")

        self.model_ = BaggedForecaster(fitted_models)
        self.best_ = {"config": {"n_bags": len(fitted_models),
                                  "block_size": block_size,
                                  "model": ModelClass.__name__},
                       "val_score": float("nan")}
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the model before predicting.")
        return self.model_.predict(h)


# =====================================================================
# 3. Dynamic (Horizon-Adaptive) Ensemble
# =====================================================================

class DynamicEnsemble:
    """
    Horizon-adaptive ensemble where model weights vary by forecast step.

    Parameters
    ----------
    base_models : list of fitted forecaster instances
    weight_matrix : ndarray of shape (h_max, n_models)
        Per-step weights (row t = weights at forecast step t).
    """

    def __init__(self, base_models: List[Any],
                 weight_matrix: np.ndarray):
        self.base_models = base_models
        self.weight_matrix = np.asarray(weight_matrix, float)
        self._fitted = True

    def predict(self, h: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")
        h = int(h)
        n_models = len(self.base_models)

        preds_matrix = []
        for m in self.base_models:
            try:
                p = np.asarray(m.predict(h), float)[:h]
                if len(p) < h:
                    p = np.pad(p, (0, h - len(p)), constant_values=np.nan)
                preds_matrix.append(p)
            except Exception:
                preds_matrix.append(np.full(h, np.nan))

        preds_matrix = np.array(preds_matrix)  # (n_models, h)
        result = np.zeros(h)

        for t in range(h):
            # Use weight_matrix row, or last row if h > h_max
            w_idx = min(t, len(self.weight_matrix) - 1)
            w = self.weight_matrix[w_idx]
            col = preds_matrix[:, t]
            valid = ~np.isnan(col)
            if np.any(valid):
                w_valid = w[valid]
                w_sum = np.sum(w_valid)
                if w_sum > 1e-12:
                    w_valid = w_valid / w_sum
                else:
                    w_valid = np.ones(np.sum(valid)) / np.sum(valid)
                result[t] = np.dot(w_valid, col[valid])
            else:
                result[t] = 0.0
        return result


class AutoDynamic:
    """
    Auto-tuned dynamic (horizon-adaptive) ensemble.

    Learns per-step weights by evaluating how each base model's error
    changes across the forecast horizon on a validation set.  Models
    that are accurate at short horizons get more weight early; models
    that maintain accuracy at long horizons get more weight later.

    Parameters
    ----------
    base_pool : list of (AutoClass, kwargs) or None
        Base model pool. None = default pool.
    smoothing : float
        Exponential smoothing factor for the per-step weight curves.
    """

    def __init__(self, base_pool=None, smoothing: float = 0.3):
        self.base_pool = base_pool
        self.smoothing = float(smoothing)
        self.model_: Optional[DynamicEnsemble] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction: float = 0.25,
            metric: str = "mae") -> "AutoDynamic":
        y = np.asarray(y, float)
        N = len(y)
        n_val = max(6, int(N * val_fraction))
        split = N - n_val
        y_train, y_val = y[:split], y[split:]
        h_val = len(y_val)

        pool = self.base_pool if self.base_pool is not None else _default_pool()

        # Fit each base model on training data, predict full validation horizon
        val_preds = []
        successful = []

        for ModelClass, kwargs in pool:
            try:
                m = ModelClass(**kwargs)
                m.fit(y_train)
                p = np.asarray(m.predict(h_val), float)[:h_val]
                if len(p) < h_val:
                    p = np.pad(p, (0, h_val - len(p)), constant_values=np.nan)
                val_preds.append(p)
                successful.append((ModelClass, kwargs))
            except Exception:
                continue

        if len(successful) < 1:
            raise RuntimeError("No base models could be fitted.")

        val_preds = np.array(val_preds)  # (n_models, h_val)
        n_models = len(successful)

        # Compute per-step absolute errors for each model
        per_step_errors = np.abs(val_preds - y_val[np.newaxis, :])  # (n_models, h_val)

        # Replace NaN errors with large value
        per_step_errors = np.where(np.isfinite(per_step_errors),
                                   per_step_errors, 1e12)

        # Convert errors to weights: inverse error, smoothed across horizon
        # First, compute raw inverse-error weights per step
        weight_matrix = np.zeros((h_val, n_models))
        for t in range(h_val):
            errors_t = per_step_errors[:, t]
            errors_t = np.maximum(errors_t, 1e-12)
            inv_err = 1.0 / errors_t
            weight_matrix[t] = inv_err / np.sum(inv_err)

        # Smooth the weight curves across horizon with exponential smoothing
        alpha = self.smoothing
        for t in range(1, h_val):
            weight_matrix[t] = alpha * weight_matrix[t] + (1 - alpha) * weight_matrix[t - 1]
            # Re-normalise
            s = np.sum(weight_matrix[t])
            if s > 1e-12:
                weight_matrix[t] /= s

        # Compute validation score using dynamic weights
        dynamic_pred = np.zeros(h_val)
        for t in range(h_val):
            dynamic_pred[t] = np.dot(weight_matrix[t], val_preds[:, t])
        score_fn = mae if metric == "mae" else rmse
        val_score = score_fn(y_val, dynamic_pred)

        # Refit all base models on full data
        final_models = []
        for ModelClass, kwargs in successful:
            try:
                m = ModelClass(**kwargs)
                m.fit(y)
                final_models.append(m)
            except Exception:
                final_models.append(None)

        keep = [i for i, m in enumerate(final_models) if m is not None]
        final_models = [final_models[i] for i in keep]
        weight_matrix = weight_matrix[:, keep]

        self.model_ = DynamicEnsemble(final_models, weight_matrix)
        self.best_ = {"config": {"n_models": len(final_models),
                                  "smoothing": self.smoothing},
                       "val_score": val_score}
        return self

    def predict(self, h: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the model before predicting.")
        return self.model_.predict(h)
