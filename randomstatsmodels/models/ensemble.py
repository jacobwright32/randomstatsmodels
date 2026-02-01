import numpy as np
from typing import Optional, Union, Sequence, Dict, List, Any

from ..metrics import mae, rmse

ArrayLike = Union[Sequence[float], np.ndarray]


# ================= EnsembleForecaster =================
class EnsembleForecaster:
    """
    Ensemble forecaster that combines multiple base forecasters.

    Aggregates predictions from base models using specified weighting method.

    Parameters
    ----------
    base_models : list of forecaster instances
        Base forecasters to combine. Each must have fit() and predict() methods.
    weights : array-like of float or None, default=None
        Manual weights for each base model. If None, uses uniform weights.
    weighting : {"uniform", "manual"}, default="uniform"
        Weighting method. "uniform" uses equal weights, "manual" uses provided weights.

    Attributes
    ----------
    weights_ : ndarray
        Final weights after fitting.
    fitted_models_ : list
        List of fitted base model instances.
    """

    def __init__(
        self,
        base_models: List[Any],
        weights: Optional[ArrayLike] = None,
        weighting: str = "uniform",
    ) -> None:
        self.base_models = base_models
        self.weights = weights
        self.weighting = weighting.lower()

        if self.weighting not in ("uniform", "manual"):
            raise ValueError("weighting must be 'uniform' or 'manual'")
        if self.weighting == "manual" and weights is None:
            raise ValueError("Must provide weights when weighting='manual'")

        self.weights_: Optional[np.ndarray] = None
        self.fitted_models_: Optional[List[Any]] = None

    def fit(self, y: ArrayLike) -> "EnsembleForecaster":
        """
        Fit all base models to the time series.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Training time series.

        Returns
        -------
        self : EnsembleForecaster
            Fitted instance.
        """
        y = np.asarray(y, dtype=float)
        n_models = len(self.base_models)

        if n_models == 0:
            raise ValueError("Must provide at least one base model.")

        # Fit all base models
        self.fitted_models_ = []
        for model in self.base_models:
            # Clone the model by creating a new instance with same params
            # For simplicity, we'll just fit the original model
            fitted = model.fit(y)
            self.fitted_models_.append(fitted)

        # Set weights
        if self.weighting == "uniform":
            self.weights_ = np.ones(n_models) / n_models
        else:
            self.weights_ = np.asarray(self.weights, dtype=float)
            if len(self.weights_) != n_models:
                raise ValueError(
                    f"Number of weights ({len(self.weights_)}) must match "
                    f"number of models ({n_models})."
                )
            # Normalize weights
            self.weights_ = self.weights_ / np.sum(self.weights_)

        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast by weighted average of base model predictions.

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
        if self.fitted_models_ is None:
            raise RuntimeError("Fit the model before calling predict().")

        h = int(h)
        if h <= 0:
            return np.array([], dtype=float)

        # Collect predictions from all models
        all_preds = []
        for model in self.fitted_models_:
            try:
                pred = model.predict(h)
                all_preds.append(pred)
            except Exception:
                # If a model fails, use NaN (will be handled in averaging)
                all_preds.append(np.full(h, np.nan))

        all_preds = np.array(all_preds)  # Shape: (n_models, h)

        # Weighted average, handling potential NaNs
        preds = np.zeros(h)
        for i in range(h):
            valid_mask = ~np.isnan(all_preds[:, i])
            if np.any(valid_mask):
                valid_weights = self.weights_[valid_mask]
                valid_weights = valid_weights / np.sum(valid_weights)
                preds[i] = np.dot(valid_weights, all_preds[valid_mask, i])
            else:
                preds[i] = np.nan

        return preds


# ================= AutoEnsemble =================
class AutoEnsemble:
    """
    Automatic ensemble tuner that learns optimal model weights.

    Creates an ensemble from multiple base forecasters, using validation
    performance to determine optimal weights.

    Parameters
    ----------
    base_model_classes : list of tuples (class, kwargs), default=None
        List of (ModelClass, init_kwargs) pairs. If None, uses default set
        of models from randomstatsmodels.
    weighting_options : iterable of str, default=("uniform", "validation", "optimal")
        Candidate weighting methods:
        - "uniform": Equal weights for all models
        - "validation": Weights inversely proportional to validation error
        - "optimal": Solve for weights that minimize validation error
    metric : {"mae", "rmse"}, default="mae"
        Validation error metric.

    Attributes
    ----------
    model_ : EnsembleForecaster or None
        Best fitted ensemble.
    best_ : dict or None
        Dictionary with best configuration and validation score.
    base_scores_ : dict or None
        Validation scores for each base model.
    """

    def __init__(
        self,
        base_model_classes: Optional[List[tuple]] = None,
        weighting_options: Sequence[str] = ("uniform", "validation", "optimal"),
        metric: str = "mae",
    ) -> None:
        self.base_model_classes = base_model_classes
        self.weighting_options = list(weighting_options)
        self.metric = metric.lower()
        if self.metric not in ("mae", "rmse"):
            raise ValueError("metric must be 'mae' or 'rmse'")
        self.model_: Optional[EnsembleForecaster] = None
        self.best_: Optional[Dict] = None
        self.base_scores_: Optional[Dict] = None

    def _get_default_models(self):
        """Get default set of base models."""
        from .naive import NaiveForecaster
        from .holt_winters import HoltWintersForecaster
        from .local_linear import LocalLinearForecaster
        from .fourier import FourierForecaster

        return [
            (NaiveForecaster, {"method": "last"}),
            (NaiveForecaster, {"method": "drift"}),
            (HoltWintersForecaster, {"trend": "add", "seasonal": "none"}),
            (LocalLinearForecaster, {"decay": 0.95, "degree": 1}),
            (FourierForecaster, {"n_harmonics": 3}),
        ]

    def fit(self, y: ArrayLike, val_fraction: float = 0.25) -> "AutoEnsemble":
        """
        Fit ensemble with optimal weights determined by validation.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time series data.
        val_fraction : float, default=0.25
            Fraction of series reserved for validation.

        Returns
        -------
        self : AutoEnsemble
            Instance with best ensemble fitted on full dataset.

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
        if N - n_val < 3:
            raise ValueError("Not enough data for validation split.")
        split = N - n_val
        y_train, y_val = y[:split], y[split:]

        score_fn = mae if self.metric == "mae" else rmse

        # Get base model specs
        model_specs = (
            self.base_model_classes
            if self.base_model_classes is not None
            else self._get_default_models()
        )

        # Fit base models and collect validation predictions
        fitted_models = []
        val_predictions = []
        val_scores = {}

        for i, (ModelClass, kwargs) in enumerate(model_specs):
            model_name = f"{ModelClass.__name__}_{i}"
            try:
                # Fit on training data
                model = ModelClass(**kwargs).fit(y_train)

                # Rolling one-step forecast through validation
                preds = []
                current_data = y_train.copy()
                for t in range(split, N):
                    m = ModelClass(**kwargs).fit(current_data)
                    yhat = m.predict(1)[0]
                    preds.append(yhat)
                    current_data = np.append(current_data, y[t])

                preds = np.array(preds)
                score = score_fn(y_val, preds)

                fitted_models.append((ModelClass, kwargs))
                val_predictions.append(preds)
                val_scores[model_name] = score
            except Exception:
                continue

        if len(fitted_models) == 0:
            raise RuntimeError("No base models could be fitted successfully.")

        self.base_scores_ = val_scores
        val_predictions = np.array(val_predictions)  # Shape: (n_models, n_val)
        n_models = len(fitted_models)

        # Try different weighting strategies
        best_score = np.inf
        best_weights = None
        best_weighting = None

        for weighting in self.weighting_options:
            if weighting == "uniform":
                weights = np.ones(n_models) / n_models

            elif weighting == "validation":
                # Weights inversely proportional to validation error
                scores = np.array(list(val_scores.values()))
                # Avoid division by zero
                scores = np.maximum(scores, 1e-10)
                inv_scores = 1.0 / scores
                weights = inv_scores / np.sum(inv_scores)

            elif weighting == "optimal":
                # Solve for optimal weights using least squares
                # min_w ||Pw - y_val||^2 s.t. sum(w)=1, w>=0
                # Simplified: just use non-negative least squares or simple solve
                try:
                    # Stack predictions: P @ w = y_val
                    P = val_predictions.T  # Shape: (n_val, n_models)

                    # Simple least squares (may give negative weights)
                    w, *_ = np.linalg.lstsq(P, y_val, rcond=None)

                    # Project to simplex (non-negative, sum to 1)
                    w = np.maximum(w, 0)
                    if np.sum(w) > 0:
                        weights = w / np.sum(w)
                    else:
                        weights = np.ones(n_models) / n_models
                except Exception:
                    weights = np.ones(n_models) / n_models
            else:
                continue

            # Compute ensemble prediction with these weights
            ensemble_pred = np.dot(weights, val_predictions)
            score = score_fn(y_val, ensemble_pred)

            if score < best_score:
                best_score = score
                best_weights = weights
                best_weighting = weighting

        if best_weights is None:
            raise RuntimeError("AutoEnsemble failed to find valid weights.")

        # Create final ensemble fitted on full data
        final_models = []
        for ModelClass, kwargs in fitted_models:
            final_models.append(ModelClass(**kwargs))

        self.model_ = EnsembleForecaster(
            base_models=final_models,
            weights=best_weights,
            weighting="manual",
        ).fit(y)

        self.best_ = {
            "config": {
                "weighting": best_weighting,
                "n_models": n_models,
                "weights": best_weights.tolist(),
            },
            "val_score": best_score,
            "metric": self.metric,
        }
        return self

    def predict(self, h: int) -> np.ndarray:
        """
        Forecast using the fitted ensemble.

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
            raise RuntimeError("AutoEnsemble is not fitted yet.")
        return self.model_.predict(h)
