from typing import Optional, Dict, Iterable, Callable, Union
import numpy as np
from ..metrics import mae, rmse
from .model_utils import _penalty_value, _weighted_quantile, _golden_section_minimize

ArrayLike = Union[np.ndarray, Iterable[float]]


# ================= PALF (Proximal Aggregation Lag Forecaster) =================
class PALF:
    """
    Proximal Aggregation Lag Forecaster (configured model).

    Parameters
    ----------
    p : int, default=8
        Number of autoregressive lags.
    penalty : {"huber", "l1", "l2", "pinball"}, default="huber"
        Type of penalty function applied to residuals.
    decay_param : float, default=5.0
        Decay parameter for exponential lag weights.
    huber_delta : float, default=1.0
        Threshold parameter for the Huber loss.
    pinball_tau : float, default=0.5
        Quantile parameter for the pinball (quantile) loss.
    level_penalty : {"l1", "l2", "pinball"}, default="l2"
        Penalty applied to deviation from level anchor.
    level_weight : float, default=1.0
        Weight for the level penalty term.
    irregular_timestamps : array-like of shape (n_samples,) or None, default=None
        Optional timestamps to handle irregular spacing.
        If provided, exponential decay is based on time gaps.

    Attributes
    ----------
    mu_ : float or None
        Internal level anchor, updated during forecasting.
    y_ : ndarray of shape (n_samples,) or None
        Training time series data.
    """

    def __init__(
        self,
        p: int = 8,
        penalty: str = "huber",
        decay_param: float = 5.0,
        huber_delta: float = 1.0,
        pinball_tau: float = 0.5,
        level_penalty: str = "l2",
        level_weight: float = 1.0,
        irregular_timestamps: Optional[np.ndarray] = None,
    ) -> None:
        self.p = int(p)
        self.penalty = penalty
        self.decay_param = float(decay_param)
        self.huber_delta = float(huber_delta)
        self.pinball_tau = float(pinball_tau)
        self.level_penalty = level_penalty
        self.level_weight = float(level_weight)
        self.irregular_timestamps = irregular_timestamps

        self.mu_: Optional[float] = None
        self.y_: Optional[np.ndarray] = None

    def _lag_weights(self, t_index: int) -> np.ndarray:
        """
        Compute exponential weights for lagged observations.

        Parameters
        ----------
        t_index : int
            Index in the time series for which lag weights are computed.

        Returns
        -------
        weights : ndarray of shape (p,)
            Exponential weights for each lag.
        """
        if self.irregular_timestamps is None:
            return np.array(
                [np.exp(-(i - 1) / self.decay_param) for i in range(1, self.p + 1)],
                dtype=float,
            )
        ts = self.irregular_timestamps
        t0 = ts[t_index]
        gaps = np.array([t0 - ts[t_index - i] for i in range(1, self.p + 1)], dtype=float)
        return np.exp(-gaps / max(self.decay_param, 1e-9))

    def _objective_factory(
        self,
        anchors: np.ndarray,
        weights: np.ndarray,
        level_anchor: Optional[float]
    ) -> Callable[[float], float]:
        """
        Create an objective function for penalized aggregation.

        Parameters
        ----------
        anchors : array_like of shape (p,)
            Lagged values serving as anchors.
        weights : array_like of shape (p,)
            Corresponding lag weights.
        level_anchor : float or None
            Reference anchor for level penalty.

        Returns
        -------
        J : callable
            Objective function mapping a candidate value to a penalty score.
        """
        kind = self.penalty
        delta = self.huber_delta
        tau = self.pinball_tau
        a_vals = np.asarray(anchors, float)
        w = np.asarray(weights, float)

        def J(z: float) -> float:
            r = z - a_vals
            val = np.sum(w * _penalty_value(r, kind, delta, tau))
            if level_anchor is not None and self.level_weight > 0.0:
                val += self.level_weight * _penalty_value(z - level_anchor, self.level_penalty, delta, 0.5)
            return float(val)

        return J

    def _solve_argmin(self, anchors: np.ndarray, weights: np.ndarray, level_anchor: Optional[float]) -> float:
        """
        Solve the penalized aggregation problem for a single forecast.

        Parameters
        ----------
        anchors : ndarray of shape (p,)
            Lagged values.
        weights : ndarray of shape (p,)
            Lag weights.
        level_anchor : float or None
            Reference anchor for level penalty.

        Returns
        -------
        yhat : float
            One-step forecast value.
        """
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

    def fit(self, y: np.ndarray) -> "PALF":
        """
        Fit the PALF model to a univariate time series.

        Parameters
        ----------
        y : array_like of shape (n_samples,)
            Training time series.

        Returns
        -------
        self : PALF
            Fitted model instance.
        """
        y = np.asarray(y, float)
        self.y_ = y.copy()
        self.mu_ = float(np.median(y[: max(self.p, 5)]))
        return self

    def _one_step(self, t: int) -> float:
        """
        Perform one-step forecasting at index `t`.

        Parameters
        ----------
        t : int
            Current time index.

        Returns
        -------
        yhat : float
            One-step forecast.
        """
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

    def predict(self, h: int) -> np.ndarray:
        """
        Generate iterative multi-step forecasts.

        Parameters
        ----------
        h : int
            Forecast horizon.

        Returns
        -------
        preds : ndarray of shape (h,)
            Forecasted values.
        """
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
    """
    Automatic hyperparameter tuner for :class:`PALF`.

    Parameters
    ----------
    p_candidates : iterable of int, default=(4, 8, 12)
        Candidate lag lengths.
    penalties : iterable of str, default=("huber", "l2", "l1", "pinball")
        Candidate penalty functions.
    decay_params : iterable of float, default=(3.0, 5.0, 8.0)
        Candidate decay parameters for exponential weights.
    huber_deltas : iterable of float, default=(0.5, 1.0, 2.0)
        Candidate Huber delta values.
    pinball_taus : iterable of float, default=(0.5,)
        Candidate pinball (quantile) loss tau values.
    level_penalty : {"l1", "l2", "pinball"}, default="l2"
        Penalty for deviation from level anchor.
    level_weight : float, default=1.0
        Weight of the level penalty term.
    irregular_timestamps : array_like or None, default=None
        Optional timestamps for irregular sampling.

    Attributes
    ----------
    model_ : PALF or None
        The best-fit PALF model after tuning.
    best_ : dict or None
        Dictionary with validation score.
    """

    def __init__(
        self,
        p_candidates: Iterable[int] = (4, 8, 12),
        penalties: Iterable[str] = ("huber", "l2", "l1", "pinball"),
        decay_params: Iterable[float] = (3.0, 5.0, 8.0),
        huber_deltas: Iterable[float] = (0.5, 1.0, 2.0),
        pinball_taus: Iterable[float] = (0.5,),
        level_penalty: str = "l2",
        level_weight: float = 1.0,
        irregular_timestamps: Optional[np.ndarray] = None,
    ) -> None:
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
        self.model_: Optional[PALF] = None
        self.best_: Optional[Dict] = None

    def fit(self, y: np.ndarray, val_fraction: float = 0.25, metric: str = "mae") -> "AutoPALF":
        """
        Fit AutoPALF on a univariate series by grid search.

        Parameters
        ----------
        y : array_like of shape (n_samples,)
            Input time series.
        val_fraction : float, default=0.25
            Fraction of data used for validation (minimum 16 samples).
        metric : {"mae", "rmse"}, default="mae"
            Validation metric.

        Returns
        -------
        self : AutoPALF
            Instance with best model stored in :attr:`model_`.
        """
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

    def predict(self, h: int) -> np.ndarray:
        """
        Generate forecasts using the best-fit model.

        Parameters
        ----------
        h : int
            Forecast horizon.

        Returns
        -------
        preds : ndarray of shape (h,)
            Forecasted values.
        """
        return self.model_.predict(h)
