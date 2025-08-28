import numpy as np


# ========== AutoThetaAR ==========
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
