"""
Wrappers for statsforecast models to match the randomstatsmodels API.

Each wrapper adapts a statsforecast model to the .fit(y) / .predict(h) interface
so it can be used with the evaluation framework.
"""

import numpy as np


def _detect_season(y, max_period=60):
    """Simple ACF-based period detection."""
    n = len(y)
    if n < 10:
        return 1
    yc = y - np.mean(y)
    var = np.dot(yc, yc)
    if var < 1e-12:
        return 1
    mp = min(max_period, n // 3)
    for k in range(2, mp):
        acf_k = np.dot(yc[:n - k], yc[k:]) / var
        acf_prev = np.dot(yc[:n - k + 1], yc[k - 1:]) / var
        acf_next = np.dot(yc[:n - k - 1], yc[k + 1:]) / var if k + 1 < n else 0
        if acf_k > acf_prev and acf_k > acf_next and acf_k > 0.15:
            return k
    return 1


class SFAutoARIMA:
    """statsforecast AutoARIMA wrapper."""
    __name__ = "SF_AutoARIMA"

    def __init__(self):
        self._model = None
        self._fitted = False

    def fit(self, y):
        from statsforecast.models import AutoARIMA
        y = np.asarray(y, float)
        sl = _detect_season(y)
        self._model = AutoARIMA(season_length=max(sl, 1))
        self._model.fit(y)
        self._fitted = True
        return self

    def predict(self, h):
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")
        result = self._model.predict(h=int(h))
        return np.asarray(result["mean"], float)


class SFAutoETS:
    """statsforecast AutoETS wrapper."""
    __name__ = "SF_AutoETS"

    def __init__(self):
        self._model = None
        self._fitted = False

    def fit(self, y):
        from statsforecast.models import AutoETS
        y = np.asarray(y, float)
        sl = _detect_season(y)
        self._model = AutoETS(season_length=max(sl, 1))
        self._model.fit(y)
        self._fitted = True
        return self

    def predict(self, h):
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")
        result = self._model.predict(h=int(h))
        return np.asarray(result["mean"], float)


class SFAutoCES:
    """statsforecast AutoCES wrapper."""
    __name__ = "SF_AutoCES"

    def __init__(self):
        self._model = None
        self._fitted = False

    def fit(self, y):
        from statsforecast.models import AutoCES
        y = np.asarray(y, float)
        sl = _detect_season(y)
        self._model = AutoCES(season_length=max(sl, 1))
        self._model.fit(y)
        self._fitted = True
        return self

    def predict(self, h):
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")
        result = self._model.predict(h=int(h))
        return np.asarray(result["mean"], float)


class SFAutoTheta:
    """statsforecast AutoTheta wrapper."""
    __name__ = "SF_AutoTheta"

    def __init__(self):
        self._model = None
        self._fitted = False

    def fit(self, y):
        from statsforecast.models import AutoTheta
        y = np.asarray(y, float)
        sl = _detect_season(y)
        self._model = AutoTheta(season_length=max(sl, 1))
        self._model.fit(y)
        self._fitted = True
        return self

    def predict(self, h):
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")
        result = self._model.predict(h=int(h))
        return np.asarray(result["mean"], float)


class SFAutoTBATS:
    """statsforecast AutoTBATS wrapper."""
    __name__ = "SF_AutoTBATS"

    def __init__(self):
        self._model = None
        self._fitted = False

    def fit(self, y):
        from statsforecast.models import AutoTBATS
        y = np.asarray(y, float)
        sl = _detect_season(y)
        self._model = AutoTBATS(season_length=max(sl, 1))
        self._model.fit(y)
        self._fitted = True
        return self

    def predict(self, h):
        if not self._fitted:
            raise RuntimeError("Fit the model before predicting.")
        result = self._model.predict(h=int(h))
        return np.asarray(result["mean"], float)
