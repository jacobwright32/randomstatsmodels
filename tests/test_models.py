"""
Unit tests for randomstatsmodels forecasting models.

Tests verify:
- Model instantiation
- fit() returns self
- predict() returns correct shape (ndarray of length h)
- Predictions are finite numbers
- Error handling (predict before fit raises RuntimeError)
"""

import numpy as np
import pytest

from randomstatsmodels import (
    AutoFourier,
    AutoHybridForecaster,
    AutoMELD,
    AutoKNN,
    AutoPALF,
    AutoNEO,
    AutoThetaAR,
    AutoPolymath,
    AutoNaive,
    AutoHoltWinters,
    AutoSSA,
    AutoLocalLinear,
    AutoEnsemble,
    AutoRIFT,
    FourierForecaster,
    NaiveForecaster,
    HoltWintersForecaster,
    SSAForecaster,
    LocalLinearForecaster,
    EnsembleForecaster,
    RIFTForecaster,
)


# --- Fixtures ---

@pytest.fixture
def synthetic_data():
    """Generate synthetic time series with trend and seasonality."""
    rng = np.random.default_rng(42)
    t = np.arange(200)
    y = 10 + 0.05 * t + np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=t.size)
    return y


@pytest.fixture
def short_data():
    """Short time series for edge case testing."""
    rng = np.random.default_rng(42)
    return rng.normal(size=50)


@pytest.fixture
def forecast_horizon():
    """Standard forecast horizon for tests."""
    return 12


# --- Base Forecaster Tests ---

class TestFourierForecaster:
    def test_instantiation(self):
        model = FourierForecaster()
        assert model.n_harmonics == 3
        assert model.trend == "linear"

    def test_fit_returns_self(self, synthetic_data):
        model = FourierForecaster()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = FourierForecaster().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_predictions_finite(self, synthetic_data, forecast_horizon):
        model = FourierForecaster().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert np.all(np.isfinite(preds))

    def test_predict_before_fit_raises(self, forecast_horizon):
        model = FourierForecaster()
        with pytest.raises(RuntimeError):
            model.predict(forecast_horizon)


class TestNaiveForecaster:
    def test_instantiation(self):
        model = NaiveForecaster()
        assert model.method == "last"

    def test_fit_returns_self(self, synthetic_data):
        model = NaiveForecaster()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = NaiveForecaster().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_seasonal_method(self, synthetic_data, forecast_horizon):
        model = NaiveForecaster(method="seasonal", seasonal_period=12)
        model.fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert preds.shape == (forecast_horizon,)


class TestHoltWintersForecaster:
    def test_instantiation(self):
        model = HoltWintersForecaster()
        assert model.seasonal_period == 12

    def test_fit_returns_self(self, synthetic_data):
        model = HoltWintersForecaster()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = HoltWintersForecaster().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestSSAForecaster:
    def test_instantiation(self):
        model = SSAForecaster()
        assert model.window_length is None  # uses N//3 by default

    def test_fit_returns_self(self, synthetic_data):
        model = SSAForecaster()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = SSAForecaster().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestLocalLinearForecaster:
    def test_instantiation(self):
        model = LocalLinearForecaster()
        assert model.decay == 0.95

    def test_fit_returns_self(self, synthetic_data):
        model = LocalLinearForecaster()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = LocalLinearForecaster().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestRIFTForecaster:
    def test_instantiation(self):
        model = RIFTForecaster()
        assert model.n_frequencies == 4

    def test_fit_returns_self(self, synthetic_data):
        model = RIFTForecaster()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = RIFTForecaster().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_predict_before_fit_raises(self, forecast_horizon):
        model = RIFTForecaster()
        with pytest.raises(RuntimeError):
            model.predict(forecast_horizon)


# --- Auto Tuner Tests ---

class TestAutoFourier:
    def test_instantiation(self):
        model = AutoFourier()
        assert model.metric == "mae"

    def test_fit_returns_self(self, synthetic_data):
        model = AutoFourier()
        result = model.fit(synthetic_data)
        assert result is model

    def test_best_config_set(self, synthetic_data):
        model = AutoFourier().fit(synthetic_data)
        assert model.best_ is not None
        assert "config" in model.best_

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoFourier().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_predict_before_fit_raises(self, forecast_horizon):
        model = AutoFourier()
        with pytest.raises(RuntimeError):
            model.predict(forecast_horizon)


class TestAutoNaive:
    def test_instantiation(self):
        model = AutoNaive()
        assert model.metric == "mae"

    def test_fit_returns_self(self, synthetic_data):
        model = AutoNaive()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoNaive().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoHoltWinters:
    def test_instantiation(self):
        model = AutoHoltWinters()
        assert model.metric == "mae"

    def test_fit_returns_self(self, synthetic_data):
        model = AutoHoltWinters()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoHoltWinters().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoSSA:
    def test_instantiation(self):
        model = AutoSSA()
        assert model.metric == "mae"

    def test_fit_returns_self(self, synthetic_data):
        model = AutoSSA()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoSSA().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoLocalLinear:
    def test_instantiation(self):
        model = AutoLocalLinear()
        assert model.metric == "mae"

    def test_fit_returns_self(self, synthetic_data):
        model = AutoLocalLinear()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoLocalLinear().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoEnsemble:
    def test_instantiation(self):
        model = AutoEnsemble()
        assert model.metric == "mae"

    def test_fit_returns_self(self, synthetic_data):
        model = AutoEnsemble()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoEnsemble().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoRIFT:
    def test_instantiation(self):
        model = AutoRIFT()
        assert model.metric == "mae"

    def test_fit_returns_self(self, synthetic_data):
        model = AutoRIFT()
        result = model.fit(synthetic_data)
        assert result is model

    def test_best_config_set(self, synthetic_data):
        model = AutoRIFT().fit(synthetic_data)
        assert model.best_ is not None
        assert "config" in model.best_

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoRIFT().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_predict_before_fit_raises(self, forecast_horizon):
        model = AutoRIFT()
        with pytest.raises(RuntimeError):
            model.predict(forecast_horizon)


class TestAutoNEO:
    def test_instantiation(self):
        model = AutoNEO()
        assert model is not None

    def test_fit_returns_self(self, synthetic_data):
        model = AutoNEO()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoNEO().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoPolymath:
    def test_instantiation(self):
        model = AutoPolymath()
        assert model is not None

    def test_fit_returns_self(self, synthetic_data):
        model = AutoPolymath()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoPolymath().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoKNN:
    def test_instantiation(self):
        model = AutoKNN()
        assert hasattr(model, "metric")

    def test_fit_returns_self(self, synthetic_data):
        model = AutoKNN()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoKNN().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoMELD:
    def test_instantiation(self):
        model = AutoMELD()
        assert hasattr(model, "metric")

    def test_fit_returns_self(self, synthetic_data):
        model = AutoMELD()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoMELD().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoPALF:
    def test_instantiation(self):
        model = AutoPALF()
        assert model is not None

    def test_fit_returns_self(self, synthetic_data):
        model = AutoPALF()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoPALF().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoThetaAR:
    def test_instantiation(self):
        model = AutoThetaAR()
        assert model is not None

    def test_fit_returns_self(self, synthetic_data):
        model = AutoThetaAR()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoThetaAR().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)


class TestAutoHybridForecaster:
    def test_instantiation(self):
        model = AutoHybridForecaster()
        assert hasattr(model, "candidate_fourier")

    def test_fit_returns_self(self, synthetic_data):
        model = AutoHybridForecaster()
        result = model.fit(synthetic_data)
        assert result is model

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        model = AutoHybridForecaster().fit(synthetic_data)
        preds = model.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)
