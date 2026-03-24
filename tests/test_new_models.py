"""
Unit tests for the 5 new calculus-based forecasting models.

Tests verify:
- Model instantiation
- fit() returns self
- predict() returns correct shape (ndarray of length h)
- Predictions are finite numbers
- Error handling (predict before fit raises RuntimeError)
- Auto* tuners populate best_ with "config" key
"""

import numpy as np
import pytest

from randomstatsmodels import (
    FracDiffForecaster,
    AutoFracDiff,
    SpectralGradientForecaster,
    AutoSpectralGradient,
    GreensKernelForecaster,
    AutoGreensKernel,
    PDEFieldForecaster,
    AutoPDEField,
    VariationalPathForecaster,
    AutoVariationalPath,
    KoopmanForecaster,
    AutoKoopman,
)


# --- Fixtures ---

@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    t = np.arange(200)
    y = 10 + 0.05 * t + np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=t.size)
    return y


@pytest.fixture
def short_data():
    rng = np.random.default_rng(123)
    return rng.random(50) * 100


@pytest.fixture
def forecast_horizon():
    return 12


# ===================================================================
# FracDiffForecaster
# ===================================================================

class TestFracDiffForecaster:
    def test_instantiation(self):
        m = FracDiffForecaster()
        assert m.d == 0.4
        assert m.ar_order == 5

    def test_fit_returns_self(self, synthetic_data):
        m = FracDiffForecaster()
        result = m.fit(synthetic_data)
        assert result is m

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = FracDiffForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_predictions_finite(self, synthetic_data, forecast_horizon):
        m = FracDiffForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert np.all(np.isfinite(preds))

    def test_predict_before_fit_raises(self):
        m = FracDiffForecaster()
        with pytest.raises(RuntimeError):
            m.predict(5)


class TestAutoFracDiff:
    def test_instantiation(self):
        m = AutoFracDiff()
        assert m.model_ is None

    def test_fit_returns_self(self, synthetic_data):
        m = AutoFracDiff(d_grid=(0.3, 0.5), ar_grid=(3, 5))
        result = m.fit(synthetic_data)
        assert result is m

    def test_best_populated(self, synthetic_data):
        m = AutoFracDiff(d_grid=(0.3, 0.5), ar_grid=(3, 5))
        m.fit(synthetic_data)
        assert m.best_ is not None
        assert "config" in m.best_

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = AutoFracDiff(d_grid=(0.3, 0.5), ar_grid=(3, 5))
        m.fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert preds.shape == (forecast_horizon,)

    def test_predict_before_fit_raises(self):
        m = AutoFracDiff()
        with pytest.raises(RuntimeError):
            m.predict(5)


# ===================================================================
# SpectralGradientForecaster
# ===================================================================

class TestSpectralGradientForecaster:
    def test_instantiation(self):
        m = SpectralGradientForecaster()
        assert m.window_size == 32

    def test_fit_returns_self(self, synthetic_data):
        m = SpectralGradientForecaster()
        result = m.fit(synthetic_data)
        assert result is m

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = SpectralGradientForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_predictions_finite(self, synthetic_data, forecast_horizon):
        m = SpectralGradientForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert np.all(np.isfinite(preds))

    def test_predict_before_fit_raises(self):
        m = SpectralGradientForecaster()
        with pytest.raises(RuntimeError):
            m.predict(5)


class TestAutoSpectralGradient:
    def test_instantiation(self):
        m = AutoSpectralGradient()
        assert m.model_ is None

    def test_fit_returns_self(self, synthetic_data):
        m = AutoSpectralGradient(window_grid=(16, 32), hop_grid=(8,),
                                  damping_grid=(0.9,), trend_weight_grid=(0.5,))
        result = m.fit(synthetic_data)
        assert result is m

    def test_best_populated(self, synthetic_data):
        m = AutoSpectralGradient(window_grid=(16, 32), hop_grid=(8,),
                                  damping_grid=(0.9,), trend_weight_grid=(0.5,))
        m.fit(synthetic_data)
        assert m.best_ is not None
        assert "config" in m.best_

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = AutoSpectralGradient(window_grid=(16,), hop_grid=(8,),
                                  damping_grid=(0.9,), trend_weight_grid=(0.5,))
        m.fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert preds.shape == (forecast_horizon,)

    def test_predict_before_fit_raises(self):
        m = AutoSpectralGradient()
        with pytest.raises(RuntimeError):
            m.predict(5)


# ===================================================================
# GreensKernelForecaster
# ===================================================================

class TestGreensKernelForecaster:
    def test_instantiation(self):
        m = GreensKernelForecaster()
        assert m.n_components == 3

    def test_fit_returns_self(self, synthetic_data):
        m = GreensKernelForecaster()
        result = m.fit(synthetic_data)
        assert result is m

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = GreensKernelForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_predictions_finite(self, synthetic_data, forecast_horizon):
        m = GreensKernelForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert np.all(np.isfinite(preds))

    def test_predict_before_fit_raises(self):
        m = GreensKernelForecaster()
        with pytest.raises(RuntimeError):
            m.predict(5)


class TestAutoGreensKernel:
    def test_instantiation(self):
        m = AutoGreensKernel()
        assert m.model_ is None

    def test_fit_returns_self(self, synthetic_data):
        m = AutoGreensKernel(n_components_grid=(2, 3), max_lag_grid=(20,),
                             kernel_window_grid=(0,))
        result = m.fit(synthetic_data)
        assert result is m

    def test_best_populated(self, synthetic_data):
        m = AutoGreensKernel(n_components_grid=(2, 3), max_lag_grid=(20,),
                             kernel_window_grid=(0,))
        m.fit(synthetic_data)
        assert m.best_ is not None
        assert "config" in m.best_

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = AutoGreensKernel(n_components_grid=(2,), max_lag_grid=(20,),
                             kernel_window_grid=(0,))
        m.fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert preds.shape == (forecast_horizon,)

    def test_predict_before_fit_raises(self):
        m = AutoGreensKernel()
        with pytest.raises(RuntimeError):
            m.predict(5)


# ===================================================================
# PDEFieldForecaster
# ===================================================================

class TestPDEFieldForecaster:
    def test_instantiation(self):
        m = PDEFieldForecaster()
        assert m.n_scales == 8

    def test_fit_returns_self(self, synthetic_data):
        m = PDEFieldForecaster()
        result = m.fit(synthetic_data)
        assert result is m

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = PDEFieldForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_predictions_finite(self, synthetic_data, forecast_horizon):
        m = PDEFieldForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert np.all(np.isfinite(preds))

    def test_predict_before_fit_raises(self):
        m = PDEFieldForecaster()
        with pytest.raises(RuntimeError):
            m.predict(5)


class TestAutoPDEField:
    def test_instantiation(self):
        m = AutoPDEField()
        assert m.model_ is None

    def test_fit_returns_self(self, synthetic_data):
        m = AutoPDEField(n_scales_grid=(5,), max_sigma_grid=(10.0,),
                         dt_steps_grid=(2,))
        result = m.fit(synthetic_data)
        assert result is m

    def test_best_populated(self, synthetic_data):
        m = AutoPDEField(n_scales_grid=(5,), max_sigma_grid=(10.0,),
                         dt_steps_grid=(2,))
        m.fit(synthetic_data)
        assert m.best_ is not None
        assert "config" in m.best_

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = AutoPDEField(n_scales_grid=(5,), max_sigma_grid=(10.0,),
                         dt_steps_grid=(2,))
        m.fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert preds.shape == (forecast_horizon,)

    def test_predict_before_fit_raises(self):
        m = AutoPDEField()
        with pytest.raises(RuntimeError):
            m.predict(5)


# ===================================================================
# VariationalPathForecaster
# ===================================================================

class TestVariationalPathForecaster:
    def test_instantiation(self):
        m = VariationalPathForecaster()
        assert m.smoothness == 1.0

    def test_fit_returns_self(self, synthetic_data):
        m = VariationalPathForecaster()
        result = m.fit(synthetic_data)
        assert result is m

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = VariationalPathForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_predictions_finite(self, synthetic_data, forecast_horizon):
        m = VariationalPathForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert np.all(np.isfinite(preds))

    def test_predict_before_fit_raises(self):
        m = VariationalPathForecaster()
        with pytest.raises(RuntimeError):
            m.predict(5)


class TestAutoVariationalPath:
    def test_instantiation(self):
        m = AutoVariationalPath()
        assert m.model_ is None

    def test_fit_returns_self(self, synthetic_data):
        m = AutoVariationalPath(smoothness_grid=(1.0,), jerk_grid=(0.1,),
                                potential_grid=(0.5,), seasonal_weight_grid=(1.0,),
                                attractor_grid=("mean",))
        result = m.fit(synthetic_data)
        assert result is m

    def test_best_populated(self, synthetic_data):
        m = AutoVariationalPath(smoothness_grid=(1.0,), jerk_grid=(0.1,),
                                potential_grid=(0.5,), seasonal_weight_grid=(1.0,),
                                attractor_grid=("mean",))
        m.fit(synthetic_data)
        assert m.best_ is not None
        assert "config" in m.best_

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = AutoVariationalPath(smoothness_grid=(1.0,), jerk_grid=(0.1,),
                                potential_grid=(0.5,), seasonal_weight_grid=(1.0,),
                                attractor_grid=("mean",))
        m.fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert preds.shape == (forecast_horizon,)

    def test_predict_before_fit_raises(self):
        m = AutoVariationalPath()
        with pytest.raises(RuntimeError):
            m.predict(5)


# ===================================================================
# KoopmanForecaster
# ===================================================================

class TestKoopmanForecaster:
    def test_instantiation(self):
        m = KoopmanForecaster()
        assert m.embed_dim == 12

    def test_fit_returns_self(self, synthetic_data):
        m = KoopmanForecaster()
        result = m.fit(synthetic_data)
        assert result is m

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = KoopmanForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (forecast_horizon,)

    def test_predictions_finite(self, synthetic_data, forecast_horizon):
        m = KoopmanForecaster().fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert np.all(np.isfinite(preds))

    def test_predict_before_fit_raises(self):
        m = KoopmanForecaster()
        with pytest.raises(RuntimeError):
            m.predict(5)


class TestAutoKoopman:
    def test_instantiation(self):
        m = AutoKoopman()
        assert m.model_ is None

    def test_fit_returns_self(self, synthetic_data):
        m = AutoKoopman(embed_grid=(4, 8), rank_grid=(0,),
                        damping_grid=(0.98,), deseason_grid=(True,),
                        detrend_grid=(True,))
        result = m.fit(synthetic_data)
        assert result is m

    def test_best_populated(self, synthetic_data):
        m = AutoKoopman(embed_grid=(4, 8), rank_grid=(0,),
                        damping_grid=(0.98,), deseason_grid=(True,),
                        detrend_grid=(True,))
        m.fit(synthetic_data)
        assert m.best_ is not None
        assert "config" in m.best_

    def test_predict_shape(self, synthetic_data, forecast_horizon):
        m = AutoKoopman(embed_grid=(8,), rank_grid=(0,),
                        damping_grid=(0.98,), deseason_grid=(True,),
                        detrend_grid=(True,))
        m.fit(synthetic_data)
        preds = m.predict(forecast_horizon)
        assert preds.shape == (forecast_horizon,)

    def test_predict_before_fit_raises(self):
        m = AutoKoopman()
        with pytest.raises(RuntimeError):
            m.predict(5)


# ===================================================================
# Dataset & evaluation module smoke tests
# ===================================================================

class TestDatasets:
    def test_load_datasets(self):
        from randomstatsmodels.benchmarking.datasets import load_datasets
        ds = load_datasets()
        assert len(ds) >= 30

    def test_get_dataset(self):
        from randomstatsmodels.benchmarking.datasets import get_dataset
        d = get_dataset("AirPassengers")
        assert d["name"] == "AirPassengers"
        assert len(d["values"]) == 144

    def test_train_test_split(self):
        from randomstatsmodels.benchmarking.datasets import train_test_split
        y = np.arange(100, dtype=float)
        y_train, y_test = train_test_split(y, 0.2)
        assert len(y_train) + len(y_test) == 100
        assert len(y_test) >= 6


class TestEvaluation:
    def test_evaluate_model(self, synthetic_data):
        from randomstatsmodels.benchmarking.evaluation import evaluate_model
        result = evaluate_model(FracDiffForecaster, synthetic_data[:180],
                                synthetic_data[180:])
        assert result["error"] is None
        assert "mae" in result
        assert result["mae"] >= 0
