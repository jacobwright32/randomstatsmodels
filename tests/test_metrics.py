"""
Unit tests for randomstatsmodels metrics module.
"""

import numpy as np
import pytest

from randomstatsmodels import mae, mse, rmse, mape, smape


class TestMAE:
    def test_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert mae(y_true, y_pred) == 0.0

    def test_constant_error(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])  # error of 1 everywhere
        assert mae(y_true, y_pred) == 1.0

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(size=100)
        y_pred = rng.normal(size=100)
        assert mae(y_true, y_pred) >= 0


class TestMSE:
    def test_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert mse(y_true, y_pred) == 0.0

    def test_known_value(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])  # errors: 1, 1, 1
        # mse = mean(1^2, 1^2, 1^2) = 1.0
        assert mse(y_true, y_pred) == 1.0

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(size=100)
        y_pred = rng.normal(size=100)
        assert mse(y_true, y_pred) >= 0


class TestRMSE:
    def test_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert rmse(y_true, y_pred) == 0.0

    def test_sqrt_of_mse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        assert np.isclose(rmse(y_true, y_pred), np.sqrt(mse(y_true, y_pred)))

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(size=100)
        y_pred = rng.normal(size=100)
        assert rmse(y_true, y_pred) >= 0


class TestMAPE:
    def test_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert mape(y_true, y_pred) == 0.0

    def test_non_negative(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        assert mape(y_true, y_pred) >= 0


class TestSMAPE:
    def test_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert smape(y_true, y_pred) == 0.0

    def test_symmetric(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        # SMAPE should be the same if we swap true and pred
        assert np.isclose(smape(y_true, y_pred), smape(y_pred, y_true))

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = np.abs(rng.normal(size=100)) + 1  # positive values
        y_pred = np.abs(rng.normal(size=100)) + 1
        assert smape(y_true, y_pred) >= 0

    def test_bounded(self):
        # SMAPE should be bounded (typically 0-200 or 0-100 depending on formula)
        y_true = np.array([1, 2, 3])
        y_pred = np.array([100, 100, 100])
        result = smape(y_true, y_pred)
        assert 0 <= result <= 200
