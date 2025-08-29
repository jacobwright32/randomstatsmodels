import numpy as np
from typing import Union, Sequence

ArrayLike = Union[Sequence[float], np.ndarray]


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    epsilon : float, default=1e-8
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Mean absolute percentage error (in percent).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    epsilon : float, default=1e-8
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Symmetric mean absolute percentage error (in percent).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)
