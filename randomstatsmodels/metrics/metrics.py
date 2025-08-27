import numpy as np


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred, epsilon=1e-8):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), epsilon)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def smape(y_true, y_pred, epsilon=1e-8):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, epsilon)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0
