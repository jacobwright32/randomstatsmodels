import time
import math
import numpy as np
from ..metrics import mae, rmse, mape, smape

import time
import numpy as np
from ..metrics import mae, rmse, mape, smape


def benchmark_model(model_class, data, iterations=1, h=7):
    """
    Benchmark the training + prediction speed of a time series model,
    and compute MAE, RMSE, MAPE, sMAPE on the last h points.

    Parameters
    ----------
    model_class : class
        The model class to initialize (e.g., AutoNEO).
    data : array-like
        The time series data.
    iterations : int, default=5
        Number of times to run the benchmark.
    h : int, default=20
        Forecast horizon.
    **fit_kwargs : dict
        Additional arguments passed to model.fit().

    Returns
    -------
    results : dict
        {
          "avg_total_time_s": float,
          "avg_fit_time_s": float,
          "avg_predict_time_s": float,
          "avg_mae": float,
          "avg_rmse": float,
          "avg_mape": float,
          "avg_smape": float,
          "per_iteration": [
              {
                "fit_time_s": float,
                "predict_time_s": float,
                "total_time_s": float,
                "mae": float,
                "rmse": float,
                "mape": float,
                "smape": float
              },
              ...
          ]
        }
    """
    data = np.asarray(data)
    assert len(data) > h, "Data length must be greater than forecast horizon h."

    per_iter = []
    for i in range(iterations):

        model = model_class()  # fresh model each run

        model_name = model_class.__name__

        t0 = time.time()
        model.fit(data[:-h])
        fit_time = time.time() - t0

        t1 = time.time()
        y_pred = model.predict(h)

        predict_time = time.time() - t1

        total_time = fit_time + predict_time

        y_true = data[-h:]
        # Ensure shapes compatible
        y_pred = np.asarray(y_pred).reshape(-1)[:h]

        if model_name == "AutoETS":
            y_pred = y_pred[0]["mean"]

        iter_metrics = {
            "fit_time_s": fit_time,
            "predict_time_s": predict_time,
            "total_time_s": total_time,
            "mae": float(mae(y_true, y_pred)),
            "rmse": float(rmse(y_true, y_pred)),
            "mape": float(mape(y_true, y_pred)),
            "smape": float(smape(y_true, y_pred)),
        }
        per_iter.append(iter_metrics)

    # Averages
    avg_total = float(np.mean([x["total_time_s"] for x in per_iter]))
    avg_fit = float(np.mean([x["fit_time_s"] for x in per_iter]))
    avg_predict = float(np.mean([x["predict_time_s"] for x in per_iter]))
    avg_mae_ = float(np.mean([x["mae"] for x in per_iter]))
    avg_rmse_ = float(np.mean([x["rmse"] for x in per_iter]))
    avg_mape_ = float(np.mean([x["mape"] for x in per_iter]))
    avg_smape_ = float(np.mean([x["smape"] for x in per_iter]))

    print(
        f"\nAverages over {iterations} runs --> "
        f"fit: {avg_fit:.4f}s | predict: {avg_predict:.4f}s | total: {avg_total:.4f}s | "
        f"MAE: {avg_mae_:.4f} | RMSE: {avg_rmse_:.4f} | MAPE: {avg_mape_:.4f} | sMAPE: {avg_smape_:.4f}"
    )

    return {
        "avg_total_time_s": round(avg_total, 2),
        "avg_fit_time_s": round(avg_fit, 2),
        "avg_predict_time_s": round(avg_predict, 2),
        "avg_mae": round(avg_mae_, 3),
        "avg_rmse": round(avg_rmse_, 3),
        "avg_mape": round(avg_mape_, 3),
        "avg_smape": round(avg_smape_, 3),
        "per_iteration": per_iter,
    }


import time
import numpy as np
from ..metrics import mae, rmse, mape, smape


def _coerce_forecast(yp, h, model_name):
    """
    Coerce various model outputs to a 1D np.ndarray of length h.
    Handles special cases like AutoETS structure.
    """
    # Special-case: your AutoETS wrapper shape
    if model_name == "AutoETS":
        # expected like [{'mean': np.array([...])}, ...] or similar
        try:
            yp = yp["mean"]

        except Exception:
            pass

    if yp.shape[0] != h:
        raise ValueError(f"{model_name}.predict({h}) returned length {yp.shape[0]} (expected {h}).")
    return yp


def _metrics_dict(y_true, y_pred):
    return {
        "mae": float(mae(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
    }


def _avg_block(per_iter):
    return {
        "avg_total_time_s": (
            round(float(np.mean([x["total_time_s"] for x in per_iter])), 2)
            if per_iter and "total_time_s" in per_iter[0]
            else None
        ),
        "avg_fit_time_s": (
            round(float(np.mean([x["fit_time_s"] for x in per_iter])), 2)
            if per_iter and "fit_time_s" in per_iter[0]
            else None
        ),
        "avg_predict_time_s": (
            round(float(np.mean([x["predict_time_s"] for x in per_iter])), 2)
            if per_iter and "predict_time_s" in per_iter[0]
            else None
        ),
        "avg_mae": (round(float(np.mean([x["mae"] for x in per_iter])), 3) if per_iter else None),
        "avg_rmse": (round(float(np.mean([x["rmse"] for x in per_iter])), 3) if per_iter else None),
        "avg_mape": (round(float(np.mean([x["mape"] for x in per_iter])), 3) if per_iter else None),
        "avg_smape": (round(float(np.mean([x["smape"] for x in per_iter])), 3) if per_iter else None),
    }


def _safe_fmt(x, fmt=".3f"):
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "—"
        return format(x, fmt)
    except Exception:
        return "—"


def benchmark_models(
    model_classes,
    data,
    iterations=1,
    h=7,
    ensembles=("mean", "median"),
    exclude_from_ensemble=None,
):
    """
    (docstring unchanged)
    """
    data = np.asarray(data)
    assert len(data) > h, "Data length must be greater than forecast horizon h."
    y_true = data[-h:]
    model_classes = list(model_classes)

    if exclude_from_ensemble is None:
        exclude_from_ensemble = []
    exclude_names = {(cls.__name__ if not isinstance(cls, str) else cls) for cls in exclude_from_ensemble}

    results = {
        "meta": {"iterations": iterations, "h": h, "n_models": len(model_classes)},
        "models": {},
        "ensembles": {},
    }

    # Prepare per-model storage
    per_model_iters = {cls.__name__: [] for cls in model_classes}
    failed_models = set()

    # Per-iteration: collect predictions for ensembles
    ens_iters_preds = []  # list per iteration: 2D array [n_models_used x h]

    for i in range(iterations):
        iter_preds = []
        for cls in model_classes:
            model_name = cls.__name__
            try:
                # fresh model each run
                t0 = time.time()
                model = cls()
                model.fit(data[:-h])
                fit_time = time.time() - t0

                t1 = time.time()
                y_pred = model.predict(h)
                predict_time = time.time() - t1

                y_pred = _coerce_forecast(y_pred, h, model_name)
                if np.isnan(y_pred).any():
                    print(f"Skipping model {model_name}: NaN in predictions")
                    failed_models.add(model_name)
                    continue

                total_time = fit_time + predict_time

                # metrics
                m = _metrics_dict(y_true, y_pred)

                per_model_iters[model_name].append(
                    {
                        "fit_time_s": float(fit_time),
                        "predict_time_s": float(predict_time),
                        "total_time_s": float(total_time),
                        **m,
                    }
                )

                # include in ensemble only if not excluded
                if model_name not in exclude_names:
                    iter_preds.append(y_pred)

            except Exception as e:
                print(f"Skipping model {model_name}: {e}")
                failed_models.add(model_name)
                continue

        # Store stacked predictions for ensembles this iteration
        if iter_preds:
            ens_iters_preds.append(np.vstack(iter_preds))  # shape: (n_used_models, h)

    # Aggregate per-model (only include models with at least one valid iteration)
    for model_name, per_iter in per_model_iters.items():
        if not per_iter:
            # Do not include empty models to avoid None in summary formatting
            continue
        results["models"][model_name] = {
            **_avg_block(per_iter),
            "per_iteration": per_iter,
        }

    # Compute ensembles (metrics only; no timing)
    valid_ens = set([e.lower() for e in ensembles]) if ensembles else set()
    for ens_type in ("mean", "median"):
        if ens_type in valid_ens and ens_iters_preds:
            per_iter_metrics = []
            for stacked in ens_iters_preds:
                if stacked.size == 0:
                    continue
                if ens_type == "mean":
                    y_ens = np.nanmean(stacked, axis=0)
                else:  # median
                    y_ens = np.nanmedian(stacked, axis=0)
                per_iter_metrics.append(_metrics_dict(y_true, y_ens))

            if per_iter_metrics:
                results["ensembles"][ens_type] = {
                    "avg_mae": round(float(np.mean([x["mae"] for x in per_iter_metrics])), 3),
                    "avg_rmse": round(float(np.mean([x["rmse"] for x in per_iter_metrics])), 3),
                    "avg_mape": round(float(np.mean([x["mape"] for x in per_iter_metrics])), 3),
                    "avg_smape": round(float(np.mean([x["smape"] for x in per_iter_metrics])), 3),
                    "per_iteration": per_iter_metrics,
                }

    # Optional: brief console summary (use safe formatting)
    print(f"\nBenchmark over {iterations} runs (h={h})")
    if failed_models:
        print(f"(Some models were skipped due to errors/NaNs: {sorted(failed_models)})")

    return results
