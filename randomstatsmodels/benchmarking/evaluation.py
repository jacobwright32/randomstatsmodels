"""
Comprehensive evaluation framework for time-series forecasting models.

Provides multi-metric, multi-horizon evaluation with ranking, statistical
summaries, and CSV export — all NumPy-only.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..metrics import mae, mse, rmse, mape, smape


# -----------------------------------------------------------------------
# Additional metrics (beyond the 5 already in the package)
# -----------------------------------------------------------------------

def mase(y_true: np.ndarray, y_pred: np.ndarray,
         y_train: np.ndarray, season: int = 1) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    y_train = np.asarray(y_train, float)
    naive_errors = np.abs(y_train[season:] - y_train[:-season])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    if scale < 1e-12:
        scale = 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Error."""
    return float(np.median(np.abs(np.asarray(y_true, float)
                                  - np.asarray(y_pred, float))))


def msse(y_true: np.ndarray, y_pred: np.ndarray,
         y_train: np.ndarray, season: int = 1) -> float:
    """Mean Squared Scaled Error (root gives RMSSE)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    y_train = np.asarray(y_train, float)
    naive_sq = (y_train[season:] - y_train[:-season]) ** 2
    scale = np.mean(naive_sq) if len(naive_sq) > 0 else 1.0
    if scale < 1e-12:
        scale = 1.0
    return float(np.mean((y_true - y_pred) ** 2) / scale)


# -----------------------------------------------------------------------
# Core evaluation helpers
# -----------------------------------------------------------------------

_METRIC_FNS = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "smape": smape,
    "median_ae": median_ae,
}


def _compute_metrics(y_true, y_pred, y_train=None, season=1):
    """Compute all metrics for a single forecast."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    result = {
        "mae":  mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "median_ae": median_ae(y_true, y_pred),
    }
    if y_train is not None:
        result["mase"] = mase(y_true, y_pred, y_train, season)
        result["msse"] = msse(y_true, y_pred, y_train, season)
    return result


def _safe(val, fmt=".4f"):
    if val is None or not np.isfinite(val):
        return "N/A"
    return format(val, fmt)


# -----------------------------------------------------------------------
# Single-model evaluator
# -----------------------------------------------------------------------

def evaluate_model(model_class, y_train, y_test, *,
                   season: int = 1,
                   timeout: float = 300.0) -> dict:
    """
    Fit *model_class* on y_train, predict len(y_test) steps.

    Returns dict with timing, predictions, and all metrics.
    On failure returns dict with 'error' key.
    """
    h = len(y_test)
    try:
        t0 = time.perf_counter()
        model = model_class()
        model.fit(y_train)
        t_fit = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(h)
        t_pred = time.perf_counter() - t1

        y_pred = np.asarray(y_pred, float).ravel()[:h]
        if len(y_pred) < h:
            y_pred = np.pad(y_pred, (0, h - len(y_pred)),
                            constant_values=np.nan)

        metrics = _compute_metrics(y_test, y_pred, y_train, season)
        return {
            "model": model_class.__name__,
            "fit_time": t_fit,
            "pred_time": t_pred,
            "predictions": y_pred,
            **metrics,
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model_class.__name__,
            "fit_time": None,
            "pred_time": None,
            "predictions": None,
            "error": str(exc),
        }


# -----------------------------------------------------------------------
# Multi-model, multi-dataset evaluation
# -----------------------------------------------------------------------

def evaluate_all(model_classes: list,
                 datasets: Dict[str, dict],
                 test_fraction: float = 0.2,
                 min_test: int = 6,
                 season: int = 1,
                 verbose: bool = True) -> dict:
    """
    Run every model on every dataset.

    Parameters
    ----------
    model_classes : list of classes
        Each must support ``__init__() -> .fit(y) -> .predict(h)``.
    datasets : dict
        name -> {"values": np.ndarray, ...}  (as returned by load_datasets).
    test_fraction, min_test : float, int
        How to split each series.
    season : int
        Seasonal period for MASE/MSSE (1 = non-seasonal naive).
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        "results"  : list[dict]  — one row per (dataset, model)
        "rankings" : dict        — per-metric model rankings
        "summary"  : dict        — per-model aggregated scores
    """
    from .datasets import train_test_split

    results = []
    n_ds = len(datasets)
    n_mod = len(model_classes)

    for i, (ds_name, ds) in enumerate(datasets.items()):
        y = ds["values"]
        y_train, y_test = train_test_split(y, test_fraction, min_test)
        if verbose:
            print(f"[{i+1}/{n_ds}] {ds_name}  "
                  f"(train={len(y_train)}, test={len(y_test)})")

        for mc in model_classes:
            if verbose:
                print(f"   {mc.__name__} ... ", end="", flush=True)
            res = evaluate_model(mc, y_train, y_test, season=season)
            res["dataset"] = ds_name
            res["n_train"] = len(y_train)
            res["n_test"] = len(y_test)
            results.append(res)
            if verbose:
                if res["error"]:
                    print(f"FAIL: {res['error'][:60]}")
                else:
                    print(f"MAE={_safe(res['mae'])}  "
                          f"sMAPE={_safe(res['smape'])}")

    # Build rankings
    rankings = _build_rankings(results, model_classes, datasets)
    summary = _build_summary(results, model_classes, datasets)

    return {"results": results, "rankings": rankings, "summary": summary}


# -----------------------------------------------------------------------
# Ranking helpers
# -----------------------------------------------------------------------

_RANK_METRICS = ("mae", "rmse", "smape", "mase")


def _build_rankings(results, model_classes, datasets):
    """Per-dataset, per-metric ranking of models (1 = best)."""
    model_names = [mc.__name__ for mc in model_classes]
    ds_names = list(datasets.keys())
    rankings = {}

    for metric in _RANK_METRICS:
        rankings[metric] = {}
        for ds in ds_names:
            scores = {}
            for r in results:
                if r["dataset"] == ds and r["error"] is None:
                    val = r.get(metric)
                    if val is not None and np.isfinite(val):
                        scores[r["model"]] = val
            if not scores:
                continue
            sorted_models = sorted(scores, key=scores.get)
            rank_map = {m: i + 1 for i, m in enumerate(sorted_models)}
            # assign worst rank to failed models
            for m in model_names:
                if m not in rank_map:
                    rank_map[m] = len(model_names)
            rankings[metric][ds] = rank_map

    return rankings


def _build_summary(results, model_classes, datasets):
    """Aggregate rankings into per-model summary table."""
    model_names = [mc.__name__ for mc in model_classes]
    ds_names = list(datasets.keys())
    n_ds = len(ds_names)

    rankings = _build_rankings(results, model_classes, datasets)
    summary = {}

    for m in model_names:
        row = {"model": m}
        for metric in _RANK_METRICS:
            ranks = []
            for ds in ds_names:
                rm = rankings[metric].get(ds, {})
                ranks.append(rm.get(m, len(model_names)))
            row[f"avg_rank_{metric}"] = np.mean(ranks)
            row[f"n_first_{metric}"] = sum(1 for r in ranks if r == 1)
            row[f"n_top3_{metric}"] = sum(1 for r in ranks if r <= 3)

        # overall average rank (across metrics)
        avg_ranks = [row[f"avg_rank_{metric}"] for metric in _RANK_METRICS]
        row["overall_avg_rank"] = np.mean(avg_ranks)

        # median raw scores
        model_results = [r for r in results
                         if r["model"] == m and r["error"] is None]
        for metric in ("mae", "rmse", "smape", "mape"):
            vals = [r[metric] for r in model_results
                    if r.get(metric) is not None and np.isfinite(r[metric])]
            row[f"median_{metric}"] = np.median(vals) if vals else np.nan

        n_fail = sum(1 for r in results
                     if r["model"] == m and r["error"] is not None)
        row["n_failures"] = n_fail
        summary[m] = row

    return summary


# -----------------------------------------------------------------------
# Output / export
# -----------------------------------------------------------------------

def results_to_csv(results: list, path: str):
    """Write evaluation results to a CSV file."""
    cols = ["dataset", "model", "mae", "rmse", "mape", "smape",
            "median_ae", "mase", "msse", "fit_time", "pred_time",
            "n_train", "n_test", "error"]
    lines = [",".join(cols)]
    for r in results:
        vals = []
        for c in cols:
            v = r.get(c)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.6f}" if np.isfinite(v) else "")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    with open(path, "w", newline="") as f:
        f.write("\n".join(lines) + "\n")


def summary_to_csv(summary: dict, path: str):
    """Write model summary / rankings to CSV."""
    if not summary:
        return
    first = next(iter(summary.values()))
    cols = list(first.keys())
    lines = [",".join(cols)]
    # sort by overall rank
    for m, row in sorted(summary.items(),
                         key=lambda kv: kv[1].get("overall_avg_rank", 999)):
        vals = []
        for c in cols:
            v = row.get(c)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}" if np.isfinite(v) else "")
            elif isinstance(v, int):
                vals.append(str(v))
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    with open(path, "w", newline="") as f:
        f.write("\n".join(lines) + "\n")


def print_summary(summary: dict):
    """Pretty-print the model summary table."""
    if not summary:
        print("No summary available.")
        return

    sorted_models = sorted(summary.values(),
                           key=lambda r: r.get("overall_avg_rank", 999))

    print("\n" + "=" * 95)
    print(f"{'Model':<28} {'AvgRank':>8} {'#1st':>5} {'#Top3':>6} "
          f"{'MdnMAE':>10} {'MdnsMAPE':>10} {'Fails':>6}")
    print("-" * 95)

    for row in sorted_models:
        n_first = row.get("n_first_mae", 0) + row.get("n_first_smape", 0)
        n_top3 = row.get("n_top3_mae", 0) + row.get("n_top3_smape", 0)
        print(f"{row['model']:<28} "
              f"{_safe(row['overall_avg_rank'], '.2f'):>8} "
              f"{n_first:>5} "
              f"{n_top3:>6} "
              f"{_safe(row.get('median_mae', float('nan')), '.3f'):>10} "
              f"{_safe(row.get('median_smape', float('nan')), '.3f'):>10} "
              f"{row['n_failures']:>6}")

    print("=" * 90 + "\n")
