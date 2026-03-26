#!/usr/bin/env python
"""Benchmark: slow models (>30s fit) on 20 datasets with speed=slow grids."""
import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from randomstatsmodels.benchmarking.datasets import load_datasets, train_test_split
from randomstatsmodels.benchmarking.evaluation import _compute_metrics
from randomstatsmodels.presets import get_model_class

# Slow models (>30s on 200pts with default grids)
SLOW_MODELS = [
    ("AutoRIFT", "randomstatsmodels"),
    ("AutoHybridForecaster", "randomstatsmodels"),
]

SPEED = "slow"

def run_one(model, y_train, y_test):
    h = len(y_test)
    try:
        t0 = time.perf_counter()
        model.fit(y_train)
        fit_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds = np.asarray(model.predict(h), float).ravel()[:h]
        pred_t = time.perf_counter() - t1
        if len(preds) < h:
            preds = np.pad(preds, (0, h - len(preds)), constant_values=np.nan)
        metrics = _compute_metrics(y_test, preds, y_train, season=1)
        return {**metrics, "fit_time": fit_t, "pred_time": pred_t, "error": None}
    except Exception as e:
        return {"error": str(e)}


def main():
    datasets = load_datasets()
    n_ds = len(datasets)
    total = len(SLOW_MODELS) * n_ds
    print(f"Slow benchmark (speed={SPEED}): {len(SLOW_MODELS)} models x {n_ds} datasets = {total}\n")

    cols = ["dataset", "model", "model_type", "speed", "mae", "rmse", "mape",
            "smape", "median_ae", "mase", "msse", "fit_time", "pred_time",
            "n_train", "n_test", "error"]
    path = "benchmark_slow_results.csv"

    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

    for i, (ds_name, ds) in enumerate(datasets.items()):
        y = ds["values"]
        y_train, y_test = train_test_split(y, 0.2, 6)
        print(f"[{i+1}/{n_ds}] {ds_name} (n={len(y)}, train={len(y_train)}, test={len(y_test)})")

        for name, model_type in SLOW_MODELS:
            cls = get_model_class(name)
            instance = cls(speed=SPEED)
            print(f"   {name} ...", end=" ", flush=True)
            res = run_one(instance, y_train, y_test)
            if res.get("error"):
                print(f"FAIL: {res['error'][:50]}")
            else:
                print(f"MAE={res['mae']:.4f}  sMAPE={res['smape']:.4f}  fit={res['fit_time']:.1f}s")

            row = {"dataset": ds_name, "model": name, "model_type": model_type,
                   "speed": SPEED, "n_train": len(y_train), "n_test": len(y_test)}
            for c in cols:
                if c not in row:
                    row[c] = res.get(c)

            clean = {c: ("" if row.get(c) is None else
                         f"{row[c]:.6f}" if isinstance(row.get(c), float) and np.isfinite(row[c]) else
                         str(row.get(c, ""))) for c in cols}
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow(clean)

    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
