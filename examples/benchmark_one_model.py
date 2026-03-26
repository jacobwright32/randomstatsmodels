#!/usr/bin/env python
"""
Run one model across all 100 datasets at all 5 speeds.
Prints a markdown summary table ready for README.

Usage: python benchmark_one_model.py AutoNaive
"""
import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from randomstatsmodels.benchmarking.datasets import load_datasets, train_test_split
from randomstatsmodels.benchmarking.evaluation import _compute_metrics
from randomstatsmodels.presets import get_model_class, SPEEDS

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else None
if not MODEL_NAME:
    print("Usage: python benchmark_one_model.py <ModelName>")
    sys.exit(1)

cls = get_model_class(MODEL_NAME)


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
    total = n_ds * len(SPEEDS)
    print(f"{MODEL_NAME}: {len(SPEEDS)} speeds x {n_ds} datasets = {total} runs\n")

    cols = ["model", "speed", "dataset", "mae", "rmse", "mape", "smape",
            "fit_time", "n_train", "n_test", "error"]
    path = f"benchmark_{MODEL_NAME}_results.csv"

    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

    agg = {s: [] for s in SPEEDS}
    done = 0

    for speed in SPEEDS:
        for ds_name, ds in datasets.items():
            y = ds["values"]
            y_train, y_test = train_test_split(y, 0.2, 6)
            done += 1
            print(f"[{done}/{total}] {speed} {ds_name} (n={len(y)}) ...", end=" ", flush=True)

            instance = cls(speed=speed)
            res = run_one(instance, y_train, y_test)

            if res.get("error"):
                print(f"FAIL")
            else:
                print(f"RMSE={res['rmse']:.4f} fit={res['fit_time']:.2f}s")
                agg[speed].append(res)

            row = {"model": MODEL_NAME, "speed": speed, "dataset": ds_name,
                   "n_train": len(y_train), "n_test": len(y_test)}
            for c in cols:
                if c not in row:
                    row[c] = res.get(c)
            clean = {c: ("" if row.get(c) is None else
                         f"{row[c]:.6f}" if isinstance(row.get(c), float) and np.isfinite(row[c]) else
                         str(row.get(c, ""))) for c in cols}
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow(clean)

        # Print speed summary
        runs = agg[speed]
        if runs:
            def med(k):
                v = [r[k] for r in runs if r.get(k) is not None and np.isfinite(r[k])]
                return np.median(v) if v else float("nan")
            print(f"  >> {speed}: MdnRMSE={med('rmse'):.2f} MdnSMAPE={med('smape'):.2f}% "
                  f"MdnFit={med('fit_time'):.2f}s ({len(runs)}/{n_ds} ok)\n")

    # Final markdown table
    print(f"\n### {MODEL_NAME}\n")
    print(f"| Speed | Median MAE | Median RMSE | Median MAPE | Median sMAPE | Median Fit (s) | OK |")
    print(f"|-------|-----------|------------|------------|-------------|---------------|-----|")
    for speed in SPEEDS:
        runs = agg[speed]
        if not runs:
            print(f"| {speed} | - | - | - | - | - | 0 |")
            continue
        def med(k):
            v = [r[k] for r in runs if r.get(k) is not None and np.isfinite(r[k])]
            return np.median(v) if v else float("nan")
        print(f"| {speed} | {med('mae'):.2f} | {med('rmse'):.2f} | "
              f"{med('mape'):.2f}% | {med('smape'):.2f}% | {med('fit_time'):.2f} | {len(runs)} |")

    print(f"\nResults: {path}")


if __name__ == "__main__":
    main()
