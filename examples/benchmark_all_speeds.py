#!/usr/bin/env python
"""
Benchmark every model at every speed level on all 20 datasets.
Models ordered fastest-first. Results saved incrementally.

Output: benchmark_all_speeds_results.csv (per model-dataset-speed row)
        benchmark_all_speeds_summary.csv (per model-speed averages)
"""
import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from randomstatsmodels.benchmarking.datasets import load_datasets, train_test_split
from randomstatsmodels.benchmarking.evaluation import _compute_metrics
from randomstatsmodels.presets import get_model_class, SPEEDS

# Models ordered by fit time (fastest first, measured on 200pts at speed=normal)
MODELS_FASTEST_FIRST = [
    "AutoNaive",            # 0.01s
    "AutoThetaAR",          # 0.01s
    "AutoKNN",              # 0.03s
    "AutoPDEField",         # 0.07s
    "AutoFourier",          # 0.09s
    "AutoGreensKernel",     # 0.10s
    "AutoSpectralGradient", # 0.13s
    "AutoFracDiff",         # 0.13s
    "AutoKoopman",          # 0.37s
    "AutoVariationalPath",  # 0.40s
    "AutoNEO",              # 0.51s
    "AutoLocalLinear",      # 0.63s
    "AutoPolymath",         # 0.99s
    "AutoSSA",              # 1.88s
    "AutoPALF",             # 3.42s
    "AutoMELD",             # 6.43s
    "AutoHoltWinters",      # 17.5s
    "AutoRIFT",             # 104s
    "AutoHybridForecaster", # 253s
]


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
    n_mod = len(MODELS_FASTEST_FIRST)
    n_speeds = len(SPEEDS)
    total = n_mod * n_ds * n_speeds
    print(f"Full speed benchmark: {n_mod} models x {n_speeds} speeds x {n_ds} datasets = {total} runs\n")

    cols = ["model", "speed", "dataset", "mae", "rmse", "mape", "smape",
            "fit_time", "pred_time", "n_train", "n_test", "error"]
    results_path = "benchmark_all_speeds_results.csv"

    with open(results_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

    # Track per model-speed aggregates
    agg = {}  # (model, speed) -> list of metric dicts

    done = 0
    for model_name in MODELS_FASTEST_FIRST:
        cls = get_model_class(model_name)

        for speed in SPEEDS:
            key = (model_name, speed)
            agg[key] = []

            for ds_name, ds in datasets.items():
                y = ds["values"]
                y_train, y_test = train_test_split(y, 0.2, 6)
                done += 1

                print(f"[{done}/{total}] {model_name} speed={speed} {ds_name} "
                      f"(n={len(y)}) ...", end=" ", flush=True)

                instance = cls(speed=speed)
                res = run_one(instance, y_train, y_test)

                if res.get("error"):
                    print(f"FAIL: {res['error'][:40]}")
                else:
                    print(f"RMSE={res['rmse']:.4f} fit={res['fit_time']:.1f}s")
                    agg[key].append(res)

                # Save row
                row = {"model": model_name, "speed": speed, "dataset": ds_name,
                       "n_train": len(y_train), "n_test": len(y_test)}
                for c in cols:
                    if c not in row:
                        row[c] = res.get(c)

                clean = {}
                for c in cols:
                    v = row.get(c)
                    if v is None:
                        clean[c] = ""
                    elif isinstance(v, float):
                        clean[c] = f"{v:.6f}" if np.isfinite(v) else ""
                    else:
                        clean[c] = str(v)
                with open(results_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=cols).writerow(clean)

            # Print running summary for this model-speed
            runs = agg[key]
            if runs:
                med_rmse = np.median([r["rmse"] for r in runs if np.isfinite(r.get("rmse", float("nan")))])
                med_smape = np.median([r["smape"] for r in runs if np.isfinite(r.get("smape", float("nan")))])
                med_fit = np.median([r["fit_time"] for r in runs])
                print(f"  >> {model_name} [{speed}]: median RMSE={med_rmse:.4f} "
                      f"sMAPE={med_smape:.2f}% fit={med_fit:.1f}s ({len(runs)}/{n_ds} ok)\n")

    # Write summary
    summary_path = "benchmark_all_speeds_summary.csv"
    summary_cols = ["model", "speed", "n_ok", "median_mae", "median_rmse",
                    "median_mape", "median_smape", "median_fit_time"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for model_name in MODELS_FASTEST_FIRST:
            for speed in SPEEDS:
                runs = agg.get((model_name, speed), [])
                if not runs:
                    continue
                def med(key):
                    vals = [r[key] for r in runs if r.get(key) is not None and np.isfinite(r[key])]
                    return np.median(vals) if vals else float("nan")
                w.writerow({
                    "model": model_name, "speed": speed, "n_ok": len(runs),
                    "median_mae": f"{med('mae'):.4f}",
                    "median_rmse": f"{med('rmse'):.4f}",
                    "median_mape": f"{med('mape'):.4f}",
                    "median_smape": f"{med('smape'):.4f}",
                    "median_fit_time": f"{med('fit_time'):.2f}",
                })

    # Print final table
    print(f"\n{'='*110}")
    print(f"{'Model':<25} {'Speed':<12} {'OK':>4} {'MdnRMSE':>10} {'MdnSMAPE':>10} {'MdnFitTime':>12}")
    print(f"{'-'*110}")
    for model_name in MODELS_FASTEST_FIRST:
        for speed in SPEEDS:
            runs = agg.get((model_name, speed), [])
            if not runs:
                continue
            def med(key):
                vals = [r[key] for r in runs if r.get(key) is not None and np.isfinite(r[key])]
                return np.median(vals) if vals else float("nan")
            print(f"{model_name:<25} {speed:<12} {len(runs):>4} "
                  f"{med('rmse'):>10.2f} {med('smape'):>10.2f} {med('fit_time'):>10.1f}s")
        print()
    print(f"{'='*110}")
    print(f"\nResults: {results_path}, {summary_path}")


if __name__ == "__main__":
    main()
