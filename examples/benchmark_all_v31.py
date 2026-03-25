#!/usr/bin/env python
"""
Full benchmark: all randomstatsmodels + statsforecast + ensemble models
on 36 real-world datasets. Outputs CSV with model_type column.
"""
import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from randomstatsmodels.benchmarking.datasets import load_datasets, train_test_split
from randomstatsmodels.benchmarking.evaluation import _compute_metrics

# --- Model registry with types ---
MODELS = []

def reg(cls, model_type):
    MODELS.append((cls, model_type))

# randomstatsmodels single models
from randomstatsmodels import (
    AutoNaive, AutoHoltWinters, AutoFourier, AutoNEO, AutoKNN,
    AutoPolymath, AutoThetaAR, AutoSSA, AutoLocalLinear, AutoPALF,
    AutoMELD, AutoRIFT, AutoHybridForecaster,
    AutoFracDiff, AutoSpectralGradient, AutoGreensKernel,
    AutoPDEField, AutoVariationalPath, AutoKoopman,
)
for m in [AutoNaive, AutoHoltWinters, AutoFourier, AutoNEO, AutoKNN,
          AutoPolymath, AutoThetaAR, AutoSSA, AutoLocalLinear, AutoPALF,
          AutoMELD, AutoRIFT, AutoHybridForecaster,
          AutoFracDiff, AutoSpectralGradient, AutoGreensKernel,
          AutoPDEField, AutoVariationalPath, AutoKoopman]:
    reg(m, "randomstatsmodels")

# ensembles
from randomstatsmodels import AutoStacked, AutoBagged, AutoDynamic
for m in [AutoStacked, AutoBagged, AutoDynamic]:
    reg(m, "ensemble")

# statsforecast
from randomstatsmodels.benchmarking.statsforecast_wrappers import (
    SFAutoARIMA, SFAutoETS, SFAutoCES, SFAutoTheta, SFAutoTBATS,
)
for m in [SFAutoARIMA, SFAutoETS, SFAutoCES, SFAutoTheta, SFAutoTBATS]:
    reg(m, "statsforecast")


def run_one(model_class, y_train, y_test):
    h = len(y_test)
    try:
        t0 = time.perf_counter()
        m = model_class()
        m.fit(y_train)
        fit_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds = np.asarray(m.predict(h), float).ravel()[:h]
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
    n_mod = len(MODELS)
    print(f"Benchmark: {n_mod} models x {n_ds} datasets = {n_mod * n_ds} evaluations\n")

    # CSV output
    cols = ["dataset", "model", "model_type", "mae", "rmse", "mape", "smape",
            "median_ae", "mase", "msse", "fit_time", "pred_time",
            "n_train", "n_test", "error"]
    rows = []

    for i, (ds_name, ds) in enumerate(datasets.items()):
        y = ds["values"]
        y_train, y_test = train_test_split(y, 0.2, 6)
        print(f"[{i+1}/{n_ds}] {ds_name} (train={len(y_train)}, test={len(y_test)})")

        for model_class, model_type in MODELS:
            name = getattr(model_class, '__name__', model_class.__name__
                           if hasattr(model_class, '__name__') else str(model_class))
            print(f"   {name} ...", end=" ", flush=True)
            res = run_one(model_class, y_train, y_test)
            if res.get("error"):
                print(f"FAIL: {res['error'][:50]}")
            else:
                print(f"MAE={res['mae']:.4f}  sMAPE={res['smape']:.4f}")

            row = {"dataset": ds_name, "model": name, "model_type": model_type,
                   "n_train": len(y_train), "n_test": len(y_test)}
            for c in cols:
                if c not in row:
                    row[c] = res.get(c)
            rows.append(row)

    # Write CSV
    with open("benchmark_v31_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            clean = {}
            for c in cols:
                v = r.get(c)
                if v is None:
                    clean[c] = ""
                elif isinstance(v, float):
                    clean[c] = f"{v:.6f}" if np.isfinite(v) else ""
                else:
                    clean[c] = str(v)
            w.writerow(clean)

    # Compute rankings
    from collections import defaultdict
    rank_sums = defaultdict(list)
    first_place = defaultdict(int)
    top3 = defaultdict(int)
    smape_vals = defaultdict(list)
    model_types = {}

    ds_names = sorted(set(r["dataset"] for r in rows))
    for ds in ds_names:
        ds_rows = [(r["model"], r.get("mae"), r.get("smape"), r["model_type"])
                   for r in rows if r["dataset"] == ds
                   and r.get("error") is None
                   and r.get("mae") is not None
                   and isinstance(r.get("mae"), (int, float))
                   and np.isfinite(r["mae"])]
        ds_rows.sort(key=lambda x: x[1])
        for rank, (model, _, sm, mt) in enumerate(ds_rows, 1):
            rank_sums[model].append(rank)
            model_types[model] = mt
            if sm is not None and np.isfinite(sm):
                smape_vals[model].append(sm)
            if rank == 1:
                first_place[model] += 1
            if rank <= 3:
                top3[model] += 1

    # Print rankings
    print(f"\n{'='*95}")
    print(f"{'Model':<28} {'Type':<18} {'AvgRank':>8} {'#1st':>5} {'#Top3':>6} {'MdnSMAPE':>10}")
    print(f"{'-'*95}")
    ranked = sorted(rank_sums.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    for model, ranks in ranked:
        avg = sum(ranks) / len(ranks)
        sm = float(np.median(smape_vals[model])) if smape_vals[model] else float("nan")
        mt = model_types.get(model, "?")
        print(f"{model:<28} {mt:<18} {avg:>8.2f} {first_place[model]:>5} {top3[model]:>6} {sm:>10.2f}")
    print(f"{'='*95}")

    # Write summary CSV
    with open("benchmark_v31_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "model_type", "avg_rank", "n_first", "n_top3", "median_smape"])
        for model, ranks in ranked:
            avg = sum(ranks) / len(ranks)
            sm = float(np.median(smape_vals[model])) if smape_vals[model] else ""
            mt = model_types.get(model, "?")
            w.writerow([model, mt, f"{avg:.4f}", first_place[model], top3[model],
                        f"{sm:.4f}" if isinstance(sm, float) and np.isfinite(sm) else ""])

    print("\nResults: benchmark_v31_results.csv, benchmark_v31_summary.csv")


if __name__ == "__main__":
    main()
