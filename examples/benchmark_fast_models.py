#!/usr/bin/env python
"""Benchmark: fast models (<30s fit) on 20 datasets with speed=slow grids."""
import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from randomstatsmodels.benchmarking.datasets import load_datasets, train_test_split
from randomstatsmodels.benchmarking.evaluation import _compute_metrics
from randomstatsmodels.presets import get_model_class, get_grid

# Fast models (<30s on 200pts with default grids)
FAST_MODELS = [
    "AutoNaive", "AutoThetaAR", "AutoKNN", "AutoPDEField", "AutoFourier",
    "AutoGreensKernel", "AutoSpectralGradient", "AutoFracDiff",
    "AutoKoopman", "AutoVariationalPath", "AutoNEO", "AutoLocalLinear",
    "AutoPolymath", "AutoSSA", "AutoPALF", "AutoMELD", "AutoHoltWinters",
]

# statsforecast
from randomstatsmodels.benchmarking.statsforecast_wrappers import (
    SFAutoARIMA, SFAutoETS, SFAutoCES, SFAutoTheta, SFAutoTBATS,
)
SF_MODELS = [
    (SFAutoARIMA, "SF_AutoARIMA"), (SFAutoETS, "SF_AutoETS"),
    (SFAutoCES, "SF_AutoCES"), (SFAutoTheta, "SF_AutoTheta"),
    (SFAutoTBATS, "SF_AutoTBATS"),
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
    total = (len(FAST_MODELS) + len(SF_MODELS)) * n_ds
    print(f"Fast benchmark (speed={SPEED}): {len(FAST_MODELS)+len(SF_MODELS)} models x {n_ds} datasets = {total}\n")

    cols = ["dataset", "model", "model_type", "speed", "mae", "rmse", "mape",
            "smape", "median_ae", "mase", "msse", "fit_time", "pred_time",
            "n_train", "n_test", "error"]
    path = "benchmark_fast_results.csv"
    rows = []

    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

    for i, (ds_name, ds) in enumerate(datasets.items()):
        y = ds["values"]
        y_train, y_test = train_test_split(y, 0.2, 6)
        print(f"[{i+1}/{n_ds}] {ds_name} (n={len(y)}, train={len(y_train)}, test={len(y_test)})")

        # randomstatsmodels fast models
        for name in FAST_MODELS:
            cls = get_model_class(name)
            instance = cls(speed=SPEED)
            print(f"   {name} ...", end=" ", flush=True)
            res = run_one(instance, y_train, y_test)
            if res.get("error"):
                print(f"FAIL: {res['error'][:50]}")
            else:
                print(f"MAE={res['mae']:.4f}  sMAPE={res['smape']:.4f}  fit={res['fit_time']:.1f}s")

            row = {"dataset": ds_name, "model": name, "model_type": "randomstatsmodels",
                   "speed": SPEED, "n_train": len(y_train), "n_test": len(y_test)}
            for c in cols:
                if c not in row:
                    row[c] = res.get(c)
            rows.append(row)

            clean = {c: ("" if row.get(c) is None else
                         f"{row[c]:.6f}" if isinstance(row.get(c), float) and np.isfinite(row[c]) else
                         str(row.get(c, ""))) for c in cols}
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow(clean)

        # statsforecast
        for sf_cls, sf_name in SF_MODELS:
            instance = sf_cls()
            print(f"   {sf_name} ...", end=" ", flush=True)
            res = run_one(instance, y_train, y_test)
            if res.get("error"):
                print(f"FAIL: {res['error'][:50]}")
            else:
                print(f"MAE={res['mae']:.4f}  sMAPE={res['smape']:.4f}  fit={res['fit_time']:.1f}s")

            row = {"dataset": ds_name, "model": sf_name, "model_type": "statsforecast",
                   "speed": "n/a", "n_train": len(y_train), "n_test": len(y_test)}
            for c in cols:
                if c not in row:
                    row[c] = res.get(c)
            rows.append(row)

            clean = {c: ("" if row.get(c) is None else
                         f"{row[c]:.6f}" if isinstance(row.get(c), float) and np.isfinite(row[c]) else
                         str(row.get(c, ""))) for c in cols}
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow(clean)

    # Rankings by RMSE
    from collections import defaultdict
    all_ranks = defaultdict(list)
    first_place = defaultdict(int)
    top3 = defaultdict(int)
    top7 = defaultdict(int)
    model_types = {}
    metric_vals = defaultdict(lambda: defaultdict(list))

    ds_names = sorted(set(r["dataset"] for r in rows))
    for ds in ds_names:
        scored = [(r["model"], r.get("rmse"), r["model_type"]) for r in rows
                  if r["dataset"] == ds and r.get("error") is None
                  and r.get("rmse") is not None and isinstance(r.get("rmse"), (int, float))
                  and np.isfinite(r["rmse"])]
        scored.sort(key=lambda x: x[1])
        scored_set = set()
        for rank, (m, _, mt) in enumerate(scored, 1):
            all_ranks[m].append(rank)
            model_types[m] = mt
            scored_set.add(m)
            if rank == 1: first_place[m] += 1
            if rank <= 3: top3[m] += 1
            if rank <= 7: top7[m] += 1

        # Collect metric values
        for r in rows:
            if r["dataset"] == ds and r.get("error") is None:
                for met in ("mae", "rmse", "mape", "smape"):
                    v = r.get(met)
                    if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                        metric_vals[r["model"]][met].append(v)

    ranked = sorted(all_ranks.items(), key=lambda kv: np.mean(kv[1]))

    print(f"\n{'='*120}")
    print(f"{'Rk':<4} {'Model':<28} {'Type':<18} {'AvgRank':>8} {'#1st':>5} {'#Top3':>6} {'#Top7':>6} "
          f"{'MdnMAE':>10} {'MdnRMSE':>10} {'MdnMAPE':>10} {'MdnSMAPE':>10}")
    print(f"{'-'*120}")
    for i, (m, ranks) in enumerate(ranked, 1):
        avg = np.mean(ranks)
        mt = model_types.get(m, "?")
        mv = {k: np.median(v) if v else float("nan") for k, v in metric_vals[m].items()}
        print(f"{i:<4} {m:<28} {mt:<18} {avg:>8.2f} {first_place[m]:>5} {top3[m]:>6} {top7[m]:>6} "
              f"{mv.get('mae', float('nan')):>10.2f} {mv.get('rmse', float('nan')):>10.2f} "
              f"{mv.get('mape', float('nan')):>10.2f} {mv.get('smape', float('nan')):>10.2f}")
    print(f"{'='*120}")
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
