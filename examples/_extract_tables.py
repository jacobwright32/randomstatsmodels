"""Extract markdown tables from completed benchmark CSVs."""
import csv, numpy as np, os, glob

SPEEDS = ("super_fast", "fast", "normal", "slow", "super_slow")

for f in sorted(glob.glob("benchmark_Auto*_results.csv")):
    model = f.replace("benchmark_", "").replace("_results.csv", "")
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 500:
        continue

    print(f"#### {model}\n")
    print("| Speed | Mean MAE | Mean RMSE | Mean sMAPE | Median MAE | Median RMSE | Median sMAPE | Median Fit (s) | OK |")
    print("|-------|---------|----------|-----------|-----------|------------|-------------|---------------|-----|")

    for speed in SPEEDS:
        sr = [r for r in rows if r["speed"] == speed]
        ok = []
        for r in sr:
            try:
                mae_v = float(r["mae"])
                if np.isfinite(mae_v):
                    ok.append({k: float(r[k]) for k in ("mae", "rmse", "smape", "fit_time") if r.get(k)})
            except (ValueError, KeyError):
                pass

        if not ok:
            print(f"| {speed} | - | - | - | - | - | - | - | 0 |")
            continue

        def med(k): return np.median([r[k] for r in ok if k in r])
        def avg(k): return np.mean([r[k] for r in ok if k in r])

        print(f"| {speed} | {avg('mae'):.2f} | {avg('rmse'):.2f} | {avg('smape'):.2f}% "
              f"| {med('mae'):.2f} | {med('rmse'):.2f} | {med('smape'):.2f}% "
              f"| {med('fit_time'):.2f} | {len(ok)} |")
    print()
