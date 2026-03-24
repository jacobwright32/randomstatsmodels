#!/usr/bin/env python
"""Benchmark the 3 new ensemble models on all 36 datasets."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from randomstatsmodels import AutoStacked, AutoBagged, AutoDynamic
from randomstatsmodels.benchmarking.datasets import load_datasets
from randomstatsmodels.benchmarking.evaluation import evaluate_all, print_summary

datasets = load_datasets()
model_classes = [AutoStacked, AutoBagged, AutoDynamic]

print(f"Running 3 ensemble models on {len(datasets)} datasets...\n")

out = evaluate_all(model_classes, datasets, test_fraction=0.2,
                   min_test=6, season=1, verbose=True)

print_summary(out["summary"])

print("\nPer-dataset results:")
for r in out["results"]:
    if r["error"]:
        print(f"  {r['dataset']:<25} {r['model']:<20} FAIL: {r['error'][:40]}")
    else:
        print(f"  {r['dataset']:<25} {r['model']:<20} MAE={r['mae']:.4f}  sMAPE={r['smape']:.4f}")
