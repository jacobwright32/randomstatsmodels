#!/usr/bin/env python
"""
Comprehensive benchmark: all models × all 36 real-world datasets.

Usage:
    python examples/comprehensive_benchmark.py

Outputs:
    benchmark_full_results.csv    — per-dataset, per-model metrics
    benchmark_full_summary.csv    — model rankings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from randomstatsmodels import (
    AutoNEO,
    AutoFourier,
    AutoKNN,
    AutoPolymath,
    AutoThetaAR,
    AutoHybridForecaster,
    AutoMELD,
    AutoPALF,
    AutoNaive,
    AutoHoltWinters,
    AutoSSA,
    AutoLocalLinear,
    AutoRIFT,
    AutoFracDiff,
    AutoSpectralGradient,
    AutoGreensKernel,
    AutoPDEField,
    AutoVariationalPath,
)
from randomstatsmodels.benchmarking.datasets import load_datasets
from randomstatsmodels.benchmarking.evaluation import (
    evaluate_all,
    results_to_csv,
    summary_to_csv,
    print_summary,
)


def main():
    # All Auto* model classes
    model_classes = [
        # --- Existing models ---
        AutoNaive,
        AutoHoltWinters,
        AutoFourier,
        AutoNEO,
        AutoKNN,
        AutoPolymath,
        AutoThetaAR,
        AutoSSA,
        AutoLocalLinear,
        AutoPALF,
        AutoMELD,
        AutoRIFT,
        AutoHybridForecaster,
        # --- New calculus-based models ---
        AutoFracDiff,
        AutoSpectralGradient,
        AutoGreensKernel,
        AutoPDEField,
        AutoVariationalPath,
    ]

    datasets = load_datasets()

    print(f"Running benchmark: {len(model_classes)} models × "
          f"{len(datasets)} datasets\n")

    out = evaluate_all(
        model_classes,
        datasets,
        test_fraction=0.2,
        min_test=6,
        season=1,
        verbose=True,
    )

    # Write results
    results_to_csv(out["results"], "benchmark_full_results.csv")
    summary_to_csv(out["summary"], "benchmark_full_summary.csv")
    print_summary(out["summary"])

    print("Results written to:")
    print("  benchmark_full_results.csv")
    print("  benchmark_full_summary.csv")


if __name__ == "__main__":
    main()
