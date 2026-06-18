"""Run the BELLWETHER drift-detection benchmark and print results with confidence intervals.

Usage:
    uv run python examples/benchmark.py            # full default benchmark
    uv run python examples/benchmark.py --quick    # fast, smaller config

The reported numbers are over N independent seeds with 95% bootstrap CIs (brief §6). The
ground truth is exact because faults are injected by the fixture harness (eval design doc 02).
"""

from __future__ import annotations

import sys
import time

from bellwether.eval import BenchmarkConfig, run_benchmark


def main() -> None:
    quick = "--quick" in sys.argv
    cfg = (
        BenchmarkConfig(
            n_train=50, n_cal=30, n_eval_benign=40, n_eval_per_fault=8, n_seeds=5, min_samples=20
        )
        if quick
        else BenchmarkConfig()
    )

    t0 = time.perf_counter()
    report = run_benchmark(config=cfg)
    elapsed = time.perf_counter() - t0

    print(report.render())
    print(f"\n  (completed in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
