"""Evaluation harness: ground-truth benchmark for detection quality (design doc 02).

L1 of the layered harness: build a baseline from benign runs, calibrate on a benign hold-out,
score an eval split of benign + injected-fault runs, and report precision / recall / F1 /
false-positive-rate / lead-time. Everything is run over N independent seeds and reported with
bootstrap confidence intervals — never single-run numbers (brief §6).
"""

from bellwether.eval.harness import BenchmarkConfig, BenchmarkReport, run_benchmark
from bellwether.eval.metrics import EpisodeOutcome, MetricSet, evaluate_episode, metric_set

__all__ = [
    "BenchmarkConfig",
    "BenchmarkReport",
    "run_benchmark",
    "EpisodeOutcome",
    "MetricSet",
    "evaluate_episode",
    "metric_set",
]
