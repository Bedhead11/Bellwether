"""Benchmark runner: N-seed evaluation with bootstrap confidence intervals.

For each seed we build disjoint train / calibrate / eval splits from the fixture agent, learn a
baseline, calibrate per-track thresholds on the benign calibration split, score the eval split,
and compute a :class:`MetricSet`. Across seeds we report every metric as a point estimate with a
95% bootstrap CI — the honest reporting the brief demands (§6). Held-out eval seeds are disjoint
from train/cal seeds, so nothing the threshold saw is scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.detect.report import SignatureProvider
from bellwether.eval.metrics import EpisodeOutcome, MetricSet, evaluate_episode, metric_set
from bellwether.fixtures import FAULT_INJECTORS, FaultSpec, FixtureAgent, faulted_run
from bellwether.stats import bootstrap_ci

# Per-seed run-id namespaces are kept far apart so seed spaces never collide.
_SEED_STRIDE = 1_000_000


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point:.3f}  [{self.lo:.3f}, {self.hi:.3f}]"


@dataclass
class BenchmarkConfig:
    fault_specs: list[FaultSpec] = field(
        default_factory=lambda: [
            FaultSpec(ft, severity=0.7, onset_step=1) for ft in sorted(FAULT_INJECTORS)
        ]
    )
    n_train: int = 120
    n_cal: int = 150
    n_eval_benign: int = 100
    n_eval_per_fault: int = 10
    target_fp_rate: float = 0.02
    n_seeds: int = 20
    min_samples: int = 30
    boot_seed: int = 12345
    scoring: ScoringConfig | None = None
    library: SignatureProvider | None = None  # skill-tier signatures consulted during scoring

    def scoring_config(self) -> ScoringConfig:
        if self.scoring is not None:
            return self.scoring
        return ScoringConfig(min_samples=self.min_samples, window_w=3, min_hits_k=2)


@dataclass
class BenchmarkReport:
    config: BenchmarkConfig
    metrics: dict[str, CI]
    per_fault_recall: dict[str, CI]
    per_seed: list[MetricSet]

    def render(self) -> str:
        lines = [
            "BELLWETHER drift-detection benchmark",
            f"  seeds={self.config.n_seeds}  train={self.config.n_train}  "
            f"cal={self.config.n_cal}  eval_benign={self.config.n_eval_benign}  "
            f"eval/fault={self.config.n_eval_per_fault}  target_fp={self.config.target_fp_rate}",
            "",
            "  metric (95% bootstrap CI across seeds)",
            "  " + "-" * 52,
        ]
        order = [
            "precision",
            "recall",
            "detection_rate",
            "f1",
            "fp_rate",
            "median_lead_time",
            "mean_lead_time",
        ]
        for name in order:
            ci = self.metrics[name]
            unit = " steps" if "lead_time" in name else ""
            lines.append(f"  {name:<18} {ci.point:6.3f}  [{ci.lo:6.3f}, {ci.hi:6.3f}]{unit}")
        lines += ["", "  per-fault timely recall", "  " + "-" * 52]
        for fault in sorted(self.per_fault_recall):
            ci = self.per_fault_recall[fault]
            lines.append(f"  {fault:<18} {ci.point:6.3f}  [{ci.lo:6.3f}, {ci.hi:6.3f}]")
        return "\n".join(lines)


def _ci(values: list[float], boot_seed: int) -> CI:
    point, lo, hi = bootstrap_ci(values, seed=boot_seed)
    return CI(point, lo, hi)


def _run_one_seed(
    agent: FixtureAgent, cfg: BenchmarkConfig, seed_index: int
) -> tuple[MetricSet, dict[str, list[EpisodeOutcome]]]:
    base = (seed_index + 1) * _SEED_STRIDE
    cursor = base

    train = range(cursor, cursor + cfg.n_train)
    cursor += cfg.n_train
    cal = range(cursor, cursor + cfg.n_cal)
    cursor += cfg.n_cal
    eval_benign = range(cursor, cursor + cfg.n_eval_benign)
    cursor += cfg.n_eval_benign

    mgr = BaselineManager()
    for s in train:
        mgr.learn(agent.clean_run(seed=s))
    baseline = mgr.baseline_for(agent.clean_run(seed=base))
    assert baseline is not None
    scorer = DriftScorer(cfg.scoring_config(), library=cfg.library)
    thr = calibrate_threshold(
        scorer, baseline, [agent.clean_run(seed=s) for s in cal], target_fp_rate=cfg.target_fp_rate
    )

    outcomes: list[EpisodeOutcome] = []
    per_fault: dict[str, list[EpisodeOutcome]] = {}

    for s in eval_benign:
        run = agent.clean_run(seed=s)
        outcomes.append(evaluate_episode(scorer.evaluate(run, baseline, thr), run))

    for spec in cfg.fault_specs:
        bucket = per_fault.setdefault(spec.fault_type, [])
        for _ in range(cfg.n_eval_per_fault):
            run = faulted_run(agent, cursor, spec)
            cursor += 1
            outcome = evaluate_episode(scorer.evaluate(run, baseline, thr), run)
            outcomes.append(outcome)
            bucket.append(outcome)

    return metric_set(outcomes), per_fault


def run_benchmark(
    agent: FixtureAgent | None = None, config: BenchmarkConfig | None = None
) -> BenchmarkReport:
    """Run the full N-seed benchmark and return a report with bootstrap CIs."""
    agent = agent or FixtureAgent()
    cfg = config or BenchmarkConfig()

    per_seed: list[MetricSet] = []
    per_fault_recall_samples: dict[str, list[float]] = {}

    for si in range(cfg.n_seeds):
        ms, per_fault = _run_one_seed(agent, cfg, si)
        per_seed.append(ms)
        for fault, bucket in per_fault.items():
            timely = sum(1 for o in bucket if o.timely)
            recall = timely / len(bucket) if bucket else 0.0
            per_fault_recall_samples.setdefault(fault, []).append(recall)

    def col(attr: str) -> list[float]:
        return [float(getattr(ms, attr)) for ms in per_seed]

    metrics = {
        name: _ci(col(name), cfg.boot_seed)
        for name in (
            "precision",
            "recall",
            "detection_rate",
            "f1",
            "fp_rate",
            "median_lead_time",
            "mean_lead_time",
        )
    }
    per_fault_recall = {
        fault: _ci(samples, cfg.boot_seed)
        for fault, samples in sorted(per_fault_recall_samples.items())
    }
    return BenchmarkReport(cfg, metrics, per_fault_recall, per_seed)
