"""Tests for the eval harness: metric formulas (L0) + a small end-to-end benchmark (L1)."""

from datetime import UTC, datetime, timedelta

from bellwether.detect.report import Alert, DriftReport, Thresholds
from bellwether.eval import BenchmarkConfig, evaluate_episode, metric_set, run_benchmark
from bellwether.eval.metrics import EpisodeOutcome
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run
from bellwether.schema import AgentRun, Span, SpanKind

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _report(triggered: bool, step_index: int | None) -> DriftReport:
    return DriftReport(
        run_id="r",
        agent_id="a",
        baseline_key=("a", "default", "fp"),
        thresholds=Thresholds(0.9, 0.9, 0.9),
        observation_scores=(),
        alert=Alert(triggered, step_index, 0.99 if triggered else 0.0, None, None),
    )


def _run_with_steps(n: int, fault=None) -> AgentRun:  # type: ignore[no-untyped-def]
    spans = [
        Span(
            span_id=f"s{i}",
            parent_span_id="root",
            name=f"step{i}",
            kind=SpanKind.TOOL,
            start_time=T0 + timedelta(seconds=i),
            end_time=T0 + timedelta(seconds=i + 1),
        )
        for i in range(n)
    ]
    root = Span(
        span_id="root",
        name="root",
        kind=SpanKind.AGENT,
        start_time=T0,
        end_time=T0 + timedelta(seconds=n),
    )
    return AgentRun(
        run_id="r",
        agent_id="a",
        start_time=T0,
        end_time=T0 + timedelta(seconds=n),
        spans=[root, *spans],
        injected_fault=fault,
    )


# --- L0: metric formulas -----------------------------------------------------------------


def test_timely_true_positive_and_lead_time() -> None:
    run = faulted_run(
        FixtureAgent(n_steps_mean=10, n_steps_jitter=0),
        1,
        FaultSpec("latency_injection", 0.5, onset_step=2),
    )
    vf = run.injected_fault.visible_failure_step  # type: ignore[union-attr]
    outcome = evaluate_episode(_report(True, step_index=vf - 1), run)
    assert outcome.is_drift and outcome.detected and outcome.timely
    assert outcome.lead_time == 1.0


def test_late_detection_is_not_timely() -> None:
    run = faulted_run(
        FixtureAgent(n_steps_mean=10, n_steps_jitter=0),
        1,
        FaultSpec("latency_injection", 0.5, onset_step=2),
    )
    vf = run.injected_fault.visible_failure_step  # type: ignore[union-attr]
    outcome = evaluate_episode(_report(True, step_index=vf + 1), run)
    assert outcome.detected and not outcome.timely and outcome.lead_time is None


def test_benign_detection_is_false_positive() -> None:
    run = _run_with_steps(3, fault=None)
    outcome = evaluate_episode(_report(True, step_index=0), run)
    assert not outcome.is_drift and outcome.detected


def test_metric_set_confusion_and_rates() -> None:
    outcomes = [
        EpisodeOutcome(True, "f", True, True, 2.0),  # TP
        EpisodeOutcome(True, "f", True, False, None),  # detected late -> FN (not timely)
        EpisodeOutcome(True, "f", False, False, None),  # missed -> FN
        EpisodeOutcome(False, None, False, False, None),  # TN
        EpisodeOutcome(False, None, True, False, None),  # FP
    ]
    ms = metric_set(outcomes)
    assert ms.tp == 1 and ms.fn == 2 and ms.fp == 1 and ms.tn == 1
    assert ms.precision == 0.5  # 1 / (1 + 1)
    assert ms.recall == 1 / 3
    assert ms.detection_rate == 2 / 3  # 2 of 3 drifted detected (even if late)
    assert ms.fp_rate == 0.5
    assert ms.median_lead_time == 2.0


# --- L1: small end-to-end benchmark ------------------------------------------------------


def test_run_benchmark_smoke_and_quality() -> None:
    cfg = BenchmarkConfig(
        n_train=50, n_cal=30, n_eval_benign=30, n_eval_per_fault=5, n_seeds=3, min_samples=20
    )
    report = run_benchmark(config=cfg)

    # Sanity on shape.
    assert set(report.per_fault_recall) == {fs.fault_type for fs in cfg.fault_specs}
    assert len(report.per_seed) == 3

    # Quality bars (loose, to stay robust across machines): we should detect most faults while
    # keeping false positives within a small multiple of the target budget.
    assert report.metrics["detection_rate"].point >= 0.75
    assert report.metrics["fp_rate"].point <= 0.15
    assert report.metrics["recall"].point >= 0.5

    # The render is non-empty and mentions a fault.
    text = report.render()
    assert "precision" in text and "latency_injection" in text
