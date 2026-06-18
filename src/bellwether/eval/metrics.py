"""Metric definitions with explicit, defensible formulas (design doc 02).

An *episode* is one run with a known label. Drifted runs carry an ``InjectedFault`` whose
``visible_failure_step`` is the ground-truth ``t_fail`` (the step at which a human/SLA would
notice). The detector raises a *sustained* alert at ``t_alert``.

- **TP**  = drifted episode detected at or before ``t_fail`` (a *timely* detection).
- **FN**  = drifted episode not detected, or detected only after ``t_fail``.
- **FP**  = benign episode with any alert.
- **TN**  = benign episode with no alert.

We also report **detection_rate** (detected at all, even if late) so timeliness and raw
sensitivity are not conflated, and **lead_time** = ``t_fail - t_alert`` over TPs (in steps),
as a distribution (median + mean) because a few large values must not hide many near-zero ones.
A run-summary-only detection has no step ``t_alert``; we treat it as detected at the last step,
so it counts toward detection_rate but is timely only if the visible failure is at run end.
"""

from __future__ import annotations

from dataclasses import dataclass

from bellwether.detect.report import DriftReport
from bellwether.schema import AgentRun


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    is_drift: bool
    fault_type: str | None
    detected: bool
    timely: bool  # detected at or before the visible-failure step
    lead_time: float | None  # steps; defined only for timely true-positives


def _last_step_index(run: AgentRun) -> int:
    non_root = [s for s in run.spans if s.parent_span_id is not None]
    return max(len(non_root) - 1, 0)


def evaluate_episode(report: DriftReport, run: AgentRun) -> EpisodeOutcome:
    fault = run.injected_fault
    is_drift = fault is not None
    detected = report.alert.triggered

    if not is_drift:
        return EpisodeOutcome(False, None, detected, timely=False, lead_time=None)

    assert fault is not None
    t_fail = fault.visible_failure_step
    if t_fail is None:
        t_fail = _last_step_index(run)

    # Effective alert step: explicit step alert, else run-summary detection at run end.
    if report.alert.step_index is not None:
        t_alert = report.alert.step_index
    elif detected:
        t_alert = _last_step_index(run)
    else:
        t_alert = None

    timely = detected and t_alert is not None and t_alert <= t_fail
    lead_time = float(t_fail - t_alert) if (timely and t_alert is not None) else None
    return EpisodeOutcome(True, fault.fault_type, detected, timely, lead_time)


@dataclass(frozen=True, slots=True)
class MetricSet:
    """Point metrics for one evaluation pass (one seed)."""

    precision: float
    recall: float  # timely recall = TP / (TP + FN)
    detection_rate: float  # detected (even if late) / drifted
    f1: float
    fp_rate: float
    median_lead_time: float
    mean_lead_time: float
    tp: int
    fp: int
    fn: int
    tn: int
    n_drifted: int
    n_benign: int


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def metric_set(outcomes: list[EpisodeOutcome]) -> MetricSet:
    drifted = [o for o in outcomes if o.is_drift]
    benign = [o for o in outcomes if not o.is_drift]

    tp = sum(1 for o in drifted if o.timely)
    fn = len(drifted) - tp
    fp = sum(1 for o in benign if o.detected)
    tn = len(benign) - fp

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / len(drifted) if drifted else 0.0
    detection_rate = sum(1 for o in drifted if o.detected) / len(drifted) if drifted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fp_rate = fp / len(benign) if benign else 0.0

    lead_times = [o.lead_time for o in drifted if o.lead_time is not None]
    median_lt = _median(lead_times)
    mean_lt = sum(lead_times) / len(lead_times) if lead_times else 0.0

    return MetricSet(
        precision=precision,
        recall=recall,
        detection_rate=detection_rate,
        f1=f1,
        fp_rate=fp_rate,
        median_lead_time=median_lt,
        mean_lead_time=mean_lt,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n_drifted=len(drifted),
        n_benign=len(benign),
    )
