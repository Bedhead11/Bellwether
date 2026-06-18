"""Governance gate + multi-objective metrics for the self-improvement loop.

The gate is the *same* conservative, statistical rule used for CI regression (design doc 02):
a candidate is promoted only if it delivers a genuine gain on the target metric while provably
**not** regressing the guard metrics — false-positive rate and precision — beyond tolerance.

This asymmetry (easy to ship a real improvement, hard to ship a regression) plus the
multi-objective guards is what makes every degenerate strategy lose:
- "alert on nothing" → recall does not improve → no promotion;
- "alert on everything" → FP-rate upper-CI blows the budget → rejected;
- a useless/junk signature → no target gain → rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

# A metric as (point, ci_low, ci_high).
Triple = tuple[float, float, float]


@dataclass(frozen=True)
class MetricVector:
    """The multi-objective evaluation of one variant, each metric with a bootstrap CI."""

    recall: Triple  # timely recall — the primary improvement target
    fp_rate: Triple  # guard: must stay within budget
    precision: Triple  # guard: must not regress
    detection_rate: Triple
    f1: Triple
    median_lead_time: Triple

    @staticmethod
    def point(t: Triple) -> float:
        return t[0]

    @staticmethod
    def lo(t: Triple) -> float:
        return t[1]

    @staticmethod
    def hi(t: Triple) -> float:
        return t[2]


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reasons: tuple[str, ...]
    metric_deltas: dict[str, float]


@dataclass(frozen=True)
class GovernanceGate:
    """Decides whether a candidate variant should be promoted over the current one."""

    fp_budget: float = 0.02
    fp_tol: float = 0.01  # FP upper-CI may exceed the budget by at most this
    precision_tol: float = 0.03  # precision lower-CI may fall this far below current point
    min_recall_gain: float = 0.02  # minimum target point-gain to count as an improvement

    def decide(self, current: MetricVector, candidate: MetricVector) -> GateDecision:
        reasons: list[str] = []

        recall_gain = candidate.recall[0] - current.recall[0]
        fp_ok = candidate.fp_rate[2] <= max(self.fp_budget, current.fp_rate[0]) + self.fp_tol
        precision_ok = candidate.precision[1] >= current.precision[0] - self.precision_tol
        improved = recall_gain >= self.min_recall_gain

        if not improved:
            reasons.append(
                f"no target gain: recall {current.recall[0]:.3f} -> {candidate.recall[0]:.3f} "
                f"(<{self.min_recall_gain:.3f})"
            )
        if not fp_ok:
            reasons.append(
                f"fp-rate regression: candidate upper-CI {candidate.fp_rate[2]:.3f} > budget "
                f"{self.fp_budget:.3f}+{self.fp_tol:.3f}"
            )
        if not precision_ok:
            reasons.append(
                f"precision regression: candidate lower-CI {candidate.precision[1]:.3f} < "
                f"{current.precision[0]:.3f}-{self.precision_tol:.3f}"
            )

        promote = improved and fp_ok and precision_ok
        if promote:
            reasons.append(f"promote: recall +{recall_gain:.3f}, fp within budget, precision held")
        deltas = {
            "recall": recall_gain,
            "fp_rate": candidate.fp_rate[0] - current.fp_rate[0],
            "precision": candidate.precision[0] - current.precision[0],
            "detection_rate": candidate.detection_rate[0] - current.detection_rate[0],
            "median_lead_time": candidate.median_lead_time[0] - current.median_lead_time[0],
        }
        return GateDecision(promote=promote, reasons=tuple(reasons), metric_deltas=deltas)
