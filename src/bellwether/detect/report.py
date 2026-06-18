"""Result types produced by the detection engine.

These are the product's user-facing output: not just "drift = 0.97" but *which signal drifted,
on which agent, when, and by how much* — attribution is the product (brief cross-cutting
principle).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bellwether.features import FeatureFamily


@dataclass(frozen=True, slots=True)
class SubScore:
    """One feature's contribution to drift at one observation."""

    feature: str
    context: str
    family: FeatureFamily
    score: float  # in [0, 1]; higher == more anomalous (== 1 - p_value)
    p_value: float
    direction: float  # signed robust z for numeric features; 0 for categorical
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ObservationScore:
    """Aggregated drift at a single observation (a step, or the run summary)."""

    kind: str
    step_index: int | None
    drift_score: float  # in [0, 1]
    sub_scores: tuple[SubScore, ...]  # sorted descending by score
    warmup: bool  # True when no feature had enough baseline data to score

    @property
    def primary(self) -> SubScore | None:
        return self.sub_scores[0] if self.sub_scores else None

    @property
    def primary_family(self) -> FeatureFamily | None:
        p = self.primary
        return p.family if p is not None else None


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Per-track alert thresholds, set by calibration.

    Three independent tracks, each given a share of the false-positive budget:
    - ``critical``: a single step this anomalous alerts immediately (fast path for overwhelming
      drift like a brand-new tool or a latency spike);
    - ``step``: the k-of-w sustained rule for moderate, persistent step drift;
    - ``run``: the run-summary track for purely aggregate drift.
    """

    critical: float
    step: float
    run: float


@dataclass(frozen=True, slots=True)
class Alert:
    """The run-level verdict."""

    triggered: bool
    step_index: int | None  # t_alert: first step of the sustained alert (None if run-summary only)
    drift_score: float
    primary_family: FeatureFamily | None
    primary_feature: str | None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Everything we concluded about one run."""

    run_id: str
    agent_id: str
    baseline_key: tuple[str, str, str]
    thresholds: Thresholds
    observation_scores: tuple[ObservationScore, ...]
    alert: Alert
    family_contributions: dict[FeatureFamily, float] = field(default_factory=dict)

    @property
    def max_drift(self) -> float:
        return max((o.drift_score for o in self.observation_scores), default=0.0)

    @property
    def step_scores(self) -> list[ObservationScore]:
        return [o for o in self.observation_scores if o.kind == "step"]

    def summary(self) -> str:
        a = self.alert
        if not a.triggered:
            return f"[ok] {self.run_id}: no drift (max={self.max_drift:.3f})"
        where = f"step {a.step_index}" if a.step_index is not None else "run-end"
        return (
            f"[DRIFT] {self.run_id}: {a.primary_family}/{a.primary_feature} at {where} "
            f"(score={a.drift_score:.3f})"
        )
