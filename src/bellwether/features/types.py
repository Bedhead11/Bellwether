"""Feature observation types shared by extractors, the baseline, and detectors.

These are deliberately small, frozen value objects. A ``FeatureObservation`` bundles the
numeric and categorical features observed at one point in a run (a step, or the whole run).

The ``context`` field is the conditioning key (baseline doc 04): numeric step features are
conditioned on step kind (LLM-call latency and tool-call latency have different baselines), so
``context`` is usually the span kind. Kind-agnostic features use ``context="*"``; run-level
features use ``context="run"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Conditioning keys.
GLOBAL_CONTEXT = "*"
RUN_CONTEXT = "run"


class FeatureFamily(StrEnum):
    """The behavioral feature families from the brief (§3). Drives attribution."""

    STRUCTURAL = "structural"
    TOOL = "tool"
    TEMPORAL = "temporal"
    ECONOMIC = "economic"
    SEMANTIC = "semantic"
    CONTEXT = "context"


@dataclass(frozen=True, slots=True)
class NumericObs:
    """One numeric feature value, tagged with its family and conditioning context."""

    name: str
    context: str
    value: float
    family: FeatureFamily

    @property
    def key(self) -> tuple[str, str]:
        return (self.name, self.context)


@dataclass(frozen=True, slots=True)
class CategoricalObs:
    """One categorical feature value (e.g. which tool was called)."""

    name: str
    context: str
    category: str
    family: FeatureFamily

    @property
    def key(self) -> tuple[str, str]:
        return (self.name, self.context)


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    """All features observed at one point in a run.

    ``kind`` is ``"step"`` or ``"run_summary"``. ``step_index`` aligns with the span/plan index
    so it can be compared against an injected fault's ``onset_step`` / ``visible_failure_step``
    for lead-time. It is ``None`` for the run-summary observation.
    """

    kind: str
    step_index: int | None
    numerics: tuple[NumericObs, ...]
    categoricals: tuple[CategoricalObs, ...]
