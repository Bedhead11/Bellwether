"""Feature extraction: decompose an ``AgentRun`` into observations a baseline can learn.

A run becomes an ordered sequence of :class:`FeatureObservation` — one per step plus a final
run-summary. This unifies two detection cadences behind one machinery (baseline doc 04):

- **per-step** observations drive *streaming* detection, which is what makes **lead-time**
  measurable (we can alert at the step the drift starts, before the visible failure);
- the **run-summary** observation carries per-run aggregates (step count, depth, totals) that
  only make sense at run end.

Extractors are pure, deterministic functions (TDD floor, brief §11): same run in, same
observations out, no statistics and no randomness.
"""

from bellwether.features.extract import (
    coordination_features,
    extract_observations,
    run_summary_observation,
)
from bellwether.features.types import (
    CategoricalObs,
    FeatureFamily,
    FeatureObservation,
    NumericObs,
)

__all__ = [
    "FeatureFamily",
    "NumericObs",
    "CategoricalObs",
    "FeatureObservation",
    "extract_observations",
    "run_summary_observation",
    "coordination_features",
]
