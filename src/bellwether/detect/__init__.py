"""Drift detection: score feature observations against a baseline and raise calibrated alerts.

Pipeline (design docs 02/04):
- per-feature **detectors** turn each observation into calibrated sub-scores (conformal
  p-values), preserving *which* feature/family drifted (attribution);
- an **aggregator** fuses sub-scores into one per-observation drift score with a multiplicity
  correction (false-positive control);
- the **engine** runs the per-step + run-summary observations through the above, applies a
  sustained-alert debounce, and produces a :class:`DriftReport`. Thresholds are *calibrated* on
  a benign hold-out to hit a target false-positive rate, not hand-picked.
"""

from bellwether.detect.engine import DriftScorer, ScoringConfig, calibrate_threshold
from bellwether.detect.report import (
    Alert,
    DriftReport,
    ObservationScore,
    SubScore,
    Thresholds,
)

__all__ = [
    "DriftScorer",
    "ScoringConfig",
    "calibrate_threshold",
    "Alert",
    "DriftReport",
    "ObservationScore",
    "SubScore",
    "Thresholds",
]
