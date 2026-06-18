"""Aggregate per-feature sub-scores into one calibrated drift score with attribution.

The aggregator treats each sub-score's ``1 - score`` as a feature p-value and combines them via
the Šidák correction on the minimum p-value (design doc 02). This:

- rewards the single most-surprising feature (sensitivity), while
- discounting for the number of features tested (multiplicity → false-positive control), so a
  run with many mildly-noisy features does not alert just by chance.

Attribution falls out for free: the sub-scores are returned sorted by score, so the top one is
the drift driver and per-family sums explain the verdict.
"""

from __future__ import annotations

from bellwether.detect.report import ObservationScore, SubScore
from bellwether.features import FeatureFamily
from bellwether.stats import sidak_combine


def aggregate(
    kind: str,
    step_index: int | None,
    sub_scores: list[SubScore],
) -> ObservationScore:
    if not sub_scores:
        return ObservationScore(kind, step_index, 0.0, (), warmup=True)

    ordered = tuple(sorted(sub_scores, key=lambda s: s.score, reverse=True))
    p_values = [s.p_value for s in sub_scores]
    min_p = min(p_values)
    combined_p = sidak_combine(min_p, len(sub_scores))
    drift_score = 1.0 - combined_p
    return ObservationScore(kind, step_index, drift_score, ordered, warmup=False)


def family_contributions(scores: list[ObservationScore]) -> dict[FeatureFamily, float]:
    """Max sub-score per family across all observations — a compact 'who drifted' view."""
    out: dict[FeatureFamily, float] = {}
    for obs in scores:
        for s in obs.sub_scores:
            out[s.family] = max(out.get(s.family, 0.0), s.score)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
