"""Signature generator: mine a candidate drift signature from labeled incidents.

This is the skill-tier *generator* of the self-improvement loop (brief §7). Given a set of
incident runs of one known drift type, it scores them and finds the ``(feature, context)`` pairs
that consistently show the largest anomaly *magnitude* at and after the fault onset, with a
consistent direction. Those become the signature — entirely from statistics, no LLM.

Crucially the generator mines from a *training* incident set; the candidate is then judged on a
disjoint held-out benchmark by the gate, so it cannot overfit the incidents it was mined from
(the anti-collapse discipline of design doc 01).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from bellwether.baseline import Baseline
from bellwether.detect import DriftScorer
from bellwether.improve.skills import DriftSignature
from bellwether.schema import AgentRun

# A candidate feature must clear this mean magnitude to be included (filters noise).
_MIN_MEAN_MAGNITUDE = 2.0


def mine_signature(
    name: str,
    incidents: Sequence[AgentRun],
    baseline: Baseline,
    scorer: DriftScorer,
    *,
    top_k: int = 2,
    description: str = "",
) -> DriftSignature | None:
    """Mine a :class:`DriftSignature` for ``name`` from incident runs of that drift type.

    Returns ``None`` if no feature is consistently anomalous enough to form a signature.
    """
    mags: dict[tuple[str, str], list[float]] = defaultdict(list)
    dirs: dict[tuple[str, str], list[float]] = defaultdict(list)

    for run in incidents:
        fault = run.injected_fault
        onset = fault.onset_step if fault is not None else 0
        for obs in scorer.score_run(run, baseline):
            if obs.kind != "step" or obs.step_index is None or obs.step_index < onset:
                continue
            for sub in obs.sub_scores:
                key = (sub.feature, sub.context)
                mags[key].append(sub.magnitude)
                dirs[key].append(sub.direction)

    if not mags:
        return None

    # Rank features by mean magnitude over the incident steps.
    ranked = sorted(
        mags.items(),
        key=lambda kv: sum(kv[1]) / len(kv[1]),
        reverse=True,
    )

    feature_directions: list[tuple[str, str, int]] = []
    for (feature, context), magnitude_list in ranked:
        mean_mag = sum(magnitude_list) / len(magnitude_list)
        if mean_mag < _MIN_MEAN_MAGNITUDE:
            continue
        mean_dir = sum(dirs[(feature, context)]) / len(dirs[(feature, context)])
        direction = 0 if abs(mean_dir) < 1e-9 else int(math.copysign(1, mean_dir))
        feature_directions.append((feature, context, direction))
        if len(feature_directions) >= top_k:
            break

    if not feature_directions:
        return None
    return DriftSignature(
        name=name,
        feature_directions=tuple(feature_directions),
        description=description or f"mined from {len(incidents)} {name} incidents",
    )
