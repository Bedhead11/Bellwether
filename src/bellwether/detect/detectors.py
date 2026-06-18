"""Per-feature detectors: turn one feature observation into a calibrated sub-score.

Each detector is small and separable so the ensemble can be reweighted/added/removed per
(agent, task_class) by the topology tier later (design doc 01). v1 ships two:

- :func:`score_numeric` — robust conformal anomaly score for a numeric feature.
- :func:`score_categorical` — frequency-novelty score for a categorical feature.

Both return ``None`` when the baseline is still in *warmup* (too few samples), which is how
cold-start avoids crying wolf (design doc 04).
"""

from __future__ import annotations

from bellwether.baseline import Baseline
from bellwether.detect.report import SubScore
from bellwether.features import CategoricalObs, NumericObs
from bellwether.stats import (
    conformal_pvalue,
    median,
    robust_deviation,
    robust_scale,
    signed_robust_z,
)


def score_numeric(obs: NumericObs, baseline: Baseline, *, min_samples: int) -> SubScore | None:
    """Robust, distribution-free conformal score for one numeric feature.

    The new value's robust deviation is compared against the deviations of the baseline window;
    the conformal p-value of that deviation becomes ``1 - score``.
    """
    window = baseline.numeric_window(obs.name, obs.context)
    if len(window) < min_samples:
        return None  # warmup

    center = median(window)
    scale = robust_scale(window, center)
    new_dev = robust_deviation(obs.value, center, scale)
    baseline_devs = [robust_deviation(w, center, scale) for w in window]
    p = conformal_pvalue(new_dev, baseline_devs)
    z = signed_robust_z(obs.value, center, scale)
    return SubScore(
        feature=obs.name,
        context=obs.context,
        family=obs.family,
        score=1.0 - p,
        p_value=p,
        direction=z,
        detail=f"z={z:+.2f}",
    )


def score_categorical(
    obs: CategoricalObs, baseline: Baseline, *, min_samples: int, laplace_alpha: float = 1.0
) -> SubScore | None:
    """Frequency-novelty score for a categorical feature (e.g. tool identity).

    A Laplace-smoothed category probability acts as the pseudo-p-value: rare/novel categories
    are surprising (high score), common ones are not. ``+1`` reserves smoothing mass for
    never-before-seen categories, so a brand-new tool scores high.
    """
    counts, total = baseline.categorical_counts(obs.name, obs.context)
    if total < min_samples:
        return None  # warmup

    c = counts.get(obs.category, 0)
    k = len(counts)
    p = (c + laplace_alpha) / (total + laplace_alpha * (k + 1))
    p = min(p, 1.0)
    novel = c == 0
    return SubScore(
        feature=obs.name,
        context=obs.context,
        family=obs.family,
        score=1.0 - p,
        p_value=p,
        direction=0.0,
        detail="novel-category" if novel else f"freq={c}/{total}",
    )
