"""Small, dependency-free statistical primitives used across BELLWETHER.

Kept deterministic and self-contained so every unit test is exact and CI never depends on a
heavyweight numerical stack. The choices implement design doc 04:

- **Robust** center/scale (median / MAD) because agent feature distributions are heavy-tailed
  and non-normal — mean/std would be dragged by outliers.
- **Conformal (distribution-free) p-values** for calibration rather than Gaussian tail
  assumptions.
- **Bootstrap confidence intervals** for the eval, because single-run agent metrics have large
  variance and must never be reported as point estimates (brief §6).

``river``/``alibi-detect`` detectors can later be added as *additional* ensemble members (the
topology tier); these primitives are the v1 core and the deterministic reference.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

# Scale factor making MAD a consistent estimator of the standard deviation for normal data.
_MAD_TO_SIGMA = 1.4826
# Floor on scale to avoid division by zero when a baseline window is (near-)constant.
_SCALE_FLOOR = 1e-9


def median(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("median of empty sequence")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def mad(xs: Sequence[float], center: float | None = None) -> float:
    """Median absolute deviation about the median (or a supplied center)."""
    if not xs:
        raise ValueError("mad of empty sequence")
    c = median(xs) if center is None else center
    return median([abs(x - c) for x in xs])


def robust_scale(xs: Sequence[float], center: float | None = None) -> float:
    """A robust standard-deviation estimate from MAD, with a fallback for degenerate windows.

    When MAD is zero (e.g. a constant or heavily tied window) we fall back to a scaled mean
    absolute deviation, then to a tiny floor, so deviation scoring stays finite and meaningful.
    """
    if not xs:
        raise ValueError("robust_scale of empty sequence")
    c = median(xs) if center is None else center
    m = mad(xs, c)
    if m > 0:
        return _MAD_TO_SIGMA * m
    mean_abs = sum(abs(x - c) for x in xs) / len(xs)
    if mean_abs > 0:
        return mean_abs
    return _SCALE_FLOOR


def robust_deviation(value: float, center: float, scale: float) -> float:
    """|value - center| / scale, guarding the scale floor. Always non-negative."""
    return abs(value - center) / max(scale, _SCALE_FLOOR)


def signed_robust_z(value: float, center: float, scale: float) -> float:
    return (value - center) / max(scale, _SCALE_FLOOR)


def conformal_pvalue(new_deviation: float, baseline_deviations: Sequence[float]) -> float:
    """Distribution-free anomaly p-value.

    p = (1 + #{baseline_dev >= new_dev}) / (n + 1). Small p == surprising. Under exchangeable
    benign data this p is (super-)uniform, which is what lets the aggregator's calibrated
    threshold map to a target false-positive rate.
    """
    n = len(baseline_deviations)
    if n == 0:
        return 1.0
    ge = sum(1 for d in baseline_deviations if d >= new_deviation)
    return (1 + ge) / (n + 1)


def sidak_combine(min_p: float, m: int) -> float:
    """Šidák multiplicity correction for the minimum of ``m`` p-values.

    Returns the corrected combined p-value (small == surprising). With more features there are
    more chances for one to look surprising by chance, so the correction discounts accordingly —
    this is a first-class false-positive control, not an afterthought.
    """
    if m <= 0:
        return 1.0
    min_p = min(max(min_p, 0.0), 1.0)
    return 1.0 - (1.0 - min_p) ** m


def percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile, ``q`` in [0, 1]."""
    if not xs:
        raise ValueError("percentile of empty sequence")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns ``(point, lower, upper)`` at level ``1 - alpha``.

    Used to report every eval metric with an interval rather than a single number (brief §6).
    """
    if not values:
        return (0.0, 0.0, 0.0)
    stat = statistic or (lambda xs: sum(xs) / len(xs))
    point = stat(values)
    if len(values) == 1:
        return (point, point, point)

    rng = random.Random(seed)
    n = len(values)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(stat(sample))
    boots.sort()
    lo = percentile(boots, alpha / 2)
    hi = percentile(boots, 1 - alpha / 2)
    return (point, lo, hi)
