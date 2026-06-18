"""L0 unit tests for the statistical primitives."""

import math

from bellwether.stats import (
    bootstrap_ci,
    conformal_pvalue,
    mad,
    median,
    percentile,
    robust_deviation,
    robust_scale,
    sidak_combine,
    signed_robust_z,
)


def test_median_odd_even() -> None:
    assert median([3, 1, 2]) == 2
    assert median([1, 2, 3, 4]) == 2.5


def test_mad_and_scale() -> None:
    xs = [1, 1, 1, 1, 100]
    assert mad(xs) == 0  # majority identical -> MAD zero
    # robust_scale falls back to mean-abs-dev (nonzero) instead of zero
    assert robust_scale(xs) > 0


def test_robust_scale_normalish() -> None:
    xs = [10, 11, 9, 10, 12, 8, 10]
    assert robust_scale(xs) > 0
    # constant window -> floor, finite
    assert robust_scale([5, 5, 5]) > 0


def test_robust_deviation_and_z() -> None:
    center, scale = 10.0, 2.0
    assert robust_deviation(14.0, center, scale) == 2.0
    assert signed_robust_z(14.0, center, scale) == 2.0
    assert signed_robust_z(6.0, center, scale) == -2.0


def test_conformal_pvalue_bounds_and_monotonicity() -> None:
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5]
    # A huge deviation is maximally surprising -> smallest p = 1/(n+1) (floored).
    assert conformal_pvalue(100.0, baseline) == 1 / 6
    # A tiny deviation is unsurprising -> p close to 1
    assert conformal_pvalue(0.0, baseline) == 1.0
    # Monotone: larger deviation -> smaller (or equal) p
    assert conformal_pvalue(0.45, baseline) <= conformal_pvalue(0.15, baseline)
    # empty baseline -> uninformative
    assert conformal_pvalue(1.0, []) == 1.0


def test_sidak_combine() -> None:
    # With one feature, no correction.
    assert math.isclose(sidak_combine(0.05, 1), 0.05)
    # More features -> larger combined p (more conservative) for same min_p.
    assert sidak_combine(0.05, 10) > sidak_combine(0.05, 1)
    # Bounds.
    assert sidak_combine(0.0, 5) == 0.0
    assert math.isclose(sidak_combine(1.0, 5), 1.0)


def test_percentile() -> None:
    xs = [0, 1, 2, 3, 4]
    assert percentile(xs, 0.0) == 0
    assert percentile(xs, 1.0) == 4
    assert percentile(xs, 0.5) == 2


def test_bootstrap_ci_contains_point_and_is_ordered() -> None:
    values = [float(x) for x in range(100)]
    point, lo, hi = bootstrap_ci(values, seed=1)
    assert lo <= point <= hi
    assert math.isclose(point, 49.5)
    # deterministic given seed
    assert bootstrap_ci(values, seed=1) == bootstrap_ci(values, seed=1)


def test_bootstrap_ci_degenerate() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([5.0]) == (5.0, 5.0, 5.0)
