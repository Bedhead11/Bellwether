"""L3 metamorphic tests (property-based, via Hypothesis) — eval design doc 02.

Metamorphic properties test the detector's *logic*, not a single point: relationships that must
hold for any input. The baseline/scorer are built once at module load (expensive) and shared
across generated examples.
"""

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.features import FeatureFamily
from bellwether.fixtures import FAULT_INJECTORS, FaultSpec, FixtureAgent, faulted_run

_AGENT = FixtureAgent()
_MGR = BaselineManager()
for _s in range(120):
    _MGR.learn(_AGENT.clean_run(seed=_s))
_BASELINE = _MGR.baseline_for(_AGENT.clean_run(seed=0))
assert _BASELINE is not None
_SCORER = DriftScorer(ScoringConfig(min_samples=30))
_THR = calibrate_threshold(
    _SCORER, _BASELINE, [_AGENT.clean_run(seed=s) for s in range(120, 270)], target_fp_rate=0.02
)

_EXPECTED_FAMILIES = {
    "latency_injection": {FeatureFamily.TEMPORAL},
    "cost_blowup": {FeatureFamily.ECONOMIC, FeatureFamily.CONTEXT},
    "output_degradation": {FeatureFamily.SEMANTIC},
    "tool_misselection": {FeatureFamily.TOOL},
    "induced_loop": {FeatureFamily.STRUCTURAL, FeatureFamily.TOOL},
    "retry_storm": {FeatureFamily.STRUCTURAL},
}

_SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _drift(fault: str, severity: float, seed: int) -> float:
    run = faulted_run(_AGENT, seed, FaultSpec(fault, severity, onset_step=1))
    return _SCORER.evaluate(run, _BASELINE, _THR).max_drift


@_SETTINGS
@given(
    fault=st.sampled_from(sorted(FAULT_INJECTORS)),
    seed=st.integers(min_value=0, max_value=5000),
    low=st.floats(min_value=0.1, max_value=0.4),
    high=st.floats(min_value=0.7, max_value=1.0),
)
def test_severity_monotonicity(fault: str, seed: int, low: float, high: float) -> None:
    """A strictly worse fault must not produce a lower drift score (within tolerance)."""
    assume(high - low >= 0.3)
    d_low = _drift(fault, low, seed)
    d_high = _drift(fault, high, seed)
    assert d_high >= d_low - 0.05, f"{fault}: drift dropped from sev {low:.2f} to {high:.2f}"


@_SETTINGS
@given(
    fault=st.sampled_from(sorted(_EXPECTED_FAMILIES)),
    seed=st.integers(min_value=0, max_value=5000),
)
def test_attribution_matches_injected_family(fault: str, seed: int) -> None:
    """When a severe single fault is detected, attribution points at its feature family."""
    run = faulted_run(_AGENT, seed, FaultSpec(fault, severity=0.95, onset_step=1))
    report = _SCORER.evaluate(run, _BASELINE, _THR)
    if report.alert.triggered and report.alert.primary_family is not None:
        assert report.alert.primary_family in _EXPECTED_FAMILIES[fault]


@_SETTINGS
@given(seed=st.integers(min_value=10_000, max_value=30_000))
def test_benign_runs_have_bounded_drift(seed: int) -> None:
    """A benign re-run scores below the maximum — it never looks maximally drifted."""
    report = _SCORER.evaluate(_AGENT.clean_run(seed=seed), _BASELINE, _THR)
    assert report.max_drift < 0.999


@_SETTINGS
@given(
    fault=st.sampled_from(sorted(FAULT_INJECTORS)),
    seed=st.integers(min_value=0, max_value=5000),
)
def test_drift_score_in_unit_interval(fault: str, seed: int) -> None:
    """Drift scores are always valid probabilities in [0, 1]."""
    run = faulted_run(_AGENT, seed, FaultSpec(fault, severity=0.6, onset_step=1))
    for obs in _SCORER.evaluate(run, _BASELINE, _THR).observation_scores:
        assert 0.0 <= obs.drift_score <= 1.0
