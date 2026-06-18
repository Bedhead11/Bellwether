"""Tests for the detection engine: alert logic (L0) + train→calibrate→detect (L1) + metamorphic."""

import pytest

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.detect.report import ObservationScore, SubScore
from bellwether.features import FeatureFamily
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run

# Families a given fault may legitimately surface as the primary driver (correlated signals).
EXPECTED_FAMILIES = {
    "latency_injection": {FeatureFamily.TEMPORAL},
    "cost_blowup": {FeatureFamily.ECONOMIC, FeatureFamily.CONTEXT},
    "output_degradation": {FeatureFamily.SEMANTIC},
    "tool_misselection": {FeatureFamily.TOOL},
    "induced_loop": {FeatureFamily.STRUCTURAL, FeatureFamily.TOOL},
    "retry_storm": {FeatureFamily.STRUCTURAL},
}


def _obs(kind: str, idx, score: float, warmup: bool = False) -> ObservationScore:  # type: ignore[no-untyped-def]
    sub = (SubScore("step_latency", "llm", FeatureFamily.TEMPORAL, score, 1 - score, 2.0),)
    return ObservationScore(kind, idx, score, () if warmup else sub, warmup)


# --- L0: k-of-w sustained rule + level logic --------------------------------------------


def test_single_spike_does_not_alert() -> None:
    scorer = DriftScorer(ScoringConfig(window_w=3, min_hits_k=2))
    scores = [_obs("step", 0, 0.95), _obs("step", 1, 0.1), _obs("step", 2, 0.1)]
    assert scorer._step_sustained(scores, 0.9) == (False, None)


def test_alternating_hits_alert_when_rule_satisfied() -> None:
    scorer = DriftScorer(ScoringConfig(window_w=3, min_hits_k=2))
    # hits at steps 0 and 2 (a tool/cost fault pattern) within a window of 3; the rule is
    # satisfied at step 2 (the second hit), which is the honest, operational alert time.
    scores = [_obs("step", 0, 0.95), _obs("step", 1, 0.1), _obs("step", 2, 0.96)]
    assert scorer._step_sustained(scores, 0.9) == (True, 2)


def test_warmup_steps_are_not_hits() -> None:
    scorer = DriftScorer(ScoringConfig(window_w=3, min_hits_k=2))
    scores = [_obs("step", 0, 0.95), _obs("step", 1, 0.0, warmup=True), _obs("step", 2, 0.0)]
    assert scorer._step_sustained(scores, 0.9) == (False, None)


def test_step_alert_level_is_kth_largest_in_window() -> None:
    scorer = DriftScorer(ScoringConfig(window_w=3, min_hits_k=2))
    scores = [_obs("step", 0, 0.4), _obs("step", 1, 0.8), _obs("step", 2, 0.7)]
    # best window keeps the 2nd-largest: among {0.4,0.8,0.7} that is 0.7
    assert scorer.step_alert_level(scores) == pytest.approx(0.7)


# --- L1: train -> calibrate -> detect ----------------------------------------------------


def _train(agent: FixtureAgent, n: int = 80, min_samples: int = 20):  # type: ignore[no-untyped-def]
    mgr = BaselineManager()
    for s in range(n):
        mgr.learn(agent.clean_run(seed=s))
    baseline = mgr.baseline_for(agent.clean_run(seed=0))
    scorer = DriftScorer(ScoringConfig(min_samples=min_samples, window_w=3, min_hits_k=2))
    return mgr, baseline, scorer


def test_low_false_positive_on_benign() -> None:
    agent = FixtureAgent()
    _, baseline, scorer = _train(agent)
    cal = [agent.clean_run(seed=s) for s in range(80, 140)]
    thr = calibrate_threshold(scorer, baseline, cal, target_fp_rate=0.02)

    held_out = [agent.clean_run(seed=s) for s in range(200, 240)]
    fps = sum(scorer.evaluate(r, baseline, thr).alert.triggered for r in held_out)
    assert fps / len(held_out) <= 0.10  # comfortably bounded


def test_detects_faults_with_correct_attribution() -> None:
    agent = FixtureAgent()
    _, baseline, scorer = _train(agent)
    cal = [agent.clean_run(seed=s) for s in range(80, 140)]
    thr = calibrate_threshold(scorer, baseline, cal, target_fp_rate=0.02)

    for fault, families in EXPECTED_FAMILIES.items():
        triggered = 0
        correct_family = 0
        for seed in range(300, 308):
            run = faulted_run(agent, seed, FaultSpec(fault, severity=0.9, onset_step=1))
            report = scorer.evaluate(run, baseline, thr)
            if report.alert.triggered:
                triggered += 1
                if report.alert.primary_family in families:
                    correct_family += 1
        assert triggered >= 6, f"{fault}: only {triggered}/8 detected"
        assert correct_family >= 5, f"{fault}: attribution {correct_family}/8 into {families}"


def test_lead_time_nonnegative() -> None:
    agent = FixtureAgent(n_steps_mean=10, n_steps_jitter=0)
    _, baseline, scorer = _train(agent, n=80)
    cal = [agent.clean_run(seed=s) for s in range(80, 140)]
    thr = calibrate_threshold(scorer, baseline, cal, target_fp_rate=0.02)

    lead_times = []
    for seed in range(400, 412):
        run = faulted_run(agent, seed, FaultSpec("latency_injection", 0.6, onset_step=2))
        report = scorer.evaluate(run, baseline, thr)
        if report.alert.triggered and report.alert.step_index is not None:
            vf = run.injected_fault.visible_failure_step  # type: ignore[union-attr]
            lead_times.append(vf - report.alert.step_index)
    assert lead_times, "expected at least some step-level detections"
    assert sum(lt >= 0 for lt in lead_times) / len(lead_times) >= 0.7


@pytest.mark.parametrize("fault", ["latency_injection", "cost_blowup", "output_degradation"])
def test_severity_monotonicity(fault: str) -> None:
    """A worse fault must not produce a lower max drift score (metamorphic)."""
    agent = FixtureAgent()
    _, baseline, scorer = _train(agent)

    prev = -1.0
    for sev in (0.2, 0.5, 0.9):
        drifts = []
        for seed in range(500, 506):
            run = faulted_run(agent, seed, FaultSpec(fault, severity=sev, onset_step=1))
            drifts.append(scorer.evaluate(run, baseline).max_drift)
        avg = sum(drifts) / len(drifts)
        assert avg >= prev - 1e-6, f"{fault}: drift dropped at severity {sev}"
        prev = avg
