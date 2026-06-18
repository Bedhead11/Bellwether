"""Tests for multi-agent coordination drift detection (the named unsolved problem)."""

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.features import FeatureFamily, coordination_features, run_summary_observation
from bellwether.fixtures.multiagent import (
    COORDINATION_FAULTS,
    CoordinationFaultSpec,
    MultiAgentSystem,
    coordination_faulted_run,
)
from bellwether.improve import DriftSignature, SkillLibrary


def _coord(run):  # type: ignore[no-untyped-def]
    return {n.name: n.value for n in coordination_features(run)}


# --- coordination feature extraction -----------------------------------------------------


def test_single_agent_run_has_no_coordination_features() -> None:
    from bellwether.fixtures import FixtureAgent

    assert coordination_features(FixtureAgent().clean_run(seed=1)) == []


def test_multiagent_run_has_coordination_features() -> None:
    run = MultiAgentSystem().clean_run(seed=1)
    feats = _coord(run)
    assert feats["n_agents"] >= 2
    assert "handoff_rate" in feats and "pingpong_rate" in feats
    fams = {n.family for n in run_summary_observation(run).numerics}
    assert FeatureFamily.COORDINATION in fams


def test_ping_pong_raises_pingpong_rate() -> None:
    sys = MultiAgentSystem()
    clean = _coord(sys.clean_run(seed=5))
    faulted = _coord(coordination_faulted_run(sys, 5, CoordinationFaultSpec("ping_pong", 0.9)))
    assert faulted["pingpong_rate"] > clean["pingpong_rate"]


def test_role_collapse_raises_imbalance() -> None:
    sys = MultiAgentSystem()
    clean = _coord(sys.clean_run(seed=5))
    faulted = _coord(coordination_faulted_run(sys, 5, CoordinationFaultSpec("role_collapse", 0.9)))
    assert faulted["role_imbalance"] > clean["role_imbalance"]
    assert faulted["role_entropy"] < clean["role_entropy"]


def test_handoff_storm_raises_handoff_rate() -> None:
    sys = MultiAgentSystem()
    clean = _coord(sys.clean_run(seed=5))
    faulted = _coord(coordination_faulted_run(sys, 5, CoordinationFaultSpec("handoff_storm", 0.9)))
    assert faulted["handoff_rate"] > clean["handoff_rate"]


# --- end-to-end: coordination drift detected via coordination signatures -----------------


def coordination_library() -> SkillLibrary:
    """Skill-tier signatures for the coordination faults — the architecture generalizing to the
    named unsolved problem. Each keys on its *unique* discriminator and they are ordered
    specific-first so the right signature wins when patterns overlap (a ping-pong run also raises
    the handoff rate, but only ping-pong raises pingpong_rate). Magnitude-based focused scores
    avoid the conformal floor where an extreme coordination value would collide with a benign one.
    """
    return SkillLibrary(
        [
            DriftSignature("ping_pong", (("pingpong_rate", "run", 1),)),
            DriftSignature("role_collapse", (("role_imbalance", "run", 1),)),
            DriftSignature("handoff_storm", (("handoff_rate", "run", 1),)),
        ]
    )


def test_coordination_drift_detected_via_signatures() -> None:
    sys = MultiAgentSystem()
    mgr = BaselineManager()
    for s in range(150):
        mgr.learn(sys.clean_run(seed=s))
    baseline = mgr.baseline_for(sys.clean_run(seed=0))
    assert baseline is not None
    scorer = DriftScorer(ScoringConfig(min_samples=30), library=coordination_library())
    thr = calibrate_threshold(
        scorer, baseline, [sys.clean_run(seed=s) for s in range(150, 320)], target_fp_rate=0.02
    )

    # Benign multi-agent runs rarely alert.
    fps = sum(
        scorer.evaluate(sys.clean_run(seed=s), baseline, thr).alert.triggered
        for s in range(400, 440)
    )
    assert fps / 40 <= 0.12

    # Coordination faults are detected and attributed to the right coordination signature —
    # even though each agent's per-step behavior (latency, tokens) looks locally normal.
    for fault in COORDINATION_FAULTS:
        detected = 0
        right_sig = 0
        for seed in range(500, 510):
            run = coordination_faulted_run(sys, seed, CoordinationFaultSpec(fault, 0.9))
            report = scorer.evaluate(run, baseline, thr)
            if report.alert.triggered:
                detected += 1
                if report.alert.signature == fault:
                    right_sig += 1
        assert detected >= 7, f"{fault}: only {detected}/10 detected"
        assert right_sig >= 5, f"{fault}: attributed to its signature {right_sig}/10"
