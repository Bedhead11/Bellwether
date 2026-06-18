"""L0 unit tests for feature extraction (pure, deterministic)."""

from bellwether.features import (
    FeatureFamily,
    extract_observations,
    run_summary_observation,
)
from bellwether.features.extract import step_observations
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run


def _numeric(obs, name):  # type: ignore[no-untyped-def]
    return {n.name: n for n in obs.numerics}.get(name)


def _run_numeric(run, name):  # type: ignore[no-untyped-def]
    rs = run_summary_observation(run)
    return _numeric(rs, name)


def test_observation_sequence_shape() -> None:
    run = FixtureAgent().clean_run(seed=1)
    obs = extract_observations(run)
    assert obs[-1].kind == "run_summary"
    steps = [o for o in obs if o.kind == "step"]
    non_root_spans = [s for s in run.spans if s.parent_span_id is not None]
    assert len(steps) == len(non_root_spans)
    # step_index aligns with order
    assert [o.step_index for o in steps] == list(range(len(steps)))


def test_step_features_present_and_deterministic() -> None:
    run = FixtureAgent().clean_run(seed=2)
    a = step_observations(run)
    b = step_observations(run)
    assert a == b
    # First step is an LLM call -> has token/cost features
    first = a[0]
    assert _numeric(first, "step_latency") is not None
    assert _numeric(first, "step_input_tokens") is not None
    assert _numeric(first, "step_cost") is not None


def test_benign_run_has_zero_repeat_and_errors() -> None:
    run = FixtureAgent().clean_run(seed=3)
    for o in step_observations(run):
        rc = _numeric(o, "repeat_count")
        if rc is not None:
            assert rc.value == 0.0
        err = _numeric(o, "is_error")
        assert err is not None and err.value == 0.0


def test_loop_fault_raises_repeat_count() -> None:
    run = faulted_run(FixtureAgent(), 5, FaultSpec("induced_loop", 0.8, onset_step=1))
    repeats = [
        n.value
        for o in step_observations(run)
        for n in o.numerics
        if n.name == "repeat_count"
    ]
    assert max(repeats) >= 1.0  # the loop repeats an identical (tool, args) call


def test_retry_fault_sets_is_error() -> None:
    run = faulted_run(FixtureAgent(), 5, FaultSpec("retry_storm", 0.8, onset_step=1))
    errors = [
        n.value for o in step_observations(run) for n in o.numerics if n.name == "is_error"
    ]
    assert sum(errors) >= 1.0


def test_misselection_fault_introduces_novel_tool_category() -> None:
    run = faulted_run(FixtureAgent(), 5, FaultSpec("tool_misselection", 1.0, onset_step=0))
    cats = {c.category for o in step_observations(run) for c in o.categoricals}
    assert "deprecated_tool" in cats


def test_run_summary_families_covered() -> None:
    run = FixtureAgent().clean_run(seed=4)
    fams = {n.family for n in run_summary_observation(run).numerics}
    assert fams == set(FeatureFamily)  # every family represented at run level


def test_latency_fault_raises_total_duration() -> None:
    agent = FixtureAgent()
    clean = _run_numeric(agent.clean_run(seed=5), "total_duration_s")
    faulted = _run_numeric(
        faulted_run(agent, 5, FaultSpec("latency_injection", 0.9, onset_step=0)),
        "total_duration_s",
    )
    assert clean is not None and faulted is not None
    assert faulted.value > clean.value


def test_output_degradation_lowers_min_output_tokens() -> None:
    agent = FixtureAgent()
    clean = _run_numeric(agent.clean_run(seed=5), "min_output_tokens")
    degraded = _run_numeric(
        faulted_run(agent, 5, FaultSpec("output_degradation", 0.9, onset_step=0)),
        "min_output_tokens",
    )
    assert clean is not None and degraded is not None
    assert degraded.value < clean.value


def test_loop_fault_raises_loop_score() -> None:
    agent = FixtureAgent()
    clean = _run_numeric(agent.clean_run(seed=5), "loop_score")
    looped = _run_numeric(
        faulted_run(agent, 5, FaultSpec("induced_loop", 0.8, onset_step=1)), "loop_score"
    )
    assert clean is not None and looped is not None
    assert clean.value == 0.0
    assert looped.value >= 1.0
