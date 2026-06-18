"""L1 + early-L3 tests for the fixture-agent harness and fault injectors.

These assert ground-truth properties of the *injected* faults (not detector behavior yet):
determinism, severity monotonicity on the perturbed feature, and that benign runs stay benign.
They are the floor the eval harness stands on — if the faucet is wrong, every metric is wrong.
"""

import pytest

from bellwether.fixtures import (
    FAULT_INJECTORS,
    FaultSpec,
    FixtureAgent,
    faulted_run,
    generate_dataset,
    inject_fault,
)
from bellwether.schema import SpanKind, SpanStatus


def _total_duration(run) -> float:  # type: ignore[no-untyped-def]
    return sum(s.duration_s for s in run.spans if s.parent_span_id is not None)


def _total_input_tokens(run) -> int:  # type: ignore[no-untyped-def]
    return sum(s.input_tokens or 0 for s in run.spans)


def _total_output_tokens(run) -> int:  # type: ignore[no-untyped-def]
    return sum(s.output_tokens or 0 for s in run.spans)


def test_clean_run_is_deterministic() -> None:
    agent = FixtureAgent()
    a = agent.clean_run(seed=42)
    b = agent.clean_run(seed=42)
    assert a.model_dump_json() == b.model_dump_json()


def test_clean_run_has_no_fault_and_is_ok() -> None:
    run = FixtureAgent().clean_run(seed=1)
    assert run.injected_fault is None
    assert run.status.value == "ok"
    assert run.step_count > 0


def test_materialized_timestamps_are_consistent() -> None:
    run = FixtureAgent().clean_run(seed=7)
    children = [s for s in run.spans if s.parent_span_id is not None]
    # Sequential, non-overlapping, monotonic.
    for prev, nxt in zip(children, children[1:], strict=False):
        assert nxt.start_time >= prev.end_time
    root = run.root_spans()[0]
    assert root.start_time == run.start_time
    assert root.end_time == run.end_time


def test_latency_fault_is_monotone_in_severity() -> None:
    agent = FixtureAgent()
    durations = []
    for sev in (0.0, 0.25, 0.5, 0.75, 1.0):
        run = faulted_run(agent, seed=5, spec=FaultSpec("latency_injection", sev, onset_step=0))
        durations.append(_total_duration(run))
    assert durations == sorted(durations)
    assert durations[-1] > durations[0]


def test_cost_fault_is_monotone_in_severity() -> None:
    agent = FixtureAgent()
    tokens = [
        _total_input_tokens(faulted_run(agent, 5, FaultSpec("cost_blowup", sev, onset_step=0)))
        for sev in (0.0, 0.5, 1.0)
    ]
    assert tokens[0] <= tokens[1] <= tokens[2]
    assert tokens[2] > tokens[0]


def test_output_degradation_shrinks_output() -> None:
    agent = FixtureAgent()
    clean = _total_output_tokens(agent.clean_run(seed=5))
    degraded = _total_output_tokens(
        faulted_run(agent, 5, FaultSpec("output_degradation", 0.9, onset_step=0))
    )
    assert degraded < clean


def test_induced_loop_increases_step_count() -> None:
    agent = FixtureAgent()
    clean = agent.clean_run(seed=5).step_count
    looped = faulted_run(agent, 5, FaultSpec("induced_loop", 0.8, onset_step=1)).step_count
    assert looped > clean


def test_retry_storm_adds_error_spans() -> None:
    run = faulted_run(FixtureAgent(), 5, FaultSpec("retry_storm", 0.8, onset_step=1))
    errors = [s for s in run.spans if s.status == SpanStatus.ERROR]
    assert len(errors) >= 1
    assert run.status.value == "error"


def test_tool_misselection_introduces_off_profile_tool() -> None:
    run = faulted_run(FixtureAgent(), 5, FaultSpec("tool_misselection", 1.0, onset_step=0))
    tool_names = {s.tool_name for s in run.spans if s.kind == SpanKind.TOOL}
    assert "deprecated_tool" in tool_names


def test_faulted_run_carries_ground_truth() -> None:
    run = faulted_run(FixtureAgent(), 5, FaultSpec("latency_injection", 0.7, onset_step=1))
    fault = run.injected_fault
    assert fault is not None
    assert fault.fault_type == "latency_injection"
    assert fault.severity == 0.7
    assert fault.visible_failure_step is not None
    assert fault.visible_failure_step >= fault.onset_step


def test_visible_failure_earlier_for_higher_severity() -> None:
    agent = FixtureAgent(n_steps_mean=8, n_steps_jitter=0)
    low = faulted_run(agent, 5, FaultSpec("latency_injection", 0.2, onset_step=1)).injected_fault
    high = faulted_run(agent, 5, FaultSpec("latency_injection", 0.9, onset_step=1)).injected_fault
    assert low is not None and high is not None
    assert high.visible_failure_step <= low.visible_failure_step


def test_generate_dataset_labels_and_counts() -> None:
    agent = FixtureAgent()
    runs = generate_dataset(agent, n_benign=10, n_per_fault=3)
    benign = [r for r in runs if r.injected_fault is None]
    faulted = [r for r in runs if r.injected_fault is not None]
    assert len(benign) == 10
    assert len(faulted) == 3 * len(FAULT_INJECTORS)
    # Run ids are unique (no collisions across the dataset).
    assert len({r.run_id for r in runs}) == len(runs)


def test_unknown_fault_type_rejected() -> None:
    with pytest.raises(KeyError):
        inject_fault(FixtureAgent().plan(1), fault_type="nonexistent", severity=0.5)


@pytest.mark.parametrize("fault_type", sorted(FAULT_INJECTORS))
def test_every_fault_materializes_valid_run(fault_type: str) -> None:
    """Every fault type must yield a schema-valid run (no orphan spans, consistent times)."""
    run = faulted_run(FixtureAgent(), 5, FaultSpec(fault_type, 0.6, onset_step=1))
    assert run.step_count > 0
    assert run.injected_fault is not None and run.injected_fault.fault_type == fault_type
