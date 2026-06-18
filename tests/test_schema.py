"""L0 unit tests for the canonical schema (deterministic, no statistics)."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from bellwether.schema import (
    SCHEMA_VERSION,
    AgentRun,
    ConfigFingerprint,
    InjectedFault,
    Span,
    SpanKind,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _span(span_id: str, parent: str | None, offset: float, dur: float, **kw: object) -> Span:
    return Span(
        span_id=span_id,
        parent_span_id=parent,
        name=kw.pop("name", span_id),  # type: ignore[arg-type]
        kind=kw.pop("kind", SpanKind.OTHER),  # type: ignore[arg-type]
        start_time=T0 + timedelta(seconds=offset),
        end_time=T0 + timedelta(seconds=offset + dur),
        **kw,  # type: ignore[arg-type]
    )


def _run(spans: list[Span], **kw: object) -> AgentRun:
    start = min(s.start_time for s in spans)
    end = max(s.end_time for s in spans)
    return AgentRun(
        run_id=kw.pop("run_id", "r1"),  # type: ignore[arg-type]
        agent_id=kw.pop("agent_id", "a1"),  # type: ignore[arg-type]
        start_time=start,
        end_time=end,
        spans=spans,
        **kw,  # type: ignore[arg-type]
    )


def test_fingerprint_is_stable_and_order_independent() -> None:
    a = ConfigFingerprint(model_id="m", temperature=0.2, provider="openai")
    b = ConfigFingerprint(provider="openai", model_id="m", temperature=0.2)
    assert a.hash() == b.hash()
    assert len(a.hash()) == 16


def test_fingerprint_changes_with_config() -> None:
    a = ConfigFingerprint(model_id="m", temperature=0.2)
    b = ConfigFingerprint(model_id="m", temperature=0.7)
    assert a.hash() != b.hash()


def test_baseline_key_combines_agent_task_fingerprint() -> None:
    run = _run([_span("s0", None, 0, 1)], agent_id="agent-x", task_class="search")
    agent, task, fp = run.baseline_key
    assert agent == "agent-x"
    assert task == "search"
    assert fp == run.config_fingerprint.hash()


def test_span_end_before_start_rejected() -> None:
    with pytest.raises(ValidationError):
        Span(
            span_id="s",
            name="s",
            start_time=T0 + timedelta(seconds=5),
            end_time=T0,
        )


def test_run_rejects_duplicate_span_ids() -> None:
    with pytest.raises(ValidationError):
        _run([_span("dup", None, 0, 1), _span("dup", None, 1, 1)])


def test_run_rejects_orphan_parent() -> None:
    with pytest.raises(ValidationError):
        _run([_span("child", "missing-parent", 0, 1)])


def test_tree_helpers_and_depth() -> None:
    root = _span("root", None, 0, 10, kind=SpanKind.AGENT)
    c1 = _span("c1", "root", 0, 2, kind=SpanKind.LLM)
    c2 = _span("c2", "root", 2, 2, kind=SpanKind.TOOL)
    gc = _span("gc", "c2", 2, 1)
    run = _run([root, c1, c2, gc])

    assert run.step_count == 4
    assert run.root_spans() == [root]
    assert {s.span_id for s in run.children_of("root")} == {"c1", "c2"}
    assert run.max_depth() == 3  # root -> c2 -> gc
    assert [s.span_id for s in run.spans_of_kind(SpanKind.LLM)] == ["c1"]


def test_schema_version_recorded() -> None:
    run = _run([_span("s0", None, 0, 1)])
    assert run.schema_version == SCHEMA_VERSION


def test_injected_fault_bounds() -> None:
    with pytest.raises(ValidationError):
        InjectedFault(fault_type="x", severity=1.5, onset_step=0)
    with pytest.raises(ValidationError):
        InjectedFault(fault_type="x", severity=0.5, onset_step=-1)
    ok = InjectedFault(fault_type="x", severity=0.5, onset_step=2, visible_failure_step=4)
    assert ok.visible_failure_step == 4


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        AgentRun(
            run_id="r",
            agent_id="a",
            start_time=T0,
            end_time=T0,
            unexpected_field="boom",  # type: ignore[call-arg]
        )
