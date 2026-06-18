"""L0 unit tests for the OTel span -> AgentRun normalizer."""

import pytest

from bellwether.ingest.otel import agentrun_from_otel_spans
from bellwether.schema import SpanKind


def _otel_spans() -> list[dict[str, object]]:
    # Lightweight OTel-shaped spans using GenAI semantic-convention attributes.
    base_ns = 1_700_000_000_000_000_000
    return [
        {
            "span_id": "root",
            "parent_span_id": None,
            "name": "agent.run",
            "start_time": base_ns,
            "end_time": base_ns + 3_000_000_000,
            "status": "ok",
            "attributes": {},
        },
        {
            "span_id": "llm-1",
            "parent_span_id": "root",
            "name": "chat",
            "start_time": base_ns,
            "end_time": base_ns + 1_000_000_000,
            "status": "ok",
            "attributes": {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.request.temperature": 0.2,
                "gen_ai.usage.input_tokens": 400,
                "gen_ai.usage.output_tokens": 120,
            },
        },
        {
            "span_id": "tool-1",
            "parent_span_id": "root",
            "name": "search",
            "start_time": base_ns + 1_000_000_000,
            "end_time": base_ns + 1_500_000_000,
            "status": "ok",
            "attributes": {"gen_ai.tool.name": "search"},
            "tool_args": {"q": "weather"},
        },
    ]


def test_normalizes_kinds_and_tokens() -> None:
    run = agentrun_from_otel_spans(_otel_spans(), run_id="r1", agent_id="a1", task_class="qa")
    assert run.run_id == "r1"
    assert run.task_class == "qa"
    assert run.step_count == 3

    kinds = {s.span_id: s.kind for s in run.spans}
    assert kinds["llm-1"] == SpanKind.LLM
    assert kinds["tool-1"] == SpanKind.TOOL
    assert kinds["root"] == SpanKind.AGENT

    llm = next(s for s in run.spans if s.span_id == "llm-1")
    assert llm.input_tokens == 400
    assert llm.output_tokens == 120
    assert llm.model == "gpt-4o-mini"


def test_fingerprint_inferred_from_llm_span() -> None:
    run = agentrun_from_otel_spans(
        _otel_spans(), run_id="r1", agent_id="a1", config_extra={"prompt_version": "v3"}
    )
    fp = run.config_fingerprint
    assert fp.model_id == "gpt-4o-mini"
    assert fp.provider == "openai"
    assert fp.temperature == 0.2
    assert fp.prompt_version == "v3"


def test_run_time_bounds_from_spans() -> None:
    run = agentrun_from_otel_spans(_otel_spans(), run_id="r1", agent_id="a1")
    assert run.start_time == min(s.start_time for s in run.spans)
    assert run.end_time == max(s.end_time for s in run.spans)


def test_empty_spans_rejected() -> None:
    with pytest.raises(ValueError):
        agentrun_from_otel_spans([], run_id="r", agent_id="a")


def test_iso_and_datetime_timestamps_accepted() -> None:
    spans = [
        {
            "span_id": "s0",
            "name": "step",
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-01T00:00:01+00:00",
            "status": "ok",
            "attributes": {},
        }
    ]
    run = agentrun_from_otel_spans(spans, run_id="r", agent_id="a")
    assert run.duration_s == 1.0
