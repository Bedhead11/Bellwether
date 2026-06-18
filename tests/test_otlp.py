"""Tests for OTLP/JSON ingest (zero-code OpenTelemetry path)."""

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer
from bellwether.ingest.otlp import flatten_attributes, parse_otlp_json
from bellwether.monitor import DriftMonitor
from bellwether.schema import SpanKind


def _otlp_payload(trace_id: str = "trace-1") -> dict:
    base_ns = 1_700_000_000_000_000_000
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "billing-agent"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "my-instrumentation"},
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": "root",
                                "name": "agent.run",
                                "startTimeUnixNano": str(base_ns),
                                "endTimeUnixNano": str(base_ns + 3_000_000_000),
                                "status": {"code": 1},
                                "attributes": [
                                    {"key": "gen_ai.task.class", "value": {"stringValue": "qa"}}
                                ],
                            },
                            {
                                "traceId": trace_id,
                                "spanId": "llm-1",
                                "parentSpanId": "root",
                                "name": "chat",
                                "startTimeUnixNano": str(base_ns),
                                "endTimeUnixNano": str(base_ns + 1_000_000_000),
                                "status": {"code": 1},
                                "attributes": [
                                    {
                                        "key": "gen_ai.request.model",
                                        "value": {"stringValue": "gpt-4o-mini"},
                                    },
                                    {
                                        "key": "gen_ai.usage.input_tokens",
                                        "value": {"intValue": "400"},
                                    },
                                    {
                                        "key": "gen_ai.usage.output_tokens",
                                        "value": {"intValue": "120"},
                                    },
                                ],
                            },
                            {
                                "traceId": trace_id,
                                "spanId": "tool-1",
                                "parentSpanId": "root",
                                "name": "search",
                                "startTimeUnixNano": str(base_ns + 1_000_000_000),
                                "endTimeUnixNano": str(base_ns + 1_500_000_000),
                                "status": {"code": 1},
                                "attributes": [
                                    {"key": "gen_ai.tool.name", "value": {"stringValue": "search"}}
                                ],
                            },
                        ],
                    }
                ],
            }
        ]
    }


def test_flatten_attributes_decodes_value_types() -> None:
    attrs = [
        {"key": "s", "value": {"stringValue": "x"}},
        {"key": "i", "value": {"intValue": "42"}},
        {"key": "d", "value": {"doubleValue": 1.5}},
        {"key": "b", "value": {"boolValue": True}},
    ]
    flat = flatten_attributes(attrs)
    assert flat == {"s": "x", "i": 42, "d": 1.5, "b": True}


def test_parse_otlp_groups_by_trace_and_normalizes() -> None:
    runs = parse_otlp_json(_otlp_payload())
    assert len(runs) == 1
    run = runs[0]
    assert run.run_id == "trace-1"
    assert run.agent_id == "billing-agent"  # from resource service.name
    assert run.task_class == "qa"  # from gen_ai.task.class
    assert run.step_count == 3
    kinds = {s.span_id: s.kind for s in run.spans}
    assert kinds["llm-1"] == SpanKind.LLM
    assert kinds["tool-1"] == SpanKind.TOOL
    llm = next(s for s in run.spans if s.span_id == "llm-1")
    assert llm.input_tokens == 400 and llm.model == "gpt-4o-mini"


def test_two_traces_become_two_runs() -> None:
    payload = _otlp_payload("trace-a")
    payload["resourceSpans"].extend(_otlp_payload("trace-b")["resourceSpans"])
    runs = parse_otlp_json(payload)
    assert {r.run_id for r in runs} == {"trace-a", "trace-b"}


def test_explicit_agent_id_overrides_attributes() -> None:
    runs = parse_otlp_json(_otlp_payload(), agent_id="override", task_class="t")
    assert runs[0].agent_id == "override" and runs[0].task_class == "t"


def test_monitor_ingest_otlp_end_to_end() -> None:
    # A monitor with a learned baseline scores an OTLP trace with zero agent-side code.
    mgr = BaselineManager()
    reports = DriftMonitor(manager=mgr, scorer=DriftScorer()).ingest_otlp(_otlp_payload())
    # No baseline yet for this fingerprint -> None, but parsing + routing worked.
    assert reports == [None]
