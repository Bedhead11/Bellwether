"""Tests for L2 replay-injection: inject faults into recorded runs (SDK / OTLP traces)."""

import pytest

from bellwether.eval.replay import inject_into_run
from bellwether.fixtures import FixtureAgent
from bellwether.ingest.otlp import parse_otlp_json
from bellwether.schema import SpanKind, SpanStatus


def _total_duration(run) -> float:  # type: ignore[no-untyped-def]
    return sum(s.duration_s for s in run.spans if s.parent_span_id is not None)


def _recorded_run():  # type: ignore[no-untyped-def]
    # Treat a fixture clean run as a "recorded" trace to replay into.
    return FixtureAgent().clean_run(seed=1)


def test_injection_labels_and_validates() -> None:
    run = inject_into_run(_recorded_run(), fault_type="latency_injection", severity=0.8)
    assert run.injected_fault is not None
    assert run.injected_fault.fault_type == "latency_injection"
    # Schema validity: timestamps re-flowed (children sequential, non-overlapping).
    kids = [s for s in run.spans if s.parent_span_id is not None]
    for prev, nxt in zip(kids, kids[1:], strict=False):
        assert nxt.start_time >= prev.end_time


def test_latency_injection_increases_duration() -> None:
    base = _recorded_run()
    faulted = inject_into_run(base, fault_type="latency_injection", severity=0.9, onset_step=0)
    assert _total_duration(faulted) > _total_duration(base)


def test_cost_blowup_increases_input_tokens() -> None:
    base = _recorded_run()
    faulted = inject_into_run(base, fault_type="cost_blowup", severity=0.9, onset_step=0)
    base_tok = sum(s.input_tokens or 0 for s in base.spans)
    new_tok = sum(s.input_tokens or 0 for s in faulted.spans)
    assert new_tok > base_tok


def test_induced_loop_adds_spans() -> None:
    base = _recorded_run()
    faulted = inject_into_run(base, fault_type="induced_loop", severity=0.8, onset_step=1)
    assert faulted.step_count > base.step_count


def test_retry_storm_adds_errors() -> None:
    faulted = inject_into_run(_recorded_run(), fault_type="retry_storm", severity=0.8, onset_step=1)
    assert any(s.status == SpanStatus.ERROR for s in faulted.spans)
    assert faulted.status.value == "error"


def test_unknown_fault_rejected() -> None:
    with pytest.raises(KeyError):
        inject_into_run(_recorded_run(), fault_type="nope", severity=0.5)


def test_replay_into_otel_normalized_run() -> None:
    """The realism path: inject a fault into a trace that came in via OTLP."""
    base_ns = 1_700_000_000_000_000_000
    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "t1",
                                "spanId": "root",
                                "name": "agent.run",
                                "startTimeUnixNano": str(base_ns),
                                "endTimeUnixNano": str(base_ns + 2_000_000_000),
                                "status": {"code": 1},
                            },
                            {
                                "traceId": "t1",
                                "spanId": "llm",
                                "parentSpanId": "root",
                                "name": "chat",
                                "startTimeUnixNano": str(base_ns),
                                "endTimeUnixNano": str(base_ns + 1_000_000_000),
                                "status": {"code": 1},
                                "attributes": [
                                    {"key": "gen_ai.request.model", "value": {"stringValue": "m"}},
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
                        ]
                    }
                ],
            }
        ]
    }
    recorded = parse_otlp_json(payload)[0]
    faulted = inject_into_run(recorded, fault_type="output_degradation", severity=0.9, onset_step=0)
    assert faulted.injected_fault is not None
    llm = next(s for s in faulted.spans if s.kind == SpanKind.LLM)
    assert llm.output_tokens is not None and llm.output_tokens < 120
