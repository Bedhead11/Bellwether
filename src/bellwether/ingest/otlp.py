"""Parse OTLP/JSON trace payloads into canonical ``AgentRun``s — zero-code ingest.

This is the streaming-receiver side of the "consume OTel, don't reinvent it" stance (brief §5):
an agent already exporting OpenTelemetry traces can point its OTLP/HTTP exporter at BELLWETHER and
get drift detection with **no code changes**. We parse the OTLP JSON structure
(``resourceSpans → scopeSpans → spans``) directly — no heavy OTel SDK dependency — group spans by
trace id, and hand each trace to the existing GenAI-aware normalizer.

The OTLP JSON encodes attribute values as ``{key, value:{stringValue|intValue|...}}`` and times as
stringified unix-nanoseconds; we decode both here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bellwether.ingest.otel import agentrun_from_otel_spans
from bellwether.schema import AgentRun

# Attributes consulted (in order) to identify the agent / task when not supplied explicitly.
_DEFAULT_AGENT_ATTRS = ("gen_ai.agent.name", "agent.id", "agent.name", "service.name")
_DEFAULT_TASK_ATTRS = ("gen_ai.task.class", "task.class", "agent.task")

# OTLP status codes: 0 UNSET, 1 OK, 2 ERROR.
_STATUS = {1: "ok", 2: "error"}


def _attr_value(v: dict[str, Any]) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        return [_attr_value(item) for item in v["arrayValue"].get("values", [])]
    return None


def flatten_attributes(attributes: Sequence[dict[str, Any]] | None) -> dict[str, Any]:
    """OTLP attribute list -> flat ``{key: value}`` dict."""
    return {a["key"]: _attr_value(a.get("value", {})) for a in (attributes or []) if "key" in a}


def _first_attr(attrs: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for k in keys:
        if attrs.get(k) is not None:
            return attrs[k]
    return None


def _to_span_dict(span: dict[str, Any], resource_attrs: dict[str, Any]) -> dict[str, Any]:
    attrs = {**resource_attrs, **flatten_attributes(span.get("attributes"))}
    code = (span.get("status") or {}).get("code", 0)
    parent = span.get("parentSpanId") or None
    return {
        "span_id": span["spanId"],
        "parent_span_id": parent,
        "name": span.get("name", "span"),
        "start_time": int(span["startTimeUnixNano"]),
        "end_time": int(span["endTimeUnixNano"]),
        "status": _STATUS.get(int(code), "unset"),
        "attributes": attrs,
    }


def parse_otlp_json(
    payload: dict[str, Any],
    *,
    agent_id: str | None = None,
    task_class: str | None = None,
    agent_id_attrs: Sequence[str] = _DEFAULT_AGENT_ATTRS,
    task_class_attrs: Sequence[str] = _DEFAULT_TASK_ATTRS,
) -> list[AgentRun]:
    """Parse an OTLP/JSON ``ExportTraceServiceRequest`` into one ``AgentRun`` per trace.

    ``agent_id`` / ``task_class`` override attribute-based identification when provided.
    """
    traces: dict[str, list[dict[str, Any]]] = {}
    for rs in payload.get("resourceSpans", []):
        r_attrs = flatten_attributes((rs.get("resource") or {}).get("attributes"))
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                trace_id = str(span.get("traceId", "trace"))
                traces.setdefault(trace_id, []).append(_to_span_dict(span, r_attrs))

    runs: list[AgentRun] = []
    for trace_id, spans in traces.items():
        merged: dict[str, Any] = {}
        for s in spans:
            merged.update(s["attributes"])
        aid = agent_id or _first_attr(merged, agent_id_attrs) or "unknown-agent"
        tcl = task_class or _first_attr(merged, task_class_attrs) or "default"
        runs.append(
            agentrun_from_otel_spans(spans, run_id=trace_id, agent_id=str(aid), task_class=str(tcl))
        )
    return runs
