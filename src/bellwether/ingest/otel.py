"""Minimal OpenTelemetry span -> ``AgentRun`` normalizer.

The brief is emphatic: *consume* OTel, do not reinvent a trace format. This module maps OTel
spans (in the lightweight dict shape an exporter hands us) onto the canonical schema, reading
the GenAI/agent semantic-convention attributes where present. It is intentionally tolerant —
real exporters vary — and falls back to sane defaults rather than rejecting partial spans.

Phase 0 accepts a simple span-dict form so we are not yet coupled to the full OTel SDK object
model; the Phase 3 OTLP receiver will adapt protobuf spans onto the same ``_normalize_span``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from bellwether.schema import (
    AgentRun,
    ConfigFingerprint,
    RunStatus,
    Span,
    SpanKind,
    SpanStatus,
)

# OTel GenAI semantic-convention attribute keys we read (subset; extend as conventions evolve).
_ATTR_MODEL = ("gen_ai.request.model", "gen_ai.response.model", "llm.model_name")
_ATTR_SYSTEM = ("gen_ai.system", "llm.system")
_ATTR_IN_TOKENS = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "llm.token_count.prompt",
)
_ATTR_OUT_TOKENS = (
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.token_count.completion",
)
_ATTR_TOOL_NAME = ("gen_ai.tool.name", "tool.name")
_ATTR_TEMPERATURE = ("gen_ai.request.temperature",)
_ATTR_TOP_P = ("gen_ai.request.top_p",)
_ATTR_MAX_TOKENS = ("gen_ai.request.max_tokens",)
_ATTR_OPERATION = ("gen_ai.operation.name",)


def _first(attrs: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for k in keys:
        if k in attrs and attrs[k] is not None:
            return attrs[k]
    return None


def _to_dt(value: Any) -> datetime:
    """Accept unix-nanos (int), unix-seconds (float), ISO string, or datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int):
        # Heuristic: OTel uses unix nanoseconds.
        return datetime.fromtimestamp(value / 1e9, tz=UTC)
    if isinstance(value, float):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"unsupported timestamp type: {type(value)!r}")


def _infer_kind(name: str, attrs: dict[str, Any]) -> SpanKind:
    op = _first(attrs, _ATTR_OPERATION)
    if _first(attrs, _ATTR_TOOL_NAME) is not None:
        return SpanKind.TOOL
    if _first(attrs, _ATTR_MODEL) is not None or op in {"chat", "text_completion", "generate"}:
        return SpanKind.LLM
    lname = name.lower()
    if "retriev" in lname or "rag" in lname or "embed" in lname:
        return SpanKind.RETRIEVAL
    if "agent" in lname or "graph" in lname:
        return SpanKind.AGENT
    return SpanKind.OTHER


def _normalize_span(raw: dict[str, Any]) -> Span:
    attrs: dict[str, Any] = dict(raw.get("attributes") or {})
    name = str(raw.get("name", "span"))
    kind = _infer_kind(name, attrs)

    status_raw = str(raw.get("status", "ok")).lower()
    status = {
        "ok": SpanStatus.OK,
        "error": SpanStatus.ERROR,
        "unset": SpanStatus.UNSET,
    }.get(status_raw, SpanStatus.UNSET)

    in_tok = _first(attrs, _ATTR_IN_TOKENS)
    out_tok = _first(attrs, _ATTR_OUT_TOKENS)

    return Span(
        span_id=str(raw["span_id"]),
        parent_span_id=(str(raw["parent_span_id"]) if raw.get("parent_span_id") else None),
        name=name,
        kind=kind,
        start_time=_to_dt(raw["start_time"]),
        end_time=_to_dt(raw["end_time"]),
        status=status,
        input_tokens=int(in_tok) if in_tok is not None else None,
        output_tokens=int(out_tok) if out_tok is not None else None,
        cost_usd=(float(raw["cost_usd"]) if raw.get("cost_usd") is not None else None),
        model=_first(attrs, _ATTR_MODEL),
        tool_name=_first(attrs, _ATTR_TOOL_NAME),
        tool_args=dict(raw.get("tool_args") or {}),
        error=raw.get("error"),
        attributes=attrs,
    )


def _fingerprint_from_spans(spans: list[Span], extra: dict[str, Any]) -> ConfigFingerprint:
    """Best-effort fingerprint inferred from the first LLM span + caller-provided extras."""
    llm = next((s for s in spans if s.kind == SpanKind.LLM), None)
    attrs = llm.attributes if llm else {}
    return ConfigFingerprint(
        model_id=(llm.model if llm else None) or extra.get("model_id"),
        provider=_first(attrs, _ATTR_SYSTEM) or extra.get("provider"),
        temperature=_first(attrs, _ATTR_TEMPERATURE),
        top_p=_first(attrs, _ATTR_TOP_P),
        max_tokens=_first(attrs, _ATTR_MAX_TOKENS),
        prompt_version=extra.get("prompt_version"),
        framework=extra.get("framework"),
        framework_version=extra.get("framework_version"),
        agent_version=extra.get("agent_version"),
    )


def agentrun_from_otel_spans(
    spans: list[dict[str, Any]],
    *,
    run_id: str,
    agent_id: str,
    task_class: str = "default",
    config_extra: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentRun:
    """Normalize a list of OTel span dicts (one trace) into a canonical ``AgentRun``.

    ``config_extra`` supplies fingerprint fields OTel attributes don't carry (e.g.
    ``prompt_version``, ``framework``). The run's time bounds and status are derived from the
    spans.
    """
    if not spans:
        raise ValueError("cannot build an AgentRun from zero spans")

    norm = [_normalize_span(s) for s in spans]
    start = min(s.start_time for s in norm)
    end = max(s.end_time for s in norm)
    status = RunStatus.ERROR if any(s.status == SpanStatus.ERROR for s in norm) else RunStatus.OK

    return AgentRun(
        run_id=run_id,
        agent_id=agent_id,
        task_class=task_class,
        start_time=start,
        end_time=end,
        status=status,
        spans=norm,
        config_fingerprint=_fingerprint_from_spans(norm, config_extra or {}),
        metadata=dict(metadata or {}),
    )
