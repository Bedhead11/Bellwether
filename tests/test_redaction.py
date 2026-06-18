"""L0 unit tests for ingest-time PII redaction."""

from datetime import UTC, datetime, timedelta

from bellwether.ingest import RedactionConfig, redact_run
from bellwether.ingest.redaction import redact_text
from bellwether.schema import AgentRun, Span, SpanKind

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _run_with(tool_args: dict[str, object], error: str | None = None) -> AgentRun:
    span = Span(
        span_id="s0",
        name="tool.call",
        kind=SpanKind.TOOL,
        start_time=T0,
        end_time=T0 + timedelta(seconds=1),
        tool_name="search",
        tool_args=tool_args,
        error=error,
    )
    return AgentRun(
        run_id="r", agent_id="a", start_time=T0, end_time=T0 + timedelta(seconds=1), spans=[span]
    )


def test_redacts_common_pii() -> None:
    assert "REDACTED:EMAIL" in redact_text("ping me at jane.doe@example.com", RedactionConfig())
    assert "REDACTED:SSN" in redact_text("ssn 123-45-6789", RedactionConfig())
    assert "REDACTED:IP" in redact_text("from 192.168.0.1", RedactionConfig())
    assert "REDACTED:API_KEY" in redact_text("key sk-abcdEFGH012345678901", RedactionConfig())


def test_redacts_inside_run_tool_args() -> None:
    run = _run_with(
        {"email": "user@corp.com", "note": "call 415-555-1234"}, error="leaked sk-ABCDEFGH01234567"
    )
    red = redact_run(run)
    args = red.spans[0].tool_args
    assert "REDACTED" in str(args["email"])
    assert "REDACTED" in str(args["note"])
    assert red.spans[0].error is not None and "REDACTED" in red.spans[0].error


def test_redact_keys_blanks_whole_value() -> None:
    run = _run_with({"password": "anything-at-all", "q": "normal text"})
    red = redact_run(run)
    assert red.spans[0].tool_args["password"] == "[REDACTED:KEY]"
    assert red.spans[0].tool_args["q"] == "normal text"


def test_idempotent() -> None:
    run = _run_with({"email": "user@corp.com"})
    once = redact_run(run)
    twice = redact_run(once)
    assert once.spans[0].tool_args == twice.spans[0].tool_args


def test_disabled_is_passthrough() -> None:
    run = _run_with({"email": "user@corp.com"})
    red = redact_run(run, RedactionConfig(enabled=False))
    assert red.spans[0].tool_args["email"] == "user@corp.com"


def test_structure_preserved() -> None:
    """Redaction must not change behavioral structure (counts, names, tokens)."""
    run = _run_with({"email": "user@corp.com"})
    red = redact_run(run)
    assert red.step_count == run.step_count
    assert red.spans[0].tool_name == run.spans[0].tool_name
    assert red.spans[0].kind == run.spans[0].kind
