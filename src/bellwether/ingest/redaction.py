"""Ingest-time PII redaction — a first-class feature per the brief (privacy-first / local).

Traces may carry sensitive data in tool arguments, attributes, and error strings. We redact
*before* anything is persisted, so the store never holds raw PII. Phase 0 ships a solid
pattern-based redactor (emails, phones, credit cards, SSNs, IPs, common API-key shapes); a
pluggable NER/LLM redactor can be layered behind the same interface later.

Behavioral features (counts, latencies, token counts, tool *names*) are unaffected — we redact
*values*, not structure — so drift detection quality is preserved while raw content is not
retained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bellwether.schema import AgentRun, Span

# Order matters: more specific patterns first so a credit card isn't partially eaten by a
# looser numeric rule. Each entry is (label, compiled-pattern).
_DEFAULT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    # Credit-card-ish: 13-16 digits, optionally separated by spaces/dashes.
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    # Common secret shapes: sk-..., Bearer tokens, long base64-ish blobs prefixed by key=.
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9]{16,}\b")),
    ("BEARER", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}")),
    # Phone numbers (loose international/US). Kept last among value rules to avoid eating CCs.
    (
        "PHONE",
        re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .\-]?)?(?:\(?\d{3}\)?[ .\-]?)\d{3}[ .\-]?\d{4}(?!\d)"),
    ),
]


@dataclass
class RedactionConfig:
    """Configuration for the redactor.

    ``enabled=False`` is an explicit, auditable opt-out (e.g., a fully trusted local dataset).
    ``redact_keys`` redacts a whole value when its dict key matches (case-insensitive), useful
    for fields like ``password``/``authorization`` whose values have no fixed pattern.
    """

    enabled: bool = True
    patterns: list[tuple[str, re.Pattern[str]]] = field(
        default_factory=lambda: list(_DEFAULT_PATTERNS)
    )
    redact_keys: frozenset[str] = frozenset(
        {"password", "secret", "token", "authorization", "api_key", "apikey", "access_token"}
    )
    placeholder: str = "[REDACTED:{label}]"

    def _ph(self, label: str) -> str:
        return self.placeholder.format(label=label)


def redact_text(text: str, config: RedactionConfig) -> str:
    """Apply all value patterns to a single string."""
    if not config.enabled:
        return text
    out = text
    for label, pattern in config.patterns:
        out = pattern.sub(config._ph(label), out)
    return out


def _redact_value(value: object, config: RedactionConfig, *, key: str | None = None) -> object:
    if key is not None and key.lower() in config.redact_keys:
        return config._ph("KEY")
    if isinstance(value, str):
        return redact_text(value, config)
    if isinstance(value, dict):
        return {k: _redact_value(v, config, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, config) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v, config) for v in value)
    return value


def _redact_span(span: Span, config: RedactionConfig) -> Span:
    return span.model_copy(
        update={
            "tool_args": _redact_value(span.tool_args, config),
            "attributes": _redact_value(span.attributes, config),
            "error": redact_text(span.error, config) if span.error else span.error,
        }
    )


def redact_run(run: AgentRun, config: RedactionConfig | None = None) -> AgentRun:
    """Return a redacted copy of ``run``. Structure and behavioral signals are preserved.

    Idempotent: re-redacting already-redacted content is a no-op (placeholders don't match the
    PII patterns).
    """
    config = config or RedactionConfig()
    if not config.enabled:
        return run
    return run.model_copy(
        update={
            "spans": [_redact_span(s, config) for s in run.spans],
            "metadata": _redact_value(run.metadata, config),
        }
    )
