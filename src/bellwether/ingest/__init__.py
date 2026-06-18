"""Ingest layer: normalize incoming traces to ``AgentRun`` and persist them.

Phase 0 ships:
- ``store``: an embeddable DuckDB-backed event store (zero infra, local-first).
- ``otel``: a minimal OpenTelemetry span -> ``AgentRun`` normalizer.
- ``redaction``: ingest-time PII redaction (first-class per the brief).
"""

from bellwether.ingest.otlp import parse_otlp_json
from bellwether.ingest.redaction import RedactionConfig, redact_run
from bellwether.ingest.store import RunStore

__all__ = ["RunStore", "RedactionConfig", "redact_run", "parse_otlp_json"]
