"""BELLWETHER — behavioral drift detection for AI agents.

Public surface is intentionally tiny in Phase 0. The canonical event schema is the
contract every other subsystem is built against.
"""

from bellwether.schema import (
    SCHEMA_VERSION,
    AgentRun,
    ConfigFingerprint,
    DeployMarker,
    InjectedFault,
    RunStatus,
    Span,
    SpanKind,
    SpanStatus,
)

__all__ = [
    "SCHEMA_VERSION",
    "AgentRun",
    "ConfigFingerprint",
    "DeployMarker",
    "InjectedFault",
    "RunStatus",
    "Span",
    "SpanKind",
    "SpanStatus",
]

__version__ = "0.0.1"
