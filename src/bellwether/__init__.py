"""BELLWETHER — behavioral drift detection for AI agents.

The canonical ``AgentRun`` schema is the contract every subsystem is built against; the
``Bellwether`` SDK is the adoption wedge (instrument an agent, get drift detection).
"""

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftReport, DriftScorer, ScoringConfig, Thresholds
from bellwether.monitor import DriftMonitor
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
from bellwether.sdk import Bellwether
from bellwether.triage import TriageExplainer

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
    "Bellwether",
    "BaselineManager",
    "DriftScorer",
    "ScoringConfig",
    "Thresholds",
    "DriftReport",
    "TriageExplainer",
    "DriftMonitor",
]

__version__ = "0.0.1"
