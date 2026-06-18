"""Canonical, versioned ``AgentRun`` event schema.

Every ingest path (OTel receiver, SDK decorators, MCP proxy) normalizes to these types.
Downstream feature extractors, the baseline manager, detectors, and the store all depend on
this contract, so it is versioned (``SCHEMA_VERSION``) and changes are breaking unless the
version is bumped with a migration.

Design notes:
- A run is a tree/DAG of spans (LLM calls, tool calls, sub-agent spawns). We keep spans as a
  flat list with ``parent_span_id`` links rather than a nested structure — this matches how
  OpenTelemetry emits spans and keeps streaming ingest simple. Tree helpers reconstruct
  structure on demand.
- ``ConfigFingerprint`` is load-bearing for drift-vs-intended-change (design doc 03): a new
  fingerprint opens a new baseline lineage instead of polluting the old one.
- ``InjectedFault`` carries eval ground truth (design doc 02). It is ``None`` for real runs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bump on any breaking change to the structures below; ingest records the version it wrote so
# the store can migrate. Phase 0 starts at 1.
SCHEMA_VERSION = 1


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SpanKind(StrEnum):
    """The kind of work a span represents. Mirrors OTel GenAI semantic conventions loosely."""

    LLM = "llm"
    TOOL = "tool"
    AGENT = "agent"
    SUB_AGENT = "sub_agent"
    RETRIEVAL = "retrieval"
    OTHER = "other"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


class RunStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    UNKNOWN = "unknown"


class _Frozen(BaseModel):
    """Base for value objects: immutable, forbids unknown fields to catch schema drift early."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ConfigFingerprint(_Frozen):
    """The configuration whose change is *intended* to change behavior (design doc 03).

    Baselines are keyed by (agent, task_class, fingerprint hash). A new fingerprint is, by
    definition, a candidate intended change and opens a fresh baseline lineage.
    """

    model_id: str | None = None
    model_version: str | None = None
    provider: str | None = None
    prompt_version: str | None = None
    tool_schema_hash: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    framework: str | None = None
    framework_version: str | None = None
    agent_version: str | None = None

    def hash(self) -> str:
        """Stable content hash. Order-independent and whitespace-independent."""
        payload = self.model_dump(exclude_none=False)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class DeployMarker(_Frozen):
    """An explicit, human-declared intended change (design doc 03).

    The strongest disambiguation signal: it timestamps an intended change and carries
    rationale into the audit log. Emitted via ``bellwether.mark_deploy(...)`` (Phase 1 SDK).
    """

    version: str
    note: str | None = None
    at: datetime = Field(default_factory=_utcnow)


class InjectedFault(_Frozen):
    """Eval ground truth (design doc 02). ``None`` on real runs.

    ``fault_type`` is a member of the fault taxonomy; ``severity`` is a 0..1 dial so the
    eval harness can test monotonicity (worse fault must not lower the drift score).
    ``onset_step`` is the span index at which the fault begins, used to compute lead-time.
    """

    fault_type: str
    severity: float = Field(ge=0.0, le=1.0)
    onset_step: int = Field(ge=0)
    # The span index at which a human/SLA would notice (the "visible failure" marker). Used as
    # ``t_fail`` for lead-time. May be None when not yet crossed within the run.
    visible_failure_step: int | None = Field(default=None, ge=0)
    params: dict[str, float] = Field(default_factory=dict)


class Span(BaseModel):
    """One unit of work within a run. Flat list + parent links reconstruct the tree."""

    model_config = ConfigDict(extra="forbid")

    span_id: str
    parent_span_id: str | None = None
    name: str
    kind: SpanKind = SpanKind.OTHER
    start_time: datetime
    end_time: datetime
    status: SpanStatus = SpanStatus.OK

    # Economic / temporal signals (None when not applicable to the span kind).
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)

    # LLM-specific.
    model: str | None = None

    # Tool-specific.
    tool_name: str | None = None
    tool_args: dict[str, object] = Field(default_factory=dict)

    error: str | None = None
    attributes: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_times(self) -> Span:
        if self.end_time < self.start_time:
            raise ValueError(f"span {self.span_id}: end_time precedes start_time")
        return self

    @property
    def duration_s(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


class AgentRun(BaseModel):
    """Canonical normalized representation of a single agent run (one trace).

    This is the unit feature extractors consume and the store persists. ``task_class`` is the
    conditioning variable for per-task-class baselines (design doc 04); when unknown it
    defaults to ``"default"`` so cold-start still works.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: str
    agent_id: str
    task_class: str = "default"

    start_time: datetime
    end_time: datetime
    status: RunStatus = RunStatus.OK

    spans: list[Span] = Field(default_factory=list)

    config_fingerprint: ConfigFingerprint = Field(default_factory=ConfigFingerprint)
    deploy_marker: DeployMarker | None = None

    # Eval ground truth — present only for fixture/replay runs (design doc 02).
    injected_fault: InjectedFault | None = None

    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_run(self) -> AgentRun:
        if self.end_time < self.start_time:
            raise ValueError(f"run {self.run_id}: end_time precedes start_time")
        ids = [s.span_id for s in self.spans]
        if len(ids) != len(set(ids)):
            raise ValueError(f"run {self.run_id}: duplicate span_id present")
        idset = set(ids)
        for s in self.spans:
            if s.parent_span_id is not None and s.parent_span_id not in idset:
                raise ValueError(
                    f"run {self.run_id}: span {s.span_id} references unknown parent "
                    f"{s.parent_span_id}"
                )
        return self

    # --- convenience accessors used by feature extractors -------------------------------

    @property
    def fingerprint_hash(self) -> str:
        return self.config_fingerprint.hash()

    @property
    def baseline_key(self) -> tuple[str, str, str]:
        """The (agent, task_class, fingerprint) key a baseline lineage is stored under."""
        return (self.agent_id, self.task_class, self.fingerprint_hash)

    @property
    def duration_s(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @property
    def step_count(self) -> int:
        return len(self.spans)

    def spans_of_kind(self, kind: SpanKind) -> list[Span]:
        return [s for s in self.spans if s.kind == kind]

    def root_spans(self) -> list[Span]:
        return [s for s in self.spans if s.parent_span_id is None]

    def children_of(self, span_id: str) -> list[Span]:
        return [s for s in self.spans if s.parent_span_id == span_id]

    def max_depth(self) -> int:
        """Maximum span-tree depth (1 = flat). Structural feature (design doc 03/§3)."""
        if not self.spans:
            return 0
        by_id = {s.span_id: s for s in self.spans}
        memo: dict[str, int] = {}

        def depth(span_id: str) -> int:
            if span_id in memo:
                return memo[span_id]
            parent = by_id[span_id].parent_span_id
            d = 1 if parent is None else depth(parent) + 1
            memo[span_id] = d
            return d

        return max(depth(s.span_id) for s in self.spans)
