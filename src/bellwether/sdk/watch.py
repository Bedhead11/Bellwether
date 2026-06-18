"""SDK instrumentation: build canonical ``AgentRun`` traces from in-process agent calls.

Uses ``contextvars`` so nested ``llm``/``tool``/``agent`` context managers form the correct span
tree even across async tasks. On run exit the assembled run is redacted and emitted to the
configured sinks. Live scoring is optional: in ``learn`` mode the run updates the baseline; in
monitor mode it is scored and any :class:`DriftReport` is delivered via ``on_report``.
"""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import perf_counter

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftReport, DriftScorer, Thresholds
from bellwether.ingest import RedactionConfig, RunStore, redact_run
from bellwether.schema import (
    AgentRun,
    ConfigFingerprint,
    DeployMarker,
    RunStatus,
    Span,
    SpanKind,
    SpanStatus,
)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class _SpanBuilder:
    span_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    start_dt: datetime
    start_perf: float
    end_dt: datetime | None = None
    status: SpanStatus = SpanStatus.OK
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    attributes: dict[str, object] = field(default_factory=dict)

    def finish(self) -> None:
        if self.end_dt is None:
            self.end_dt = self.start_dt + timedelta(seconds=perf_counter() - self.start_perf)

    def to_span(self) -> Span:
        self.finish()
        assert self.end_dt is not None
        return Span(
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            kind=self.kind,
            start_time=self.start_dt,
            end_time=self.end_dt,
            status=self.status,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
            model=self.model,
            tool_name=self.tool_name,
            tool_args=dict(self.tool_args),
            error=self.error,
            attributes=dict(self.attributes),
        )


class SpanHandle:
    """Handle returned by ``llm``/``tool``/``agent`` for filling in details after the call."""

    def __init__(self, builder: _SpanBuilder) -> None:
        self._b = builder

    def set_tokens(self, input: int | None = None, output: int | None = None) -> SpanHandle:
        if input is not None:
            self._b.input_tokens = input
        if output is not None:
            self._b.output_tokens = output
        return self

    def set_cost(self, cost_usd: float) -> SpanHandle:
        self._b.cost_usd = cost_usd
        return self

    def set_attribute(self, key: str, value: object) -> SpanHandle:
        self._b.attributes[key] = value
        return self


@dataclass
class _RunBuilder:
    agent_id: str
    task_class: str
    fingerprint: ConfigFingerprint
    deploy_marker: DeployMarker | None
    root: _SpanBuilder
    spans: list[_SpanBuilder] = field(default_factory=list)
    parent_stack: list[str] = field(default_factory=list)

    def current_parent(self) -> str:
        return self.parent_stack[-1]

    def build(self, run_id: str) -> AgentRun:
        self.root.finish()
        all_builders = [self.root, *self.spans]
        spans = [b.to_span() for b in all_builders]
        status = (
            RunStatus.ERROR if any(s.status == SpanStatus.ERROR for s in spans) else RunStatus.OK
        )
        return AgentRun(
            run_id=run_id,
            agent_id=self.agent_id,
            task_class=self.task_class,
            start_time=self.root.start_dt,
            end_time=self.root.end_dt or _utcnow(),
            status=status,
            spans=spans,
            config_fingerprint=self.fingerprint,
            deploy_marker=self.deploy_marker,
        )


@dataclass
class RunHandle:
    """Populated after a ``watch`` block exits: the assembled run and any drift report."""

    run: AgentRun | None = None
    report: DriftReport | None = None


_run_var: contextvars.ContextVar[_RunBuilder | None] = contextvars.ContextVar(
    "bellwether_run", default=None
)


class Bellwether:
    """The SDK entry point: configure once, then instrument runs."""

    def __init__(
        self,
        agent_id: str,
        *,
        task_class: str = "default",
        fingerprint: ConfigFingerprint | None = None,
        store: RunStore | None = None,
        manager: BaselineManager | None = None,
        scorer: DriftScorer | None = None,
        thresholds: Thresholds | None = None,
        learn: bool = False,
        redaction: RedactionConfig | None = None,
        on_report: Callable[[DriftReport], None] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.task_class = task_class
        self.fingerprint = fingerprint or ConfigFingerprint()
        self.store = store
        self.manager = manager
        self.scorer = scorer
        self.thresholds = thresholds
        self.learn = learn
        self.redaction = redaction
        self.on_report = on_report
        self._pending_deploy: DeployMarker | None = None

    # --- deploy markers (design doc 03) -------------------------------------------------

    def mark_deploy(self, version: str, note: str | None = None) -> None:
        """Declare an intended change; attached to the next run for drift disambiguation."""
        self._pending_deploy = DeployMarker(version=version, note=note)

    # --- run + span context managers ----------------------------------------------------

    @contextmanager
    def watch(
        self,
        *,
        task_class: str | None = None,
        fingerprint: ConfigFingerprint | None = None,
    ) -> Iterator[RunHandle]:
        """Open a run scope. Records everything instrumented inside it as one ``AgentRun``."""
        root = _SpanBuilder(
            span_id=_new_id(),
            parent_span_id=None,
            name=f"agent.{self.agent_id}",
            kind=SpanKind.AGENT,
            start_dt=_utcnow(),
            start_perf=perf_counter(),
        )
        rb = _RunBuilder(
            agent_id=self.agent_id,
            task_class=task_class or self.task_class,
            fingerprint=fingerprint or self.fingerprint,
            deploy_marker=self._pending_deploy,
            root=root,
            parent_stack=[root.span_id],
        )
        self._pending_deploy = None
        handle = RunHandle()
        token = _run_var.set(rb)
        try:
            yield handle
        except Exception as exc:
            root.status = SpanStatus.ERROR
            root.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            root.finish()
            _run_var.reset(token)
            run = rb.build(run_id=_new_id())
            if self.redaction is not None:
                run = redact_run(run, self.redaction)
            handle.run = run
            handle.report = self._emit(run)

    @contextmanager
    def _span(self, name: str, kind: SpanKind, **fields: object) -> Iterator[SpanHandle]:
        rb = _run_var.get()
        if rb is None:
            # Not inside a watch() scope: yield an inert handle so callers don't crash.
            yield SpanHandle(_SpanBuilder(_new_id(), None, name, kind, _utcnow(), perf_counter()))
            return
        builder = _SpanBuilder(
            span_id=_new_id(),
            parent_span_id=rb.current_parent(),
            name=name,
            kind=kind,
            start_dt=_utcnow(),
            start_perf=perf_counter(),
            **fields,  # type: ignore[arg-type]
        )
        rb.spans.append(builder)
        rb.parent_stack.append(builder.span_id)
        try:
            yield SpanHandle(builder)
        except Exception as exc:
            builder.status = SpanStatus.ERROR
            builder.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            builder.finish()
            rb.parent_stack.pop()

    def llm(
        self,
        *,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        name: str = "llm.call",
    ) -> AbstractContextManager[SpanHandle]:
        return self._span(
            name,
            SpanKind.LLM,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    def tool(
        self, name: str, *, args: Mapping[str, object] | None = None
    ) -> AbstractContextManager[SpanHandle]:
        return self._span(f"tool.{name}", SpanKind.TOOL, tool_name=name, tool_args=dict(args or {}))

    def agent(self, name: str) -> AbstractContextManager[SpanHandle]:
        return self._span(f"agent.{name}", SpanKind.SUB_AGENT)

    # --- sink routing -------------------------------------------------------------------

    def _emit(self, run: AgentRun) -> DriftReport | None:
        if self.store is not None:
            self.store.add(run)

        report: DriftReport | None = None
        if self.manager is not None:
            if self.learn:
                self.manager.learn(run)
            elif self.scorer is not None:
                baseline = self.manager.baseline_for(run)
                if baseline is not None:
                    report = self.scorer.evaluate(run, baseline, self.thresholds)
                    if report is not None and self.on_report is not None:
                        self.on_report(report)
        return report
