"""DriftMonitor: a queryable runtime facade over baselines, scoring, and the audit log.

This is the surface a zero-code form factor exposes — an MCP server or an OTel/MCP proxy wraps
these methods as tools so clients can *query drift status* and feed traces without code changes
(brief §1/§9). ``ingest`` accepts a canonical ``AgentRun`` (from the OTel normalizer, the SDK, or
a proxy), redacts it, scores it against the learned baseline, and records any alert to the
tamper-evident audit log; the ``*_status``/``recent_alerts``/``baselines`` queries return plain
dicts suitable for serialization to an MCP/JSON client.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftReport, DriftScorer, Thresholds
from bellwether.governance import AuditLog, EventType
from bellwether.ingest import RedactionConfig, RunStore, redact_run
from bellwether.schema import AgentRun


@dataclass
class DriftMonitor:
    """Live monitoring facade: ingest runs, score them, and answer drift-status queries."""

    manager: BaselineManager
    scorer: DriftScorer
    audit: AuditLog = field(default_factory=AuditLog)
    store: RunStore | None = None
    redaction: RedactionConfig | None = None
    thresholds: dict[tuple[str, str, str], Thresholds] = field(default_factory=dict)
    learn_when_no_baseline: bool = False
    _last: dict[str, DriftReport] = field(default_factory=dict, init=False)

    # --- ingest -------------------------------------------------------------------------

    def set_thresholds(self, key: tuple[str, str, str], thresholds: Thresholds) -> None:
        self.thresholds[key] = thresholds

    def learn(self, run: AgentRun) -> None:
        self.manager.learn(self._prepare(run))

    def _prepare(self, run: AgentRun) -> AgentRun:
        if self.redaction is not None:
            run = redact_run(run, self.redaction)
        if self.store is not None:
            self.store.add(run)
        return run

    def ingest(self, run: AgentRun) -> DriftReport | None:
        """Score a run against its baseline; record any alert. Returns None if no baseline yet."""
        run = self._prepare(run)
        baseline = self.manager.baseline_for(run)
        if baseline is None:
            if self.learn_when_no_baseline:
                self.manager.learn(run)
            return None
        report = self.scorer.evaluate(run, baseline, self.thresholds.get(run.baseline_key))
        self._last[run.agent_id] = report
        if report.alert.triggered:
            a = report.alert
            self.audit.record_alert(
                run.agent_id,
                report.summary(),
                {
                    "run_id": run.run_id,
                    "signature": a.signature,
                    "primary_family": str(a.primary_family) if a.primary_family else None,
                    "primary_feature": a.primary_feature,
                    "step_index": a.step_index,
                    "drift_score": round(a.drift_score, 4),
                },
            )
        return report

    # --- queries (MCP-tool surface) -----------------------------------------------------

    def drift_status(self, agent_id: str) -> dict[str, object]:
        report = self._last.get(agent_id)
        if report is None:
            return {"agent_id": agent_id, "status": "unknown", "detail": "no run scored yet"}
        a = report.alert
        return {
            "agent_id": agent_id,
            "status": "drift" if a.triggered else "ok",
            "run_id": report.run_id,
            "signature": a.signature,
            "attribution": (
                f"{a.primary_family}/{a.primary_feature}" if a.primary_family else None
            ),
            "step_index": a.step_index,
            "max_drift": round(report.max_drift, 4),
        }

    def recent_alerts(self, n: int = 10) -> list[dict[str, object]]:
        alerts = self.audit.query(event_type=EventType.ALERT)[-n:]
        return [
            {"seq": e.seq, "ts": e.ts, "agent_id": e.subject, "summary": e.rationale, **e.payload}
            for e in alerts
        ]

    def baselines(self) -> list[dict[str, object]]:
        out = []
        for key in self.manager.keys:
            b = self.manager.get_or_create(key)
            out.append(
                {
                    "agent_id": key[0],
                    "task_class": key[1],
                    "fingerprint": key[2],
                    "observations": b.total_observations,
                }
            )
        return out

    def status(self) -> dict[str, object]:
        return {
            "baselines": len(self.manager.keys),
            "agents": sorted({k[0] for k in self.manager.keys}),
            "total_alerts": len(self.audit.query(event_type=EventType.ALERT)),
            "audit_events": len(self.audit),
            "audit_verified": self.audit.verify(),
        }
