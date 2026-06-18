"""ChangeAwareMonitor: distinguish intended change from drift (design doc 03).

Wraps the baseline manager + scorer with the disambiguation logic:

| step-change? | declared (new fingerprint / deploy marker)? | verdict |
|---|---|---|
| no  | —   | monitoring |
| yes | yes | intended change → quarantine → accept-new-normal |
| yes | no  | **drift** (fires) |

A new fingerprint opens a fresh baseline lineage in *quarantine* (learned, not alerted). Within a
stable fingerprint, a Page-Hinkley detector over the per-run drift stream confirms a *persistent*
regime shift (vs a transient spike); an unexplained persistent shift is genuine drift. A deploy
marker opens a learning window (resets the change-point detector). ``accept_new_normal`` promotes
a quarantined lineage to active, audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from bellwether.baseline import BaselineManager
from bellwether.change.page_hinkley import PageHinkley
from bellwether.detect import DriftReport, DriftScorer, Thresholds
from bellwether.governance import AuditLog, EventType
from bellwether.schema import AgentRun


class VerdictKind(StrEnum):
    WARMUP = "warmup"  # still establishing the baseline
    MONITORING = "monitoring"  # normal; no drift
    INTENDED_CHANGE = "intended_change"  # declared change → quarantined, not drift
    DRIFT = "drift"  # unexplained change → alert


@dataclass(frozen=True, slots=True)
class ChangeVerdict:
    kind: VerdictKind
    run_id: str
    agent_id: str
    fingerprint: str
    active_fingerprint: str | None
    regime_change: bool
    report: DriftReport | None
    note: str = ""


@dataclass
class ChangeAwareMonitor:
    manager: BaselineManager
    scorer: DriftScorer
    audit: AuditLog = field(default_factory=AuditLog)
    thresholds: dict[tuple[str, str, str], Thresholds] = field(default_factory=dict)
    ph_delta: float = 0.01
    ph_threshold: float = 1.5
    ph_min_samples: int = 8
    # Only fold a run into the active baseline if it looks benign; this stops the baseline from
    # silently absorbing a developing drift (which would mask the very regime change we watch for).
    learn_ceiling: float = 0.9

    # The fingerprint currently accepted as "normal" per (agent, task_class).
    _active_fp: dict[tuple[str, str], str] = field(default_factory=dict, init=False)
    # Fingerprints seen since a declared change, awaiting accept-new-normal.
    _quarantine: dict[tuple[str, str], set[str]] = field(default_factory=dict, init=False)
    _ph: dict[tuple[str, str, str], PageHinkley] = field(default_factory=dict, init=False)
    # Whether a lineage's baseline has reached "warm" (so its drift stream is meaningful).
    _warm: dict[tuple[str, str, str], bool] = field(default_factory=dict, init=False)

    def _ph_for(self, key: tuple[str, str, str]) -> PageHinkley:
        ph = self._ph.get(key)
        if ph is None:
            ph = PageHinkley(self.ph_delta, self.ph_threshold, self.ph_min_samples)
            self._ph[key] = ph
        return ph

    def observe(self, run: AgentRun) -> ChangeVerdict:
        at = (run.agent_id, run.task_class)
        fp = run.fingerprint_hash
        active = self._active_fp.get(at)

        def verdict(
            kind: VerdictKind,
            report: DriftReport | None = None,
            *,
            regime: bool = False,
            note: str = "",
        ) -> ChangeVerdict:
            return ChangeVerdict(kind, run.run_id, run.agent_id, fp, active, regime, report, note)

        # First run for this (agent, task_class): adopt its fingerprint as the normal lineage.
        if active is None:
            self._active_fp[at] = fp
            self.manager.learn(run)
            return verdict(VerdictKind.WARMUP, note="establishing initial baseline")

        # An explicit deploy marker opens a learning window: reset the change-point detector so a
        # deploy-induced shift on the same config isn't mistaken for drift.
        if run.deploy_marker is not None:
            self._ph_for((at[0], at[1], active)).reset()
            self.audit.append(
                EventType.NOTE,
                f"{run.agent_id}/{run.task_class}",
                f"deploy marker {run.deploy_marker.version}: opening learning window",
                {"version": run.deploy_marker.version, "note": run.deploy_marker.note},
            )

        # A new fingerprint is a declared intended change → quarantine its candidate baseline.
        if fp != active:
            self._quarantine.setdefault(at, set()).add(fp)
            self.manager.learn(run)
            self.audit.append(
                EventType.BASELINE_UPDATE,
                f"{run.agent_id}/{run.task_class}",
                f"config change {active}→{fp}: quarantined; accept_new_normal to confirm",
                {"from": active, "to": fp},
            )
            return verdict(
                VerdictKind.INTENDED_CHANGE,
                note="declared config change; learning new baseline (not drift)",
            )

        # Same fingerprint: score against the active baseline.
        baseline = self.manager.baseline_for(run)
        if baseline is None:
            self.manager.learn(run)
            return verdict(VerdictKind.WARMUP, note="baseline not yet established")

        report = self.scorer.evaluate(run, baseline, self.thresholds.get(run.baseline_key))

        # The change-point detector only runs once the baseline is warm; the warmup ramp (drift
        # scores rising from ~0 to the benign level) would otherwise look like a regime change.
        # We reset it at the warm transition so it starts from the stationary benign regime.
        rs = next((o for o in report.observation_scores if o.kind == "run_summary"), None)
        is_warm = rs is not None and not rs.warmup
        regime = False
        if is_warm:
            if not self._warm.get(run.baseline_key):
                self._warm[run.baseline_key] = True
                self._ph_for(run.baseline_key).reset()
            regime = self._ph_for(run.baseline_key).update(report.max_drift)

        if report.alert.triggered or regime:
            why = "persistent regime change" if regime else "anomalous run"
            self.audit.record_alert(
                f"{run.agent_id}/{run.task_class}",
                f"unexplained {why}: {report.summary()}",
                {
                    "run_id": run.run_id,
                    "regime_change": regime,
                    "max_drift": round(report.max_drift, 4),
                },
            )
            return verdict(VerdictKind.DRIFT, report, regime=regime, note=f"unexplained {why}")

        # Benign and stable: fold this run into the active baseline (slow online adaptation), but
        # only if it looks clearly benign — never learn a run that is drifting.
        if report.max_drift < self.learn_ceiling:
            self.manager.learn(run)
        return verdict(VerdictKind.MONITORING, report)

    def accept_new_normal(
        self, agent_id: str, task_class: str, fingerprint: str | None = None
    ) -> str:
        """Promote a quarantined fingerprint to the active baseline (audited)."""
        at = (agent_id, task_class)
        pending = self._quarantine.get(at) or set()
        if fingerprint is None:
            if not pending:
                raise ValueError(f"no quarantined change to accept for {at}")
            fingerprint = sorted(pending)[-1]
        elif fingerprint not in pending:
            raise ValueError(f"fingerprint {fingerprint} is not quarantined for {at}")

        previous = self._active_fp.get(at)
        self._active_fp[at] = fingerprint
        pending.discard(fingerprint)
        self._ph.pop((agent_id, task_class, fingerprint), None)
        self.audit.append(
            EventType.BASELINE_ACCEPTED,
            f"{agent_id}/{task_class}",
            f"accept-new-normal: {previous} → {fingerprint}",
            {"from": previous, "to": fingerprint},
        )
        return fingerprint

    # --- introspection ------------------------------------------------------------------

    def active_fingerprint(self, agent_id: str, task_class: str) -> str | None:
        return self._active_fp.get((agent_id, task_class))

    def quarantined(self, agent_id: str, task_class: str) -> set[str]:
        return set(self._quarantine.get((agent_id, task_class)) or set())
