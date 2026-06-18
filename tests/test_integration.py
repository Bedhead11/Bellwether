"""Full-stack integration: ingest -> score -> audit -> query -> store -> dashboard.

A single test that wires the major subsystems together as a regression guard that they keep
composing, end to end.
"""

from bellwether.baseline import BaselineManager
from bellwether.dashboard import render_dashboard
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run
from bellwether.ingest import RedactionConfig, RunStore
from bellwether.monitor import DriftMonitor


def test_end_to_end_monitor_audit_store_dashboard() -> None:
    agent = FixtureAgent(agent_id="prod-agent", task_class="qa")

    # 1) Learn a baseline and calibrate thresholds.
    mgr = BaselineManager()
    for s in range(120):
        mgr.learn(agent.clean_run(seed=s))
    baseline = mgr.baseline_for(agent.clean_run(seed=0))
    assert baseline is not None
    scorer = DriftScorer(ScoringConfig(min_samples=30))
    thr = calibrate_threshold(
        scorer, baseline, [agent.clean_run(seed=s) for s in range(120, 270)], target_fp_rate=0.02
    )

    # 2) A monitor that redacts at ingest, persists to a store, and audits alerts.
    with RunStore() as store:
        monitor = DriftMonitor(manager=mgr, scorer=scorer, store=store, redaction=RedactionConfig())
        monitor.set_thresholds(agent.clean_run(seed=0).baseline_key, thr)

        reports = []
        # Healthy traffic.
        for s in range(900, 910):
            reports.append(monitor.ingest(agent.clean_run(seed=s)))
        # A clear fault.
        fault = faulted_run(agent, 4242, FaultSpec("tool_misselection", 0.9, onset_step=1))
        fault_report = monitor.ingest(fault)
        reports.append(fault_report)

        # 3) The fault was detected, attributed, audited, and the chain verifies.
        assert fault_report is not None and fault_report.alert.triggered
        assert monitor.audit.verify()
        assert len(monitor.recent_alerts()) >= 1
        status = monitor.status()
        assert status["audit_verified"] is True
        assert "prod-agent" in status["agents"]  # type: ignore[operator]

        # 4) Drift status query reflects the latest verdict.
        assert monitor.drift_status("prod-agent")["status"] == "drift"

        # 5) Runs were persisted to the store.
        assert store.count() == 11

        # 6) The dashboard renders the whole picture as self-contained HTML.
        html = render_dashboard(reports=[r for r in reports if r is not None], audit=monitor.audit)
        assert html.startswith("<!doctype html>")
        assert "DRIFT" in html and "verified" in html
