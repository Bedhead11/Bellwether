"""Tests for the DriftMonitor query facade (the MCP-tool surface)."""

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run
from bellwether.ingest import RedactionConfig
from bellwether.monitor import DriftMonitor


def _trained_monitor(agent: FixtureAgent) -> DriftMonitor:
    mgr = BaselineManager()
    for s in range(120):
        mgr.learn(agent.clean_run(seed=s))
    baseline = mgr.baseline_for(agent.clean_run(seed=0))
    scorer = DriftScorer(ScoringConfig(min_samples=30))
    thr = calibrate_threshold(
        scorer, baseline, [agent.clean_run(seed=s) for s in range(120, 270)], target_fp_rate=0.02
    )
    mon = DriftMonitor(manager=mgr, scorer=scorer, redaction=RedactionConfig())
    mon.set_thresholds(agent.clean_run(seed=0).baseline_key, thr)
    return mon


def test_ingest_returns_none_without_baseline() -> None:
    mon = DriftMonitor(manager=BaselineManager(), scorer=DriftScorer())
    assert mon.ingest(FixtureAgent().clean_run(seed=1)) is None


def test_ingest_scores_and_records_alert() -> None:
    agent = FixtureAgent()
    mon = _trained_monitor(agent)

    benign = mon.ingest(agent.clean_run(seed=9999))
    assert benign is not None and not benign.alert.triggered

    faulted = mon.ingest(
        faulted_run(agent, 4242, FaultSpec("latency_injection", 0.9, onset_step=1))
    )
    assert faulted is not None and faulted.alert.triggered

    # The alert was recorded to the tamper-evident audit log.
    assert mon.audit.verify()
    alerts = mon.recent_alerts()
    assert len(alerts) == 1
    assert alerts[0]["agent_id"] == agent.agent_id


def test_query_surface_returns_serializable_dicts() -> None:
    agent = FixtureAgent()
    mon = _trained_monitor(agent)
    mon.ingest(faulted_run(agent, 4242, FaultSpec("tool_misselection", 0.9, onset_step=1)))

    status = mon.drift_status(agent.agent_id)
    assert status["status"] == "drift"
    assert "attribution" in status

    overall = mon.status()
    assert overall["baselines"] == 1
    assert overall["audit_verified"] is True
    assert agent.agent_id in overall["agents"]  # type: ignore[operator]

    baselines = mon.baselines()
    assert baselines[0]["observations"] > 0


def test_unknown_agent_status() -> None:
    mon = DriftMonitor(manager=BaselineManager(), scorer=DriftScorer())
    assert mon.drift_status("nobody")["status"] == "unknown"


def test_mcp_server_requires_fastmcp() -> None:
    # Without the optional `mcp` extra installed, building the server fails with a clear message.
    import pytest

    from bellwether.mcp_server import build_mcp_server

    mon = DriftMonitor(manager=BaselineManager(), scorer=DriftScorer())
    with pytest.raises(RuntimeError, match="FastMCP is not installed"):
        build_mcp_server(mon)
