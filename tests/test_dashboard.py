"""Tests for the self-contained HTML dashboard."""

from datetime import UTC, datetime

from bellwether.dashboard import render_dashboard
from bellwether.detect.report import Alert, DriftReport, Thresholds
from bellwether.features import FeatureFamily
from bellwether.governance import AuditLog, EventType
from bellwether.improve import Archive
from bellwether.improve.loop import ImprovementHistory, Round
from bellwether.improve.skills import SkillLibrary

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _report(run_id: str, triggered: bool) -> DriftReport:
    return DriftReport(
        run_id=run_id,
        agent_id="a",
        baseline_key=("a", "qa", "fp"),
        thresholds=Thresholds(0.9, 0.9, 0.9),
        observation_scores=(),
        alert=Alert(
            triggered, 1, 0.99 if triggered else 0.0, FeatureFamily.TEMPORAL, "step_latency"
        ),
        family_contributions={FeatureFamily.TEMPORAL: 0.99},
    )


def test_dashboard_renders_self_contained_html() -> None:
    reports = [_report("r1", False), _report("r2", True)]
    audit = AuditLog()
    audit.record_alert("a", "drift on r2", {"score": 0.99})
    rounds = [
        Round(
            0,
            "baseline",
            True,
            (0.70, 0.66, 0.74),
            (0.013, 0.0, 0.02),
            (0.87, 0.8, 0.9),
            (2.0, 1.8, 2.2),
        ),
        Round(
            1,
            "cost_blowup",
            True,
            (0.83, 0.80, 0.86),
            (0.015, 0.0, 0.03),
            (0.90, 0.85, 0.95),
            (1.9, 1.7, 2.1),
        ),
    ]
    history = ImprovementHistory(rounds, SkillLibrary(), Archive())

    html = render_dashboard(title="BELLWETHER", reports=reports, improvement=history, audit=audit)

    # Self-contained: a full HTML doc with inline style and SVG, no external refs.
    assert html.startswith("<!doctype html>")
    assert "<style>" in html and "http://" not in html and "src=" not in html
    assert "<svg" in html  # charts are inline SVG
    assert "DRIFT" in html and "cost_blowup" in html
    assert "verified" in html  # audit integrity shown


def test_dashboard_handles_missing_sections() -> None:
    # Only an audit log; other sections omitted gracefully.
    audit = AuditLog()
    audit.append(EventType.NOTE, "x", "hello")
    html = render_dashboard(audit=audit)
    assert "Governance" in html
    assert "Drift timeline" not in html


def test_dashboard_writes_openable_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "dash.html"
    out.write_text(render_dashboard(reports=[_report("r1", True)]), encoding="utf-8")
    assert out.read_text(encoding="utf-8").rstrip().endswith("</html>")
