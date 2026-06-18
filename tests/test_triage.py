"""Tests for the deterministic triage explainer."""

from bellwether.detect.report import Alert, DriftReport, Thresholds
from bellwether.features import FeatureFamily
from bellwether.triage import TriageExplainer


def _report(
    *, triggered: bool, signature=None, family=None, feature=None, score=0.99
) -> DriftReport:  # type: ignore[no-untyped-def]
    return DriftReport(
        run_id="r",
        agent_id="support-bot",
        baseline_key=("support-bot", "qa", "fp"),
        thresholds=Thresholds(0.9, 0.9, 0.9),
        observation_scores=(),
        alert=Alert(triggered, 2, score, family, feature, signature=signature),
        family_contributions={FeatureFamily.ECONOMIC: 1.0, FeatureFamily.CONTEXT: 0.7},
    )


def test_no_explanation_when_not_triggered() -> None:
    assert TriageExplainer().explain(_report(triggered=False)) is None


def test_signature_drives_cause_and_action() -> None:
    exp = TriageExplainer().explain(_report(triggered=True, signature="cost_blowup"))
    assert exp is not None
    assert "cost-blowup" in exp.suspected_cause
    assert "token budget" in exp.suggested_action
    assert exp.confidence == "high"
    assert any("economic" in e for e in exp.evidence)


def test_family_fallback_when_no_signature() -> None:
    exp = TriageExplainer().explain(
        _report(triggered=True, family=FeatureFamily.TEMPORAL, feature="step_latency")
    )
    assert exp is not None
    assert "latency" in exp.suspected_cause.lower()


def test_confidence_grading() -> None:
    low = TriageExplainer().explain(_report(triggered=True, signature="cost_blowup", score=0.85))
    assert low is not None and low.confidence == "low"


def test_narrator_hook_rewrites_headline() -> None:
    explainer = TriageExplainer(narrator=lambda headline, report: f"[LLM] {headline}")
    exp = explainer.explain(_report(triggered=True, signature="cost_blowup"))
    assert exp is not None and exp.headline.startswith("[LLM] ")


def test_render_is_readable() -> None:
    exp = TriageExplainer().explain(_report(triggered=True, signature="tool_misselection"))
    assert exp is not None
    text = exp.render()
    assert "suspected cause" in text and "suggested action" in text
