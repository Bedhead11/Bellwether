"""Deterministic triage explainer with a pluggable narrator hook.

The explainer reads a :class:`DriftReport`'s attribution and produces a structured
:class:`TriageExplanation`. A small knowledge base maps the drifted family + direction (and any
fired signature) onto a suspected cause and a suggested action. Confidence is graded from the
drift score. A ``narrator`` callable can be supplied to rewrite the headline (e.g. via a local
LLM); the structured fields stay deterministic and auditable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from bellwether.detect import DriftReport
from bellwether.features import FeatureFamily

# Knowledge base keyed by (family, direction): direction +1 = increased, -1 = decreased, 0 = any.
# Each entry is (suspected_cause, suggested_action).
_KB: dict[tuple[FeatureFamily, int], tuple[str, str]] = {
    (FeatureFamily.TEMPORAL, 1): (
        "step latency increased — slower model or tool responses",
        "check provider latency/quotas, tool timeouts, and retry behavior",
    ),
    (FeatureFamily.ECONOMIC, 1): (
        "token/cost per step increased — likely prompt bloat or context accumulation",
        "inspect prompt construction and context-window management; cap retrieval size",
    ),
    (FeatureFamily.CONTEXT, 1): (
        "prompt/input size grew — context or retrieved-document growth",
        "bound context growth; check RAG result sizes and memory accumulation",
    ),
    (FeatureFamily.SEMANTIC, -1): (
        "output got shorter — possible truncation, refusal, or quality degradation",
        "check max_tokens, refusal/hedge rate, and output-format adherence",
    ),
    (FeatureFamily.STRUCTURAL, 1): (
        "structural change — looping or retry/error escalation",
        "add a loop/max-iteration guard; inspect the failing dependency or tool",
    ),
    (FeatureFamily.TOOL, 0): (
        "off-profile tool selection — a rare or newly-seen tool was called",
        "verify tool availability/routing and recent tool-schema or prompt changes",
    ),
}

# Direction-agnostic fallbacks (used when the drift direction can't be inferred, e.g. a
# run-summary-only alert): each family's canonical drift direction.
_FAMILY_DEFAULT: dict[FeatureFamily, tuple[str, str]] = {
    FeatureFamily.TEMPORAL: _KB[(FeatureFamily.TEMPORAL, 1)],
    FeatureFamily.ECONOMIC: _KB[(FeatureFamily.ECONOMIC, 1)],
    FeatureFamily.CONTEXT: _KB[(FeatureFamily.CONTEXT, 1)],
    FeatureFamily.SEMANTIC: _KB[(FeatureFamily.SEMANTIC, -1)],
    FeatureFamily.STRUCTURAL: _KB[(FeatureFamily.STRUCTURAL, 1)],
    FeatureFamily.TOOL: _KB[(FeatureFamily.TOOL, 0)],
}

# Signature-name overrides (skill tier): a recognized drift type gives a sharper cause/action.
_SIGNATURE_KB: dict[str, tuple[str, str]] = {
    "cost_blowup": (
        "recognized cost-blowup drift: input tokens and cost rose together on LLM steps",
        "audit prompt/context growth and retrieval size; set a per-step token budget",
    ),
    "output_degradation": (
        "recognized output-degradation drift: responses shortened on LLM steps",
        "check max_tokens, refusals, and format adherence; verify the model version",
    ),
    "tool_misselection": (
        "recognized tool-misselection drift: an off-profile/deprecated tool was used",
        "verify tool routing and that recent tool/prompt changes were intended",
    ),
    "induced_loop": (
        "recognized loop drift: identical tool calls repeated within the run",
        "add loop detection and a max-iteration guard to the agent",
    ),
    "retry_storm": (
        "recognized retry-storm drift: escalating errors/retries",
        "inspect the failing dependency; add backoff and a retry cap",
    ),
    "latency_injection": (
        "recognized latency drift: per-step latency rose",
        "check provider/tool latency and timeouts",
    ),
}

_DEFAULT = (
    "behavioral drift detected; the dominant signal is unusual versus the learned baseline",
    "review the contributing features below against recent config or dependency changes",
)


@dataclass(frozen=True, slots=True)
class TriageExplanation:
    headline: str
    suspected_cause: str
    suggested_action: str
    confidence: str  # high | medium | low
    evidence: tuple[str, ...]

    def render(self) -> str:
        lines = [
            f"{self.headline}",
            f"  suspected cause:  {self.suspected_cause}",
            f"  suggested action: {self.suggested_action}",
            f"  confidence:       {self.confidence}",
        ]
        if self.evidence:
            lines.append("  evidence:")
            lines.extend(f"    - {e}" for e in self.evidence)
        return "\n".join(lines)


def _confidence(score: float) -> str:
    if score >= 0.98:
        return "high"
    if score >= 0.9:
        return "medium"
    return "low"


@dataclass
class TriageExplainer:
    """Produces a triage explanation for a drift report. ``narrator`` may refine the headline."""

    narrator: Callable[[str, DriftReport], str] | None = None

    def explain(self, report: DriftReport) -> TriageExplanation | None:
        alert = report.alert
        if not alert.triggered:
            return None

        # Prefer a recognized signature; else fall back to family + direction.
        cause, action = _DEFAULT
        if alert.signature and alert.signature in _SIGNATURE_KB:
            cause, action = _SIGNATURE_KB[alert.signature]
        elif alert.primary_family is not None:
            direction = 0
            primary = next(
                (
                    s
                    for o in report.observation_scores
                    for s in o.sub_scores
                    if s.feature == alert.primary_feature
                ),
                None,
            )
            if primary is not None and primary.direction != 0:
                direction = 1 if primary.direction > 0 else -1
            cause, action = (
                _KB.get((alert.primary_family, direction))
                or _FAMILY_DEFAULT.get(alert.primary_family)
                or _DEFAULT
            )

        where = f"step {alert.step_index}" if alert.step_index is not None else "run end"
        what = (
            f"signature '{alert.signature}'"
            if alert.signature
            else f"{alert.primary_family}/{alert.primary_feature}"
        )
        headline = (
            f"Drift on agent '{report.agent_id}': {what} at {where} (score {alert.drift_score:.3f})"
        )
        if self.narrator is not None:
            headline = self.narrator(headline, report)

        evidence = tuple(
            f"{fam.value}: {score:.2f}"
            for fam, score in list(report.family_contributions.items())[:4]
        )
        return TriageExplanation(
            headline=headline,
            suspected_cause=cause,
            suggested_action=action,
            confidence=_confidence(alert.drift_score),
            evidence=evidence,
        )
