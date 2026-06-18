"""Triage / explanation: turn a statistical drift alert into a human narrative + action.

Phase 2 ships a deterministic, knowledge-base-backed explainer (no LLM, near-zero cost). It maps
the alert's attribution — which signature fired, or which feature/family/direction drifted — onto
a suspected cause and a suggested action. The interface takes an optional ``narrator`` callable so
a local-LLM backend (the prompt tier's DSPy/GEPA optimization target, brief §7) can drop in later
to refine the prose without changing the structured output.
"""

from bellwether.triage.explain import TriageExplainer, TriageExplanation

__all__ = ["TriageExplainer", "TriageExplanation"]
