"""Self-improvement engine (brief §7): generator → evaluator → governance gate → archive.

Phase 2 implements the **skill tier** (a growing library of drift signatures) and the **prompt
tier** explainer, under the multi-objective, anti-plateau search from design doc 01. The
**topology tier** (evolving detector ensembles) is Phase 3.

A drift *signature* is a focused, direction-filtered detector for a known drift type. Because it
attends to only a couple of features, its multiplicity penalty is small, so it scores its
specific pattern more sharply than the generic detector — and it is calibrated to its own tiny
false-positive share, so adding signatures improves sensitivity without inflating false alarms.
"""

from bellwether.improve.skills import DriftSignature, SkillLibrary

__all__ = ["DriftSignature", "SkillLibrary"]
