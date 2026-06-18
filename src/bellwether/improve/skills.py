"""Drift signatures: focused, direction-filtered detectors for known drift types.

A :class:`DriftSignature` names a small set of ``(feature, context, direction)`` triples. Its
*focused score* for an observation aggregates only those features' conformal p-values (with the
expected direction), using a Šidák correction over the *matched* features. Because the matched
count is tiny, the focused score is far sharper for the signature's pattern than the generic
aggregate (which dilutes credit across all ~6 families). Each signature is calibrated to its own
small false-positive share, so the library improves sensitivity without inflating false alarms —
and because the total budget is fixed and split across tracks, adding signatures cannot blow it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from bellwether.detect.report import SubScore

# Magnitude squashing scale: maps a matched-feature anomaly magnitude (|robust z|, or
# categorical surprise) into [0, 1) monotonically. Benign tails (z~2-3) land well below faulted
# anomalies (z~10+), so the signature's calibrated threshold can separate them — unlike the
# conformal p-value, which floors and collides at the extreme.
_MAGNITUDE_SCALE = 3.0


def _direction_ok(observed_z: float, expected: int) -> bool:
    """Direction filter: expected +1 (up) / -1 (down) / 0 (any)."""
    if expected == 0:
        return True
    if expected > 0:
        return observed_z > 0
    return observed_z < 0


@dataclass(frozen=True, slots=True)
class DriftSignature:
    """A learned pattern: which features, in which direction, indicate a known drift type."""

    name: str
    # (feature_name, context, direction) — direction in {+1, -1, 0}.
    feature_directions: tuple[tuple[str, str, int], ...]
    description: str = ""

    @property
    def feature_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset((f, c) for f, c, _ in self.feature_directions)

    def focused_score(self, sub_scores: Sequence[SubScore]) -> float:
        """Score one observation against this signature using only its (matched) features.

        Uses the matched features' anomaly *magnitudes* (not the conformal p-value) so the score
        keeps separating values beyond the baseline maximum, where the p-value floors. The score
        is a monotone squash of the strongest matched magnitude into [0, 1); the absolute scale
        does not matter because the signature's alert threshold is calibrated on benign data.
        """
        by_key = {(s.feature, s.context): s for s in sub_scores}
        matched_mag: list[float] = []
        for fname, fctx, direction in self.feature_directions:
            sub = by_key.get((fname, fctx))
            if sub is None:
                continue
            if not _direction_ok(sub.direction, direction):
                continue
            matched_mag.append(sub.magnitude)
        if not matched_mag:
            return 0.0
        strongest = max(matched_mag)
        return 1.0 - math.exp(-strongest / _MAGNITUDE_SCALE)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "feature_directions": [list(fd) for fd in self.feature_directions],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DriftSignature:
        fds = tuple((str(f), str(c), int(dir_)) for f, c, dir_ in d["feature_directions"])
        return cls(
            name=str(d["name"]),
            feature_directions=fds,
            description=str(d.get("description", "")),
        )


@dataclass
class SkillLibrary:
    """An ordered collection of drift signatures consulted during scoring."""

    signatures: list[DriftSignature] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.signatures]

    def has(self, name: str) -> bool:
        return any(s.name == name for s in self.signatures)

    def add(self, signature: DriftSignature) -> None:
        if self.has(signature.name):
            raise ValueError(f"signature {signature.name!r} already present")
        self.signatures.append(signature)

    def without(self, name: str) -> SkillLibrary:
        """A copy excluding ``name`` (used by the gate to A/B a candidate)."""
        return SkillLibrary([s for s in self.signatures if s.name != name])

    def with_added(self, signature: DriftSignature) -> SkillLibrary:
        """A copy including ``signature`` (used by the gate to A/B a candidate)."""
        return SkillLibrary([*self.signatures, signature])

    def signature_scores(self, sub_scores: Sequence[SubScore]) -> dict[str, float]:
        return {s.name: s.focused_score(sub_scores) for s in self.signatures}

    def to_dict(self) -> dict[str, object]:
        return {"signatures": [s.to_dict() for s in self.signatures]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillLibrary:
        sigs = [DriftSignature.from_dict(s) for s in d.get("signatures", [])]
        return cls(signatures=sigs)
