"""Append-only, hash-chained audit log.

Each event stores the hash of the previous event; its own hash covers its content plus that
previous hash. The result is a tamper-evident chain: altering or removing any past event breaks
every subsequent hash, which :meth:`AuditLog.verify` catches. The log is append-only by
construction (no update/delete API) and optionally persisted as JSON Lines, the natural
append-only file format.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

_GENESIS = "0" * 64


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class EventType(StrEnum):
    """The kinds of governed events. Extend as new self-modifying behaviors are added."""

    ALERT = "alert"
    BASELINE_UPDATE = "baseline_update"
    BASELINE_ACCEPTED = "baseline_accepted"  # accept-new-normal (design doc 03)
    PROMOTION = "promotion"  # a self-improvement change accepted by the gate
    REJECTION = "rejection"  # a self-improvement change rejected by the gate
    SIGNATURE_ADDED = "signature_added"  # skill-tier: a new drift signature
    PLATEAU = "plateau"  # anti-plateau: improvement stalled
    ROTATION = "rotation"  # held-out benchmark rotated
    NOTE = "note"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One immutable governed event."""

    seq: int
    ts: str
    event_type: EventType
    subject: str
    rationale: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    def content_hash(self) -> str:
        """Recompute the hash from this event's content + prev_hash (for verification)."""
        body = {
            "seq": self.seq,
            "ts": self.ts,
            "event_type": str(self.event_type),
            "subject": self.subject,
            "rationale": self.rationale,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class AuditLog:
    """An append-only, hash-chained event log. Optionally persisted as JSON Lines."""

    path: str | Path | None = None
    _events: list[AuditEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path is not None and Path(self.path).exists():
            self._load()

    # --- append (the only mutation) -----------------------------------------------------

    def append(
        self,
        event_type: EventType,
        subject: str,
        rationale: str,
        payload: dict[str, Any] | None = None,
        *,
        ts: str | None = None,
    ) -> AuditEvent:
        seq = len(self._events)
        prev_hash = self._events[-1].hash if self._events else _GENESIS
        draft = AuditEvent(
            seq=seq,
            ts=ts or _utcnow_iso(),
            event_type=event_type,
            subject=subject,
            rationale=rationale,
            payload=payload or {},
            prev_hash=prev_hash,
            hash="",
        )
        event = AuditEvent(**{**asdict(draft), "hash": draft.content_hash()})
        self._events.append(event)
        if self.path is not None:
            self._persist(event)
        return event

    # convenience wrappers for common events ---------------------------------------------

    def record_alert(self, subject: str, rationale: str, payload: dict[str, Any]) -> AuditEvent:
        return self.append(EventType.ALERT, subject, rationale, payload)

    def record_promotion(
        self, subject: str, rationale: str, diff: dict[str, Any], metric_deltas: dict[str, Any]
    ) -> AuditEvent:
        return self.append(
            EventType.PROMOTION, subject, rationale, {"diff": diff, "metric_deltas": metric_deltas}
        )

    def record_rejection(
        self, subject: str, rationale: str, diff: dict[str, Any], metric_deltas: dict[str, Any]
    ) -> AuditEvent:
        return self.append(
            EventType.REJECTION, subject, rationale, {"diff": diff, "metric_deltas": metric_deltas}
        )

    # --- reads --------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[AuditEvent]:
        return iter(self._events)

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def query(
        self,
        *,
        event_type: EventType | None = None,
        subject: str | None = None,
    ) -> list[AuditEvent]:
        out = self._events
        if event_type is not None:
            out = [e for e in out if e.event_type == event_type]
        if subject is not None:
            out = [e for e in out if e.subject == subject]
        return list(out)

    @property
    def head_hash(self) -> str:
        return self._events[-1].hash if self._events else _GENESIS

    # --- integrity ----------------------------------------------------------------------

    def verify(self) -> bool:
        """Return True iff the chain is intact (no event altered, removed, or reordered)."""
        prev = _GENESIS
        for i, e in enumerate(self._events):
            if e.seq != i or e.prev_hash != prev or e.content_hash() != e.hash:
                return False
            prev = e.hash
        return True

    # --- persistence --------------------------------------------------------------------

    def _persist(self, event: AuditEvent) -> None:
        assert self.path is not None
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(event.to_json() + "\n")

    def _load(self) -> None:
        assert self.path is not None
        for line in Path(self.path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            d["event_type"] = EventType(d["event_type"])
            self._events.append(AuditEvent(**d))
