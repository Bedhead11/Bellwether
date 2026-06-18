"""L0 unit tests for the governance / audit layer."""

import dataclasses
from pathlib import Path

from bellwether.governance import AuditLog, EventType


def test_append_assigns_sequence_and_chains_hashes() -> None:
    log = AuditLog()
    e0 = log.append(EventType.NOTE, "subject-a", "first")
    e1 = log.append(EventType.NOTE, "subject-b", "second")
    assert e0.seq == 0 and e1.seq == 1
    assert e1.prev_hash == e0.hash
    assert e0.hash != e1.hash
    assert len(log) == 2


def test_verify_passes_on_intact_chain() -> None:
    log = AuditLog()
    for i in range(5):
        log.append(EventType.NOTE, f"s{i}", "r")
    assert log.verify() is True


def test_verify_detects_tampering() -> None:
    log = AuditLog()
    for i in range(5):
        log.append(EventType.NOTE, f"s{i}", "r")
    # Tamper with a past event's rationale (simulating an edit that bypasses append()).
    log._events[2] = dataclasses.replace(log._events[2], rationale="MUTATED")
    assert log.verify() is False


def test_query_by_type_and_subject() -> None:
    log = AuditLog()
    log.record_alert("agent-x", "drift", {"score": 0.99})
    log.append(EventType.NOTE, "agent-x", "note")
    log.record_promotion("ensemble", "better", {"add": "sig"}, {"recall": "+0.1"})

    assert len(log.query(event_type=EventType.ALERT)) == 1
    assert len(log.query(subject="agent-x")) == 2
    assert len(log.query(event_type=EventType.PROMOTION)) == 1


def test_promotion_and_rejection_payloads() -> None:
    log = AuditLog()
    p = log.record_promotion("ens", "improves recall", {"weight": "0.5->0.6"}, {"recall": "+0.05"})
    r = log.record_rejection("ens", "regresses fp", {"thr": "0.9->0.8"}, {"fp_rate": "+0.03"})
    assert p.payload["diff"] == {"weight": "0.5->0.6"}
    assert r.payload["metric_deltas"] == {"fp_rate": "+0.03"}


def test_persistence_roundtrip_and_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)
    log.append(EventType.NOTE, "a", "one", {"k": 1})
    log.record_alert("b", "drift", {"score": 0.98})

    reloaded = AuditLog(path=path)
    assert len(reloaded) == 2
    assert reloaded.verify() is True
    assert reloaded.head_hash == log.head_hash
    assert reloaded.query(event_type=EventType.ALERT)[0].payload["score"] == 0.98


def test_append_only_persistence_is_additive(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)
    log.append(EventType.NOTE, "a", "one")
    # New session appends; prior lines are never rewritten.
    log2 = AuditLog(path=path)
    log2.append(EventType.NOTE, "b", "two")
    assert len(path.read_text().splitlines()) == 2
    assert AuditLog(path=path).verify() is True
