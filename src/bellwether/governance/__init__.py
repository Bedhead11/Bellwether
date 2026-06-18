"""Governance / audit layer — the differentiator, treated as load-bearing (brief §4).

Every alert, baseline change, and self-improvement decision is recorded in an append-only,
hash-chained, queryable log with a diff, a rationale, and before/after metrics. Hash chaining
makes the log tamper-evident: any edit to a past entry breaks the chain, which ``verify``
detects. This is what lets BELLWETHER answer "what changed, when, and why" for itself.
"""

from bellwether.governance.audit import AuditEvent, AuditLog, EventType

__all__ = ["AuditEvent", "AuditLog", "EventType"]
