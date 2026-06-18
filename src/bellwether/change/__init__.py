"""Drift vs. intended-change disambiguation (design doc 03).

A deliberate prompt/model/config change shifts behavior and must NOT be reported as drift, or
every deploy fires every alert and the tool gets muted. This package implements the decision:

- a **config fingerprint** keys each baseline lineage, so a *declared* change (new fingerprint or
  an explicit deploy marker) opens a fresh lineage in a **quarantine** window instead of alerting;
- a **change-point detector** (Page-Hinkley) over the per-run drift stream separates a
  *persistent* regime shift from a transient one-off — an unexplained persistent shift (no
  fingerprint change) is genuine **drift** and fires;
- a human-confirmable **accept-new-normal** promotes a quarantined lineage to active, audited.
"""

from bellwether.change.aware import ChangeAwareMonitor, ChangeVerdict, VerdictKind
from bellwether.change.page_hinkley import PageHinkley

__all__ = ["ChangeAwareMonitor", "ChangeVerdict", "VerdictKind", "PageHinkley"]
