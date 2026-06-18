"""Phase 2 demo: the skill tier self-improves and the held-out recall curve climbs.

Run with: ``uv run python examples/self_improve_demo.py``

Each round, the generator mines a drift signature from training incidents, the eval harness
scores the candidate library on a held-out benchmark, and the governance gate promotes it only
if it genuinely improves recall without regressing the false-positive rate or precision. Every
decision is written to a tamper-evident audit log. The held-out recall curve (with CIs) is the
honest "are we still improving?" measurement (design docs 01/02).
"""

from __future__ import annotations

import time

from bellwether.eval import BenchmarkConfig
from bellwether.fixtures import FixtureAgent
from bellwether.governance import AuditLog, EventType
from bellwether.improve import GovernanceGate, SelfImprovementLoop


def main() -> None:
    agent = FixtureAgent(agent_id="support-bot", task_class="qa")
    audit = AuditLog()

    # More benign eval samples per seed tighten the false-positive CI so the strict gate can
    # confirm a candidate stays within budget (per-seed FP resolution is 1/n_eval_benign).
    cfg = BenchmarkConfig(
        n_train=80, n_cal=150, n_eval_benign=250, n_eval_per_fault=10, n_seeds=8, min_samples=25
    )
    # FP budget stays strict; precision tolerance absorbs CI width on the small per-fault sets.
    gate = GovernanceGate(fp_budget=0.02, fp_tol=0.015, precision_tol=0.06, min_recall_gain=0.02)
    loop = SelfImprovementLoop(agent=agent, eval_config=cfg, audit=audit, gate=gate)

    # Mix weak faults (should yield promotions) with an already-solved one (should be rejected
    # for no gain) — demonstrating the gate working in both directions.
    targets = ["cost_blowup", "output_degradation", "tool_misselection", "retry_storm"]

    t0 = time.perf_counter()
    history = loop.run(targets)
    elapsed = time.perf_counter() - t0

    print(history.render())
    print()
    print("audit log (tamper-evident; verify =", audit.verify(), end="):\n")
    for e in audit:
        if e.event_type in (EventType.PROMOTION, EventType.REJECTION):
            verb = "PROMOTE" if e.event_type == EventType.PROMOTION else "REJECT "
            print(f"  {verb} {e.subject:<26} {e.rationale[:70]}")
    print(f"\n  ({len(history.final_library.names)} signatures learned in {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
