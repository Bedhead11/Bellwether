"""Phase 0 smoke demo: fixture agent -> labeled traces -> redact -> store -> query.

Run with: ``uv run python examples/quickstart.py``

This is the Phase 0 "done when" path made executable: a fixture agent emits traces (some with
known, dial-able faults), PII is redacted at ingest, runs land in the DuckDB store, and we
query them back by label.
"""

from __future__ import annotations

from bellwether.fixtures import FaultSpec, FixtureAgent, generate_dataset
from bellwether.ingest import RunStore, redact_run


def main() -> None:
    agent = FixtureAgent(agent_id="demo", task_class="qa")

    runs = generate_dataset(
        agent,
        n_benign=20,
        fault_specs=[
            FaultSpec("latency_injection", severity=0.7),
            FaultSpec("induced_loop", severity=0.8),
            FaultSpec("cost_blowup", severity=0.9),
        ],
        n_per_fault=5,
    )

    with RunStore(":memory:") as store:
        store.add_many(redact_run(r) for r in runs)

        total = store.count()
        benign = store.query(agent_id="demo", has_fault=False)
        faulted = store.query(agent_id="demo", has_fault=True)

        print(f"stored runs:        {total}")
        print(f"  benign:           {len(benign)}")
        print(f"  faulted (labeled):{len(faulted)}")

        # Ground-truth labels are queryable for the eval harness.
        by_type: dict[str, int] = {}
        for r in faulted:
            assert r.injected_fault is not None
            by_type[r.injected_fault.fault_type] = by_type.get(r.injected_fault.fault_type, 0) + 1
        print("  fault breakdown:  ", dict(sorted(by_type.items())))

        # Spot-check one run's behavioral summary (what feature extractors will consume).
        sample = faulted[0]
        print(
            f"\nsample faulted run '{sample.run_id}': "
            f"{sample.step_count} steps, depth {sample.max_depth()}, "
            f"duration {sample.duration_s:.2f}s, fault={sample.injected_fault.fault_type}"  # type: ignore[union-attr]
        )


if __name__ == "__main__":
    main()
