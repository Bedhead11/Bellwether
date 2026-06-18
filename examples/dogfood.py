"""Dogfooding: BELLWETHER instruments its own internal operation with its own SDK (brief §11).

Run with: ``uv run python examples/dogfood.py``

The tool that watches agents is itself built like a well-governed agent system. Here BELLWETHER's
own detection work — learning a baseline, then scoring a batch of runs — is captured as a canonical
``AgentRun`` via the same ``@watch`` SDK and the same trace format it monitors for everyone else.
That run could be fed straight back into BELLWETHER (turtles all the way down).
"""

from __future__ import annotations

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.fixtures import FixtureAgent
from bellwether.ingest import RunStore
from bellwether.sdk import Bellwether


def instrument_self(store: RunStore | None = None):  # type: ignore[no-untyped-def]
    """Run a small BELLWETHER detection session, recording it as an AgentRun via the SDK."""
    target = FixtureAgent(agent_id="watched-agent")
    mgr = BaselineManager()
    for s in range(60):
        mgr.learn(target.clean_run(seed=s))
    baseline = mgr.baseline_for(target.clean_run(seed=0))
    assert baseline is not None
    scorer = DriftScorer(ScoringConfig(min_samples=20))

    # BELLWETHER monitoring itself: its own work is a watched, canonical AgentRun.
    bw = Bellwether(agent_id="bellwether-internal", task_class="detection", store=store)
    with bw.watch() as handle:
        with bw.agent("calibrate"):
            thr = calibrate_threshold(
                scorer,
                baseline,
                [target.clean_run(seed=s) for s in range(60, 110)],
                target_fp_rate=0.02,
            )
        with bw.agent("score-batch"):
            for s in range(200, 206):
                with bw.tool("evaluate", args={"seed": s}) as call:
                    report = scorer.evaluate(target.clean_run(seed=s), baseline, thr)
                    call.set_attribute("max_drift", round(report.max_drift, 3))
    return handle.run


def main() -> None:
    with RunStore() as store:
        run = instrument_self(store)
        assert run is not None
        print(f"BELLWETHER instrumented itself as run '{run.run_id}'")
        print(f"  agent_id:   {run.agent_id}  task_class: {run.task_class}")
        print(
            f"  spans:      {run.step_count}  depth: {run.max_depth()}  duration: {run.duration_s:.3f}s"
        )
        print(f"  persisted:  {store.count()} run(s) in the store")
        print("  → this trace is itself a canonical AgentRun, monitorable by BELLWETHER.")


if __name__ == "__main__":
    main()
