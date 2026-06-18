"""Dogfooding test (brief §11): BELLWETHER instruments its own operation with its own SDK.

The tool that watches agents is built like a well-governed agent system: its own internal work is
captured as a canonical ``AgentRun`` via the same SDK and trace format it monitors for everyone
else — which means that run can be fed straight back into BELLWETHER.
"""

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.fixtures import FixtureAgent
from bellwether.ingest import RunStore
from bellwether.schema import SpanKind
from bellwether.sdk import Bellwether


def test_bellwether_instruments_itself_as_canonical_run() -> None:
    target = FixtureAgent(agent_id="watched-agent")
    mgr = BaselineManager()
    for s in range(60):
        mgr.learn(target.clean_run(seed=s))
    baseline = mgr.baseline_for(target.clean_run(seed=0))
    assert baseline is not None
    scorer = DriftScorer(ScoringConfig(min_samples=20))

    with RunStore() as store:
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

        run = handle.run
        assert run is not None
        assert run.agent_id == "bellwether-internal" and run.task_class == "detection"

        # Captured with structure: sub-agent phases and a tool span per scored run.
        sub_agents = {s.name for s in run.spans if s.kind == SpanKind.SUB_AGENT}
        assert "agent.calibrate" in sub_agents and "agent.score-batch" in sub_agents
        tool_spans = [s for s in run.spans if s.kind == SpanKind.TOOL]
        assert len(tool_spans) == 6
        assert all(s.parent_span_id is not None for s in tool_spans)

        # It is a schema-valid, persistable AgentRun — re-ingestable by BELLWETHER itself.
        assert store.get(run.run_id) is not None
