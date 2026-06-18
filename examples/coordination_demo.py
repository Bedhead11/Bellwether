"""Phase 4 demo: detect multi-agent COORDINATION drift (the named unsolved problem).

Run with: ``uv run python examples/coordination_demo.py``

A multi-agent system (planner → researcher → writer → reviewer) is monitored. Each agent's own
per-step behavior (latency, tokens) stays locally normal, but the *coordination* degrades —
ping-pong between two agents, one agent collapsing the division of labor, or a handoff storm.
BELLWETHER's coordination feature family + skill-tier signatures catch and attribute it.
"""

from __future__ import annotations

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.fixtures.multiagent import (
    COORDINATION_FAULTS,
    CoordinationFaultSpec,
    MultiAgentSystem,
    coordination_faulted_run,
)
from bellwether.improve import DriftSignature, SkillLibrary
from bellwether.triage import TriageExplainer


def main() -> None:
    system = MultiAgentSystem(agent_id="research-crew", task_class="report")

    mgr = BaselineManager()
    for s in range(150):
        mgr.learn(system.clean_run(seed=s))
    baseline = mgr.baseline_for(system.clean_run(seed=0))
    assert baseline is not None

    library = SkillLibrary(
        [
            DriftSignature("ping_pong", (("pingpong_rate", "run", 1),)),
            DriftSignature("role_collapse", (("role_imbalance", "run", 1),)),
            DriftSignature("handoff_storm", (("handoff_rate", "run", 1),)),
        ]
    )
    scorer = DriftScorer(ScoringConfig(min_samples=30), library=library)
    thr = calibrate_threshold(
        scorer, baseline, [system.clean_run(seed=s) for s in range(150, 320)], target_fp_rate=0.02
    )
    explainer = TriageExplainer()

    print("Healthy multi-agent coordination:")
    print(" ", scorer.evaluate(system.clean_run(seed=9999), baseline, thr).summary())

    print("\nInjected coordination faults (each agent still looks locally normal):")
    for fault in COORDINATION_FAULTS:
        run = coordination_faulted_run(system, 4242, CoordinationFaultSpec(fault, 0.9))
        report = scorer.evaluate(run, baseline, thr)
        print(" ", report.summary())
        exp = explainer.explain(report)
        if exp is not None and report.alert.signature:
            print(f"      → {report.alert.signature}: {exp.suggested_action}")


if __name__ == "__main__":
    main()
