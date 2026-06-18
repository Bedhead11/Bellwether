"""Phase 1 demo: learn a baseline, inject faults, watch BELLWETHER alert with attribution.

Run with: ``uv run python examples/detect_demo.py``

This is the Phase 1 "done when" path: wrap an agent -> learn a baseline -> inject a fault ->
it alerts with correct attribution and a lead-time.
"""

from __future__ import annotations

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run


def main() -> None:
    agent = FixtureAgent(agent_id="support-bot", task_class="qa")

    # 1) Learn a behavioral baseline from benign runs.
    mgr = BaselineManager()
    for s in range(120):
        mgr.learn(agent.clean_run(seed=s))
    baseline = mgr.baseline_for(agent.clean_run(seed=0))
    assert baseline is not None

    # 2) Calibrate alert thresholds to a 2% false-positive budget on a benign hold-out.
    scorer = DriftScorer(ScoringConfig(min_samples=30))
    thresholds = calibrate_threshold(
        scorer, baseline, [agent.clean_run(seed=s) for s in range(120, 270)], target_fp_rate=0.02
    )
    print("learned baseline + calibrated thresholds:")
    print(
        f"  critical={thresholds.critical:.4f}  step={thresholds.step:.4f}  "
        f"run={thresholds.run:.4f}\n"
    )

    # 3) A healthy run should stay quiet.
    healthy = agent.clean_run(seed=9999)
    print(scorer.evaluate(healthy, baseline, thresholds).summary())

    # 4) Inject each fault type and watch BELLWETHER flag it with attribution.
    faults = [
        ("latency_injection", 0.8),
        ("induced_loop", 0.8),
        ("tool_misselection", 0.9),
        ("cost_blowup", 0.9),
        ("retry_storm", 0.8),
        ("output_degradation", 0.9),
    ]
    for fault_type, severity in faults:
        run = faulted_run(agent, seed=4242, spec=FaultSpec(fault_type, severity, onset_step=1))
        report = scorer.evaluate(run, baseline, thresholds)
        print(report.summary())
        if report.alert.triggered:
            top = ", ".join(
                f"{fam.value}={score:.2f}"
                for fam, score in list(report.family_contributions.items())[:3]
            )
            print(f"          contributing families: {top}")


if __name__ == "__main__":
    main()
