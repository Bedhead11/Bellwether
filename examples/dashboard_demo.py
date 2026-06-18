"""Phase 3 demo: generate the self-contained HTML dashboard.

Run with: ``uv run python examples/dashboard_demo.py``  →  writes ``bellwether_dashboard.html``

Trains a baseline, scores a stream of benign + faulted runs (the drift timeline), runs a short
skill-tier self-improvement loop (the improvement curve + audit log), and renders everything to a
single dependency-free HTML file that opens in any browser.
"""

from __future__ import annotations

from pathlib import Path

from bellwether.baseline import BaselineManager
from bellwether.dashboard import render_dashboard
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.eval import BenchmarkConfig
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run
from bellwether.governance import AuditLog
from bellwether.improve import GovernanceGate, SelfImprovementLoop

OUT = Path("bellwether_dashboard.html")


def _drift_timeline_reports(agent: FixtureAgent):  # type: ignore[no-untyped-def]
    mgr = BaselineManager()
    for s in range(120):
        mgr.learn(agent.clean_run(seed=s))
    baseline = mgr.baseline_for(agent.clean_run(seed=0))
    assert baseline is not None
    scorer = DriftScorer(ScoringConfig(min_samples=30))
    thr = calibrate_threshold(
        scorer, baseline, [agent.clean_run(seed=s) for s in range(120, 270)], target_fp_rate=0.02
    )

    reports = []
    # A run of healthy traffic, then injected faults of different kinds.
    for s in range(900, 906):
        reports.append(scorer.evaluate(agent.clean_run(seed=s), baseline, thr))
    for i, (ft, sev) in enumerate(
        [
            ("latency_injection", 0.8),
            ("tool_misselection", 0.9),
            ("induced_loop", 0.8),
            ("retry_storm", 0.8),
            ("cost_blowup", 0.9),
        ]
    ):
        run = faulted_run(agent, 950 + i, FaultSpec(ft, sev, onset_step=1))
        reports.append(scorer.evaluate(run, baseline, thr))
    return reports


def main() -> None:
    agent = FixtureAgent(agent_id="support-bot", task_class="qa")

    reports = _drift_timeline_reports(agent)

    audit = AuditLog()
    cfg = BenchmarkConfig(
        n_train=80, n_cal=150, n_eval_benign=250, n_eval_per_fault=10, n_seeds=8, min_samples=25
    )
    gate = GovernanceGate(fp_budget=0.02, fp_tol=0.015, precision_tol=0.06, min_recall_gain=0.02)
    loop = SelfImprovementLoop(agent=agent, eval_config=cfg, audit=audit, gate=gate)
    history = loop.run(["cost_blowup", "output_degradation", "retry_storm"])

    html = render_dashboard(title="BELLWETHER", reports=reports, improvement=history, audit=audit)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(html) // 1024} KB) — open it in a browser")
    print(
        f"  drift timeline: {len(reports)} runs; "
        f"self-improvement: {len(history.final_library.names)} signatures; "
        f"audit: {len(audit)} events (verify={audit.verify()})"
    )


if __name__ == "__main__":
    main()
