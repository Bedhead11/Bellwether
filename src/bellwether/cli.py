"""``bellwether`` command-line interface.

A thin entrypoint so ``pip install bellwether`` gives value in one command:

    bellwether benchmark [--quick]   # run the drift benchmark with confidence intervals
    bellwether demo                  # learn a baseline, inject faults, show alerts + triage
    bellwether dashboard [-o FILE]   # write a self-contained HTML dashboard
    bellwether eval-gate             # CI regression gate: fail if quality drops below thresholds
    bellwether version
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from bellwether.eval import BenchmarkConfig, run_benchmark

    cfg = (
        BenchmarkConfig(
            n_train=50, n_cal=30, n_eval_benign=40, n_eval_per_fault=8, n_seeds=5, min_samples=20
        )
        if args.quick
        else BenchmarkConfig()
    )
    print(run_benchmark(config=cfg).render())
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    from bellwether.baseline import BaselineManager
    from bellwether.detect import DriftScorer, ScoringConfig
    from bellwether.detect.engine import calibrate_threshold
    from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run
    from bellwether.triage import TriageExplainer

    agent = FixtureAgent(agent_id="support-bot", task_class="qa")
    mgr = BaselineManager()
    for s in range(120):
        mgr.learn(agent.clean_run(seed=s))
    baseline = mgr.baseline_for(agent.clean_run(seed=0))
    assert baseline is not None
    scorer = DriftScorer(ScoringConfig(min_samples=30))
    thr = calibrate_threshold(
        scorer, baseline, [agent.clean_run(seed=s) for s in range(120, 270)], target_fp_rate=0.02
    )
    explainer = TriageExplainer()

    print(scorer.evaluate(agent.clean_run(seed=9999), baseline, thr).summary())
    for ft, sev in [("latency_injection", 0.8), ("tool_misselection", 0.9), ("cost_blowup", 0.9)]:
        run = faulted_run(agent, 4242, FaultSpec(ft, sev, onset_step=1))
        report = scorer.evaluate(run, baseline, thr)
        print(report.summary())
        exp = explainer.explain(report)
        if exp is not None:
            print(f"          cause:  {exp.suspected_cause}")
            print(f"          action: {exp.suggested_action}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from bellwether.baseline import BaselineManager
    from bellwether.dashboard import render_dashboard
    from bellwether.detect import DriftScorer, ScoringConfig
    from bellwether.detect.engine import calibrate_threshold
    from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run

    agent = FixtureAgent(agent_id="support-bot", task_class="qa")
    mgr = BaselineManager()
    for s in range(120):
        mgr.learn(agent.clean_run(seed=s))
    baseline = mgr.baseline_for(agent.clean_run(seed=0))
    assert baseline is not None
    scorer = DriftScorer(ScoringConfig(min_samples=30))
    thr = calibrate_threshold(
        scorer, baseline, [agent.clean_run(seed=s) for s in range(120, 270)], target_fp_rate=0.02
    )
    reports = [scorer.evaluate(agent.clean_run(seed=s), baseline, thr) for s in range(900, 906)]
    for i, (ft, sev) in enumerate(
        [("latency_injection", 0.8), ("tool_misselection", 0.9), ("cost_blowup", 0.9)]
    ):
        run = faulted_run(agent, 950 + i, FaultSpec(ft, sev, onset_step=1))
        reports.append(scorer.evaluate(run, baseline, thr))

    out = Path(args.output)
    out.write_text(render_dashboard(reports=reports), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _cmd_eval_gate(args: argparse.Namespace) -> int:
    """CI regression gate: run the benchmark and fail if quality drops below thresholds."""
    from bellwether.eval import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig(
        n_train=60, n_cal=100, n_eval_benign=150, n_eval_per_fault=8, n_seeds=6, min_samples=20
    )
    report = run_benchmark(config=cfg)
    m = report.metrics
    checks = [
        ("precision", m["precision"].point, ">=", args.min_precision),
        ("detection_rate", m["detection_rate"].point, ">=", args.min_detection),
        ("fp_rate", m["fp_rate"].point, "<=", args.max_fp),
    ]
    print(report.render())
    print("\nCI eval gate:")
    ok = True
    for name, value, op, bound in checks:
        passed = value >= bound if op == ">=" else value <= bound
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} {value:.3f} {op} {bound:.3f}")
    if not ok:
        print("\nEVAL GATE FAILED — quality regressed below thresholds", file=sys.stderr)
        return 1
    print("\neval gate passed")
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    from bellwether import __version__

    print(f"bellwether {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bellwether", description="Behavioral drift detection for AI agents."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark", help="run the drift benchmark with confidence intervals")
    p_bench.add_argument("--quick", action="store_true", help="smaller, faster config")
    p_bench.set_defaults(func=_cmd_benchmark)

    sub.add_parser(
        "demo", help="learn a baseline, inject faults, show alerts + triage"
    ).set_defaults(func=_cmd_demo)

    p_dash = sub.add_parser("dashboard", help="write a self-contained HTML dashboard")
    p_dash.add_argument("-o", "--output", default="bellwether_dashboard.html")
    p_dash.set_defaults(func=_cmd_dashboard)

    p_gate = sub.add_parser("eval-gate", help="CI regression gate over the benchmark")
    p_gate.add_argument("--min-precision", type=float, default=0.90)
    p_gate.add_argument("--min-detection", type=float, default=0.80)
    p_gate.add_argument("--max-fp", type=float, default=0.04)
    p_gate.set_defaults(func=_cmd_eval_gate)

    sub.add_parser("version", help="print the version").set_defaults(func=_cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
