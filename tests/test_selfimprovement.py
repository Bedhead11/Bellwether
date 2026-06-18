"""Tests for the self-improvement loop: gate logic, signature mining, end-to-end improvement."""

from bellwether.eval import BenchmarkConfig
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run
from bellwether.governance import AuditLog, EventType
from bellwether.improve import GovernanceGate, MetricVector, SelfImprovementLoop, mine_signature
from bellwether.improve.loop import _MINE_BASELINE_BASE


def _mv(recall, fp, precision, det=0.9, f1=0.85, lead=2.0) -> MetricVector:  # type: ignore[no-untyped-def]
    t = lambda p: (p, max(p - 0.005, 0.0), p + 0.005)  # noqa: E731
    return MetricVector(t(recall), t(fp), t(precision), t(det), t(f1), t(lead))


# --- L0: governance gate logic (the same gate that governs CI) ---------------------------


def test_gate_promotes_genuine_improvement() -> None:
    gate = GovernanceGate(fp_budget=0.02)
    current = _mv(recall=0.70, fp=0.013, precision=0.97)
    candidate = _mv(recall=0.80, fp=0.015, precision=0.97)  # recall up, fp within budget
    decision = gate.decide(current, candidate)
    assert decision.promote
    assert decision.metric_deltas["recall"] > 0


def test_gate_rejects_fp_regression() -> None:
    """A change that buys recall by blowing the false-positive budget must be rejected."""
    gate = GovernanceGate(fp_budget=0.02, fp_tol=0.01)
    current = _mv(recall=0.70, fp=0.013, precision=0.97)
    candidate = _mv(recall=0.95, fp=0.08, precision=0.97)  # "alert on everything"
    decision = gate.decide(current, candidate)
    assert not decision.promote
    assert any("fp-rate regression" in r for r in decision.reasons)


def test_gate_rejects_no_improvement() -> None:
    """A junk change with no target gain is rejected (so is 'alert on nothing')."""
    gate = GovernanceGate()
    current = _mv(recall=0.70, fp=0.013, precision=0.97)
    candidate = _mv(recall=0.70, fp=0.013, precision=0.97)  # no gain
    assert not gate.decide(current, candidate).promote
    nothing = _mv(recall=0.0, fp=0.0, precision=1.0)  # alert on nothing
    assert not gate.decide(current, nothing).promote


def test_gate_rejects_precision_regression() -> None:
    gate = GovernanceGate(precision_tol=0.03)
    current = _mv(recall=0.70, fp=0.013, precision=0.97)
    candidate = _mv(recall=0.85, fp=0.015, precision=0.80)  # precision collapses
    decision = gate.decide(current, candidate)
    assert not decision.promote
    assert any("precision regression" in r for r in decision.reasons)


# --- L1: signature mining ----------------------------------------------------------------


def test_mine_signature_recovers_cost_blowup_features() -> None:
    from bellwether.baseline import BaselineManager
    from bellwether.detect import DriftScorer

    agent = FixtureAgent()
    mgr = BaselineManager()
    for s in range(120):
        mgr.learn(agent.clean_run(seed=s))
    baseline = mgr.baseline_for(agent.clean_run(seed=0))
    scorer = DriftScorer()

    incidents = [
        faulted_run(agent, 5000 + j, FaultSpec("cost_blowup", 0.8, onset_step=1)) for j in range(12)
    ]
    sig = mine_signature("cost_blowup", incidents, baseline, scorer, top_k=2)
    assert sig is not None
    features = {f for f, _, _ in sig.feature_directions}
    # The cost-driving features should be discovered, in the "up" direction.
    assert features & {"step_input_tokens", "step_cost"}
    assert all(direction >= 0 for _, _, direction in sig.feature_directions)


# --- L1: end-to-end loop improves held-out recall and audits every change ----------------


def test_loop_improves_recall_and_audits() -> None:
    agent = FixtureAgent()
    audit = AuditLog()
    cfg = BenchmarkConfig(
        n_train=60, n_cal=60, n_eval_benign=40, n_eval_per_fault=6, n_seeds=3, min_samples=20
    )
    # Tiny benchmarks have wide CIs; loosen the guards so the clear cost_blowup win can promote.
    # The full demo (examples/) uses more seeds with the strict default gate.
    gate = GovernanceGate(fp_budget=0.05, fp_tol=0.05, precision_tol=0.08, min_recall_gain=0.02)
    loop = SelfImprovementLoop(
        agent=agent, eval_config=cfg, audit=audit, gate=gate, n_mine_baseline=80
    )

    history = loop.run(["cost_blowup", "output_degradation"])

    # The held-out recall curve climbs (at least one signature promoted).
    curve = history.recall_curve()
    assert curve[-1] > curve[0]
    assert len(history.final_library.names) >= 1

    # Every change is in the tamper-evident audit log.
    assert audit.verify()
    assert len(audit.query(event_type=EventType.PROMOTION)) >= 1
    assert len(audit.query(event_type=EventType.SIGNATURE_ADDED)) >= 1

    # The archive retains the full lineage (baseline + one entry per processed round).
    assert len(history.archive.entries) >= 2
    best = history.archive.best_by_recall()
    assert best is not None


def test_loop_baseline_seeds_are_disjoint_from_benchmark() -> None:
    # Mining baseline lives in a far seed namespace so it never overlaps held-out eval data.
    assert _MINE_BASELINE_BASE > 100 * 1_000_000
