"""Tests for drift-vs-intended-change disambiguation (design doc 03)."""

from bellwether.baseline import BaselineManager
from bellwether.change import ChangeAwareMonitor, PageHinkley, VerdictKind
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run

# --- L0: Page-Hinkley change-point detector ---------------------------------------------


def test_page_hinkley_stable_stream_no_change() -> None:
    ph = PageHinkley(delta=0.01, threshold=1.5, min_samples=5)
    fired = [ph.update(0.5) for _ in range(50)]
    assert not any(fired)


def test_page_hinkley_detects_persistent_jump() -> None:
    ph = PageHinkley(delta=0.01, threshold=1.5, min_samples=5)
    for _ in range(20):
        assert not ph.update(0.2)
    fired = any(ph.update(0.95) for _ in range(15))
    assert fired


def test_page_hinkley_reset() -> None:
    ph = PageHinkley(threshold=1.5, min_samples=1)
    for _ in range(10):
        ph.update(1.0)
    ph.reset()
    assert ph.n == 0


# --- L1: change-aware monitor ------------------------------------------------------------


def _monitor() -> ChangeAwareMonitor:
    return ChangeAwareMonitor(
        manager=BaselineManager(),
        scorer=DriftScorer(ScoringConfig(min_samples=20)),
        ph_threshold=1.5,
    )


def _establish(mon: ChangeAwareMonitor, agent: FixtureAgent, n: int = 120) -> None:
    for s in range(n):
        mon.observe(agent.clean_run(seed=s))


def test_first_run_is_warmup() -> None:
    mon = _monitor()
    v = mon.observe(FixtureAgent().clean_run(seed=0))
    assert v.kind == VerdictKind.WARMUP
    assert mon.active_fingerprint("fixture-agent", "default") is not None


def test_new_fingerprint_is_intended_change_not_drift() -> None:
    mon = _monitor()
    base = FixtureAgent(temperature=0.2)  # fingerprint A
    _establish(mon, base)

    deployed = FixtureAgent(temperature=0.9)  # same agent/task, different fingerprint
    v = mon.observe(deployed.clean_run(seed=5000))
    assert v.kind == VerdictKind.INTENDED_CHANGE  # a declared config change, not drift
    assert v.fingerprint in mon.quarantined("fixture-agent", "default")


def test_accept_new_normal_then_monitoring_clears_alert() -> None:
    """Metamorphic invariant: re-baselining after a declared change clears the alert."""
    mon = _monitor()
    base = FixtureAgent(temperature=0.2)
    _establish(mon, base)

    deployed = FixtureAgent(temperature=0.9)
    # Feed enough new-config runs to build its candidate baseline (all quarantined, no drift).
    for s in range(5000, 5120):
        v = mon.observe(deployed.clean_run(seed=s))
        assert v.kind == VerdictKind.INTENDED_CHANGE

    accepted = mon.accept_new_normal("fixture-agent", "default")
    assert accepted == deployed.fingerprint().hash()

    # New-config runs are now the accepted normal → monitoring, not drift.
    kinds = [mon.observe(deployed.clean_run(seed=s)).kind for s in range(5200, 5210)]
    assert all(k == VerdictKind.MONITORING for k in kinds)
    assert mon.audit.verify()


def test_persistent_unexplained_shift_is_drift() -> None:
    """Same fingerprint, a sustained behavioral shift → drift via the change-point detector."""
    mon = _monitor()
    agent = FixtureAgent(temperature=0.2)
    _establish(mon, agent)

    # A sustained run of latency-drifted traffic (same fingerprint, no declared change).
    verdicts = []
    for s in range(6000, 6012):
        run = faulted_run(agent, s, FaultSpec("latency_injection", 0.9, onset_step=1))
        verdicts.append(mon.observe(run))

    assert any(v.kind == VerdictKind.DRIFT for v in verdicts)
    assert any(v.regime_change for v in verdicts)


def test_benign_stream_stays_monitoring() -> None:
    mon = _monitor()
    agent = FixtureAgent(temperature=0.2)
    _establish(mon, agent)
    kinds = [mon.observe(agent.clean_run(seed=s)).kind for s in range(7000, 7030)]
    assert all(k == VerdictKind.MONITORING for k in kinds)  # no false drift on benign traffic
