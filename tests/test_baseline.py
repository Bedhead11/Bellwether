"""L0 unit tests for the baseline manager (learning, keying, windowing, serialization)."""

from pathlib import Path

from bellwether.baseline import BaselineManager
from bellwether.fixtures import FixtureAgent


def test_learns_numeric_and_categorical() -> None:
    agent = FixtureAgent()
    mgr = BaselineManager()
    for s in range(10):
        mgr.learn(agent.clean_run(seed=s))

    b = mgr._baselines[("fixture-agent", "default", agent.fingerprint().hash())]
    # latency baseline for LLM steps accumulated samples
    assert b.numeric_count("step_latency", "llm") > 0
    # tool identity frequency table populated
    counts, total = b.categorical_counts("tool_id", "*")
    assert total > 0
    assert "search" in counts


def test_separate_lineage_per_fingerprint() -> None:
    mgr = BaselineManager()
    a1 = FixtureAgent(agent_id="a", temperature=0.2)
    a2 = FixtureAgent(agent_id="a", temperature=0.9)  # different fingerprint
    mgr.learn(a1.clean_run(seed=1))
    mgr.learn(a2.clean_run(seed=1))
    assert len(mgr.keys) == 2


def test_window_bounds_memory() -> None:
    mgr = BaselineManager(window_size=5)
    agent = FixtureAgent()
    for s in range(50):
        mgr.learn(agent.clean_run(seed=s))
    b = mgr.baseline_for(agent.clean_run(seed=0))
    assert b is not None
    assert b.numeric_count("step_latency", "llm") <= 5


def test_roundtrip_serialization(tmp_path: Path) -> None:
    agent = FixtureAgent()
    mgr = BaselineManager()
    for s in range(8):
        mgr.learn(agent.clean_run(seed=s))

    path = tmp_path / "baseline.json"
    mgr.save(path)
    restored = BaselineManager.load(path)

    assert restored.keys == mgr.keys
    key = mgr.keys[0]
    orig = mgr._baselines[key]
    new = restored._baselines[key]
    for name, ctx in orig.numeric_keys:
        assert new.numeric_window(name, ctx) == orig.numeric_window(name, ctx)
    for name, ctx in orig.categorical_keys:
        assert new.categorical_counts(name, ctx) == orig.categorical_counts(name, ctx)


def test_unseen_run_has_no_baseline() -> None:
    mgr = BaselineManager()
    assert mgr.baseline_for(FixtureAgent().clean_run(seed=1)) is None
