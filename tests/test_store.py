"""L0 unit tests for the DuckDB run store (round-trip, query, idempotency, persistence)."""

from pathlib import Path

from bellwether.fixtures import FixtureAgent, generate_dataset
from bellwether.ingest import RunStore


def test_roundtrip_single_run() -> None:
    agent = FixtureAgent()
    run = agent.clean_run(seed=1)
    with RunStore() as store:
        store.add(run)
        assert store.count() == 1
        got = store.get(run.run_id)
        assert got is not None
        assert got.run_id == run.run_id
        assert got.step_count == run.step_count
        assert got.fingerprint_hash == run.fingerprint_hash


def test_add_is_idempotent_on_run_id() -> None:
    agent = FixtureAgent()
    run = agent.clean_run(seed=1)
    with RunStore() as store:
        store.add(run)
        store.add(run)  # replace, not duplicate
        assert store.count() == 1


def test_query_filters() -> None:
    agent = FixtureAgent(agent_id="agent-q", task_class="search")
    runs = generate_dataset(agent, n_benign=5, n_per_fault=2)
    with RunStore() as store:
        store.add_many(runs)

        all_runs = store.query(agent_id="agent-q")
        assert len(all_runs) == len(runs)

        benign = store.query(agent_id="agent-q", has_fault=False)
        faulted = store.query(agent_id="agent-q", has_fault=True)
        assert len(benign) == 5
        assert len(faulted) == len(runs) - 5
        assert all(r.injected_fault is None for r in benign)
        assert all(r.injected_fault is not None for r in faulted)


def test_query_ordered_by_start_time() -> None:
    agent = FixtureAgent()
    runs = generate_dataset(agent, n_benign=4, n_per_fault=1)
    with RunStore() as store:
        store.add_many(runs)
        fetched = store.query()
        times = [r.start_time for r in fetched]
        assert times == sorted(times)


def test_persistence_to_disk(tmp_path: Path) -> None:
    db = tmp_path / "runs.duckdb"
    agent = FixtureAgent()
    run = agent.clean_run(seed=3)
    with RunStore(db) as store:
        store.add(run)
    # Reopen: data survived.
    with RunStore(db) as store:
        assert store.count() == 1
        assert store.get(run.run_id) is not None
