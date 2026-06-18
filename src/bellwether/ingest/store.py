"""DuckDB-backed event store for ``AgentRun`` records.

Local-first and zero-infra per the brief: DuckDB embeds in-process and reads/writes a single
file (or ``:memory:`` for tests). Phase 0 stores each run as a JSON document plus a handful of
flattened, indexed columns so the common queries (by agent, task_class, fingerprint, time
window, fault label) are fast without exploding the schema. Spans live inside the JSON blob;
a dedicated span/feature table arrives when the feature extractors need columnar access.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import TracebackType

import duckdb

from bellwether.schema import SCHEMA_VERSION, AgentRun

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            VARCHAR PRIMARY KEY,
    schema_version    INTEGER NOT NULL,
    agent_id          VARCHAR NOT NULL,
    task_class        VARCHAR NOT NULL,
    fingerprint       VARCHAR NOT NULL,
    start_time        TIMESTAMP NOT NULL,
    end_time          TIMESTAMP NOT NULL,
    status            VARCHAR NOT NULL,
    step_count        INTEGER NOT NULL,
    has_fault         BOOLEAN NOT NULL,
    fault_type        VARCHAR,
    document          JSON NOT NULL
);
"""

# DuckDB has no secondary-index DDL parity with Postgres in all versions; these are best-effort
# and ignored if unsupported. The primary key on run_id already covers dedupe/upsert.
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(agent_id, task_class)",
    "CREATE INDEX IF NOT EXISTS idx_runs_fp ON runs(fingerprint)",
    "CREATE INDEX IF NOT EXISTS idx_runs_start ON runs(start_time)",
]


class RunStore:
    """Persist and query ``AgentRun`` records.

    Use as a context manager or call :meth:`close` explicitly. ``path=":memory:"`` (default)
    gives an ephemeral store ideal for tests and the fixture harness.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(self._path)
        self._con.execute(_SCHEMA)
        for stmt in _INDEXES:
            # Index creation is an optimization, not a correctness requirement.
            with suppress(duckdb.Error):
                self._con.execute(stmt)

    # --- lifecycle ----------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- writes -------------------------------------------------------------------------

    def add(self, run: AgentRun) -> None:
        """Insert or replace a single run (idempotent on ``run_id``)."""
        self._con.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, schema_version, agent_id, task_class, fingerprint,
                 start_time, end_time, status, step_count, has_fault, fault_type, document)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run.run_id,
                run.schema_version,
                run.agent_id,
                run.task_class,
                run.fingerprint_hash,
                run.start_time,
                run.end_time,
                run.status.value,
                run.step_count,
                run.injected_fault is not None,
                run.injected_fault.fault_type if run.injected_fault else None,
                run.model_dump_json(),
            ],
        )

    def add_many(self, runs: Iterable[AgentRun]) -> int:
        n = 0
        for run in runs:
            self.add(run)
            n += 1
        return n

    # --- reads --------------------------------------------------------------------------

    def get(self, run_id: str) -> AgentRun | None:
        row = self._con.execute("SELECT document FROM runs WHERE run_id = ?", [run_id]).fetchone()
        if row is None:
            return None
        return AgentRun.model_validate_json(row[0])

    def count(self) -> int:
        row = self._con.execute("SELECT COUNT(*) FROM runs").fetchone()
        assert row is not None
        return int(row[0])

    def query(
        self,
        *,
        agent_id: str | None = None,
        task_class: str | None = None,
        fingerprint: str | None = None,
        has_fault: bool | None = None,
        limit: int | None = None,
    ) -> list[AgentRun]:
        """Fetch runs matching the given filters, ordered by ``start_time``."""
        clauses: list[str] = []
        params: list[object] = []
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if task_class is not None:
            clauses.append("task_class = ?")
            params.append(task_class)
        if fingerprint is not None:
            clauses.append("fingerprint = ?")
            params.append(fingerprint)
        if has_fault is not None:
            clauses.append("has_fault = ?")
            params.append(has_fault)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT document FROM runs{where} ORDER BY start_time"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self._con.execute(sql, params).fetchall()
        return [AgentRun.model_validate_json(r[0]) for r in rows]

    def iter_runs(self, **filters: object) -> Iterator[AgentRun]:
        yield from self.query(**filters)  # type: ignore[arg-type]

    @contextmanager
    def raw(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Escape hatch for ad-hoc analytical SQL (e.g., notebooks, the dashboard)."""
        yield self._con

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION
