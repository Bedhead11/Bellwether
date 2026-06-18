# BELLWETHER

**Behavioral drift detection for AI agents.** BELLWETHER learns a per-agent behavioral
baseline from observed runs, then flags statistically significant deviations — looping,
wrong-tool selection, latency creep, cost blowups, quietly degraded output — *before* they
surface as user complaints or runaway bills.

> A *bellwether* is the lead animal whose behavior signals where the flock is heading. This
> tool watches an agent's behavior and warns when it starts to wander.

Agents degrade **silently**: the same input yields different execution paths, so they don't
crash — they just get slower, costlier, or worse, with no error signal. Traditional
APM/observability assumes deterministic, reproducible execution and breaks here. BELLWETHER is
the intelligence layer on top of tracing: it *consumes* OpenTelemetry traces and learns what
"normal" looks like for a given agent, then detects drift from it.

- **Open-source & self-hostable.** Local-first; nothing phones home. PII redaction at ingest.
- **Rides existing rails.** OpenTelemetry-native ingest, `pip install`, DuckDB storage,
  MCP server (later) — frictionless adoption.
- **Self-improving & governed.** A closed loop tunes its own detectors, accretes a library of
  drift signatures, and evolves its detector ensemble — every change logged with rationale and
  before/after metrics.

> **Status: Phase 0 (scaffold & data faucet).** The canonical schema, OTel ingest, DuckDB
> store, PII redaction, and the fixture-agent fault-injection harness are in place and tested.
> The detection engine, baseline manager, and self-improvement loop are designed (see
> [`docs/design`](docs/design/)) and land in later phases.

## Why this exists

As of 2026, ~97% of organizations are deploying agentic AI but only ~12% have any centralized
way to control it. General agent tracing is a crowded space (Langfuse, Galileo, TrueFoundry);
BELLWETHER does **not** compete there. It builds the intelligence layer those tools explicitly
lack: *learning a per-agent baseline and detecting silent behavioral change* — the named
unsolved problem.

## Design decisions (read these first)

The four hardest design questions were worked out before implementation. Each doc presents
2–3 options, recommends one, and names the riskiest unknown:

- [#1 — Anti-plateau self-improvement strategy](docs/design/01-anti-plateau-self-improvement.md)
- [#2 — Testing & evaluation harness](docs/design/02-eval-harness.md)
- [#3 — Drift vs. intended-change disambiguation](docs/design/03-drift-vs-intended-change.md)
- [#4 — Baseline representation](docs/design/04-baseline-representation.md)
- [Index & how they interlock](docs/design/00-overview.md)

## Quickstart (developer)

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
uv run pytest                 # run the test suite
uv run ruff check .           # lint
uv run mypy                   # type-check
```

Generate labeled agent traces and land them in the store (the Phase 0 "data faucet"):

```python
from bellwether.fixtures import FixtureAgent, FaultSpec, generate_dataset
from bellwether.ingest import RunStore, redact_run

agent = FixtureAgent(agent_id="demo", task_class="qa")

# A reproducible mix of benign runs + known, dial-able faults (ground truth for eval).
runs = generate_dataset(
    agent,
    n_benign=20,
    fault_specs=[FaultSpec("latency_injection", severity=0.7),
                 FaultSpec("induced_loop", severity=0.8)],
    n_per_fault=5,
)

with RunStore("bellwether.duckdb") as store:
    store.add_many(redact_run(r) for r in runs)   # PII redacted at ingest
    print("benign:", len(store.query(has_fault=False)))
    print("faulted:", len(store.query(has_fault=True)))
```

Normalize OpenTelemetry spans into the canonical schema:

```python
from bellwether.ingest.otel import agentrun_from_otel_spans

run = agentrun_from_otel_spans(otel_span_dicts, run_id="r1", agent_id="demo")
```

## Architecture (target)

```
agent traces ─▶ INGEST/COLLECTOR ─▶ FEATURE EXTRACTORS ─▶ BASELINE MANAGER ⇄ DRIFT DETECTOR
(OTel, SDK,      (redaction,           (structural, tool,    (per agent/         ENSEMBLE
 MCP proxy)       canonical schema)     temporal, economic,   task-class,        (calibrated
                                        semantic, context)    versioned)          score + attribution)
                                                                                       │
   GOVERNANCE/AUDIT ◀─ SELF-IMPROVEMENT ENGINE ◀─ TRIAGE/EXPLANATION AGENT ◀───────────┘
   (append-only,        (prompt/skill/topology       (local LLM default,
    every change         tiers, eval-gated)           cost-controlled)
    logged w/ rationale)
```

## Phased build plan

| Phase | Deliverable | State |
|---|---|---|
| **0** | Scaffold + canonical schema + OTel ingest + DuckDB store + fixture fault-injection harness | **done** |
| 1 | Shippable v1: SDK, feature extractors, baseline manager, first detector ensemble, synthetic benchmark with CIs | next |
| 2 | Governance/audit + self-improvement (prompt & skill tiers) + CI-gated eval harness | planned |
| 3 | Topology self-improvement + MCP/OTel proxy (zero-code) + dashboard | planned |
| 4 | Optional: QLoRA triage fine-tune, multi-agent/coordination drift, published benchmark | planned |

## Project layout

```
src/bellwether/
  schema.py            canonical, versioned AgentRun event
  ingest/
    store.py           DuckDB-backed run store (local-first, zero infra)
    otel.py            OpenTelemetry span -> AgentRun normalizer
    redaction.py       ingest-time PII redaction (first-class)
  fixtures/
    agent.py           controllable synthetic agent (ground-truth generator)
    faults.py          dial-able fault taxonomy (latency, loop, cost, ...)
    dataset.py         labeled benign+faulted dataset generation
docs/design/           the four [BRAINSTORM REQUIRED] design decisions
tests/                 L0 unit + L1/L3 fixture tests
```

## License

Apache-2.0.
