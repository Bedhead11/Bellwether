# Stack Choices & Deliberate Deviations

The brief (§5) suggests a 2026 stack: `river`, `alibi-detect`, LangGraph, DSPy/GEPA, LiteLLM,
Ollama, FastMCP, Langfuse, Postgres+pgvector. This document records where the implementation
*follows* it, where it deliberately *deviates*, and why — the portfolio signal is "I know when a
dependency earns its place," not "I added every library on the list."

The unifying principle: the core engine stays **dependency-light, deterministic, and
CI-reproducible**; the heavy/optional pieces are designed as **drop-in slots** behind clean
interfaces so they can be added when they pay for themselves.

## Followed as-is

- **OpenTelemetry** as the trace contract — consumed, not reinvented (`ingest/otel.py`,
  `ingest/otlp.py`). This is what makes BELLWETHER framework-agnostic and adoptable.
- **DuckDB + Parquet/SQLite** local-first storage (`ingest/store.py`) — the brief's v1 choice.
- **`uv` / `ruff` / `mypy --strict` / `pytest`** toolchain.
- **FastMCP** for the MCP server — but as an *optional extra* (`pip install bellwether[mcp]`,
  lazy import in `mcp_server.py`), so it is never a hard dependency.

## Deliberate deviations

### `river` / `alibi-detect` → a self-contained `stats.py`
The streaming-stats and two-sample-test libraries are excellent, but the v1 core needs three
things they over-serve: robust online estimators, distribution-free anomaly scores, and bootstrap
CIs. Implementing these in ~150 lines of dependency-free, deterministic code (`stats.py`,
`change/page_hinkley.py`) makes **every unit test exact** and keeps CI free of a heavy numerical
stack. The detector ensemble is *composable by design* (design doc 04), so a `river` ADWIN or an
`alibi-detect` MMD detector can be added as **additional ensemble members** (and the topology tier
can weight them) without touching the core. *Revisit when:* multivariate kernel two-sample power
on real embeddings becomes the bottleneck.

### LangGraph → a plain, audited Python loop
The brief suggests LangGraph for stateful, checkpointed, auditable internal orchestration. The
self-improvement loop (`improve/loop.py`, `improve/topology.py`) is a deterministic Python loop
instead, because the properties LangGraph would provide are delivered more directly here:
- **checkpointing / time-travel** → the `Archive` keeps the full lineage of every variant
  (rollback is selecting an archive entry);
- **auditability** → the append-only, hash-chained audit log records every promotion/rejection/
  plateau with diffs and metric deltas;
- **determinism** → the loop has no LLM or network call, so a plain loop is fully reproducible in
  CI, which a graph runtime would complicate.
*Revisit when:* the loop gains genuinely concurrent, long-running, human-in-the-loop steps.

### DSPy/GEPA + LiteLLM + Ollama → a deterministic triage explainer with an LLM hook
The triage/explanation agent (`triage/explain.py`) is a deterministic, knowledge-base explainer
today: zero cost, zero latency, fully testable, and useful (it maps attribution → cause →
suggested action). It exposes a `narrator` hook so a local-LLM backend (Ollama via LiteLLM,
optimized with DSPy/GEPA) can refine the prose **without changing the structured, auditable
output**. This honors the cost-discipline constraint (§2) and the brief's own framing of the
fine-tune as a *deliberate* judgment call (§8) — we don't call an LLM until it earns its place.
*Revisit when:* labeled (signature → explanation) pairs accumulate enough to justify the prompt
tier / a QLoRA triage model.

### Langfuse self-hosting → the internal audit layer
Self-observability of BELLWETHER's own runs is provided by the hash-chained audit log (governance
dogfoods itself: every self-modification is recorded and verifiable). Langfuse can be added later
as an OTel *exporter* of the internal LangGraph-style traces; it is not needed for v1 correctness.

### Postgres + pgvector → DuckDB (v1), Postgres deferred to v2 scale
Local-first by default per §2; Postgres+pgvector is the documented v2 scale option, not a v1 need.

## What this buys

- `pip install bellwether` pulls **two** runtime deps (`pydantic`, `duckdb`) — fast, frictionless
  adoption (§9), no GPU/LLM required to get value.
- Every result is **deterministic and CI-reproducible** (including the Hypothesis suite, which is
  `derandomize`d), so the honesty story (CIs, held-out sets, negative results) is verifiable.
- The heavy pieces remain **one clean interface away**, so growth doesn't require a rewrite.
