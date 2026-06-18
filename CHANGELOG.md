# Changelog

All notable changes to BELLWETHER are recorded here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Phase 0 — Scaffold & data faucet

Design (the four `[BRAINSTORM REQUIRED]` decisions, in `docs/design/`):
- #1 Anti-plateau self-improvement: quality-diversity (MAP-Elites) archive + adversarial
  co-evolution (with anti-collusion tripwire) + multi-objective Pareto + plateau-triggered
  exploration, measured on a rotating held-out curve with bootstrap CIs.
- #2 Eval harness: layered L0 unit / L1 synthetic fault injection / L2 replay / L3 metamorphic
  / L4 statistical CI gate, with metric formulas and the fault taxonomy.
- #3 Drift vs. intended-change: config fingerprint + deploy markers + quarantine window +
  change-point correlation + audited accept-new-normal.
- #4 Baseline representation: composable per-family online estimators fused by a calibrated
  aggregator, EB-shrinkage cold-start, distribution-free calibration.

Implementation:
- Canonical, versioned `AgentRun` event schema (`bellwether.schema`).
- DuckDB-backed, local-first run store (`bellwether.ingest.store`).
- OpenTelemetry span → `AgentRun` normalizer reading GenAI semantic conventions
  (`bellwether.ingest.otel`).
- Ingest-time PII redaction, structure-preserving (`bellwether.ingest.redaction`).
- Fixture-agent fault-injection harness — the ground-truth data faucet
  (`bellwether.fixtures`): controllable agent, 6 dial-able fault types, labeled dataset
  generation.
- Tooling: `uv`, `ruff`, `mypy --strict`, `pytest`; GitHub Actions CI (3.11 + 3.12).
- 45 tests (L0 unit + L1/L3 fixture properties) green; mypy and ruff clean.
- `examples/quickstart.py` end-to-end smoke demo.
