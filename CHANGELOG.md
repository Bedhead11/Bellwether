# Changelog

All notable changes to BELLWETHER are recorded here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Distribution: CLI, Docker, CI eval gate

- `bellwether` console entrypoint (`cli.py`): `benchmark`, `demo`, `dashboard`, `eval-gate`,
  `version` — `pip install` to value in one command.
- `eval-gate` runs the benchmark and fails if precision/detection-rate/FP-rate cross thresholds;
  wired into CI as a regression gate (brief §6).
- `Dockerfile` for the self-hostable service exposing the CLI.

### Drift vs. intended-change runtime (design decision #3)

- `change/page_hinkley.py`: an online Page-Hinkley change-point detector to separate a persistent
  regime shift from a transient spike.
- `change/aware.py`: `ChangeAwareMonitor` implementing the doc-03 decision table — a new
  fingerprint or deploy marker opens a quarantined lineage (learned, not alerted); within a stable
  fingerprint, an unexplained *persistent* shift (Page-Hinkley over the drift stream) fires as
  drift; `accept_new_normal` promotes a quarantined lineage, audited. The change-point detector is
  gated to start only once a baseline is warm (and reset at that transition) so the warmup ramp
  isn't mistaken for a regime change; benign runs fold into the baseline only below a learn-ceiling
  so it never silently absorbs developing drift.
- Tests cover the metamorphic invariants: intended change is quarantined (not drift),
  accept-new-normal clears the alert, a persistent unexplained shift is detected, and benign
  traffic stays quiet.

### Phase 3 — Topology self-improvement + dashboard + queryable monitor

- `improve/topology.py`: MAP-Elites quality-diversity search over the detector-ensemble
  configuration (sustained-rule window/hits, FP-budget split across tracks, warmup), with
  multi-objective Pareto acceptance per behavioral niche, plateau detection + exploration boosts,
  and full audit logging. On the benchmark it explores diverse niches and confirms the hand-tuned
  default is near-optimal — a rigorous negative result that validates the machinery.
- `dashboard/`: a self-contained, dependency-free HTML dashboard (inline SVG + CSS) rendering the
  drift timeline, self-improvement curve, audit log (with hash-chain integrity), and ensemble
  niches.
- `monitor.py`: `DriftMonitor` query facade — ingest a run → scored verdict + audited alert, plus
  `drift_status`/`recent_alerts`/`baselines`/`status` queries returning serializable dicts.
- `mcp_server.py`: optional FastMCP server wrapping `DriftMonitor` as MCP tools (lazy import,
  `pip install 'bellwether[mcp]'`); no hard MCP dependency.
- All quality gates green; `examples/topology_demo.py` and `examples/dashboard_demo.py` added.

### Phase 2 — Governance + self-improvement (skill tier)

- `governance/`: append-only, hash-chained, tamper-evident audit log; `verify()` detects any
  edit/removal/reorder. Optional JSONL persistence.
- `improve/skills.py`: drift signatures — focused, direction-filtered detectors. `SubScore`
  gains a `magnitude` field so signatures separate overwhelming anomalies from rare benign tails
  where the conformal p-value floors.
- `improve/generator.py`, `gate.py`, `loop.py`: the self-improvement loop — mine a signature
  from labeled incidents → evaluate on a held-out benchmark → multi-objective governance gate
  (promote only on a genuine recall gain with no FP/precision regression) → archive lineage,
  with QD diversity, plateau detection, and full audit logging.
- `triage/`: deterministic, knowledge-base explainer mapping an alert to a cause + suggested
  action, with a pluggable LLM-narrator hook.
- Demo: a mined `cost_blowup` signature lifts held-out timely recall 0.70 → 0.83 (+0.13) within
  the 2% FP budget; an `output_degradation` candidate is correctly rejected (FP regression).
- Broke the detect↔improve import cycle via a `SignatureProvider` Protocol in `detect`.
- All quality gates green (pytest, mypy --strict, ruff); `examples/self_improve_demo.py` added.

### Phase 1 — Shippable v1 (detection engine + SDK + benchmark)

- `features/`: pure, deterministic extractors turning a run into an ordered
  `FeatureObservation` sequence (per-step + run-summary) covering all six families.
- `stats.py`: robust estimators (median/MAD), distribution-free conformal p-values, Šidák
  multiplicity correction, percentile-bootstrap CIs — dependency-free and deterministic.
- `baseline/`: per-(agent, task_class, fingerprint) windowed baselines with warmup-aware
  cold-start and JSON save/load.
- `detect/`: per-feature conformal/novelty detectors → Šidák aggregator with attribution →
  engine with three independently-calibrated tracks (critical single-step / k-of-w sustained /
  run-summary), each given a share of the false-positive budget.
- `eval/`: metric definitions (precision/recall/F1/FP-rate/lead-time with formulas) and an
  N-seed benchmark reporting every metric with 95% bootstrap CIs.
- `sdk/`: the `Bellwether` SDK — `watch`/`llm`/`tool`/`agent` context managers assemble a
  canonical run via contextvars, redact at ingest, and route to store + live scorer; plus
  `mark_deploy` for intended-change disambiguation.
- Benchmark (20 seeds, FP budget 2%): precision 0.973, detection-rate 0.878, timely recall
  0.730, FP-rate 0.013, median lead-time ~2.2 steps. All six fault types attributed to the
  correct family.
- 92 tests green; mypy --strict and ruff clean. `examples/detect_demo.py` and
  `examples/benchmark.py` added.

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
