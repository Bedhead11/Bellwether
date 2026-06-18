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

> **Status: Phase 2 (governance + self-improvement).** On top of the Phase 1 detector:
> a tamper-evident audit layer, a skill-tier self-improvement loop (generator → eval-gated
> governance gate → archive) that learns drift signatures and *demonstrably* lifts held-out
> recall under multi-objective, anti-degenerate guards, and a triage explainer that turns an
> alert into a cause + suggested action. Topology-tier self-improvement, the zero-code MCP/OTel
> proxy, and the dashboard are Phase 3.

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

## Quickstart

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
uv run pytest                              # run the test suite
uv run python examples/detect_demo.py      # learn -> inject faults -> watch it alert + triage
uv run python examples/benchmark.py        # the benchmark with confidence intervals
uv run python examples/self_improve_demo.py  # the skill tier self-improving (held-out curve)
```

### Instrument an agent with the SDK

```python
from bellwether import Bellwether, BaselineManager, DriftScorer

mgr = BaselineManager()

# Phase 1: learn a behavioral baseline from healthy runs.
bw = Bellwether(agent_id="support-bot", manager=mgr, learn=True)
for task in healthy_tasks:
    with bw.watch(task_class="qa"):
        with bw.llm(model="gpt-4o-mini") as call:
            resp = my_llm(...)
            call.set_tokens(input=resp.in_tokens, output=resp.out_tokens)
        with bw.tool("search", args={"q": query}):
            results = search(query)

# Phase 2: monitor — every run is scored; drift is delivered to your callback.
monitor = Bellwether(
    agent_id="support-bot", manager=mgr, scorer=DriftScorer(),
    on_report=lambda r: print(r.summary()) if r.alert.triggered else None,
)
with monitor.watch(task_class="qa"):
    ...  # same instrumentation; alerts fire when behavior drifts

# Declare an intended change so a deploy isn't mistaken for drift (design doc 03):
monitor.mark_deploy(version="2.0", note="new system prompt")
```

A drift alert names *which* signal drifted, on which agent, at which step:

```
[DRIFT] support-bot-...: tool/tool_id at step 1 (score=0.998)
          contributing families: tool=1.00, temporal=0.95, context=0.70
```

### Or feed OpenTelemetry traces directly

```python
from bellwether.ingest.otel import agentrun_from_otel_spans

run = agentrun_from_otel_spans(otel_span_dicts, run_id="r1", agent_id="demo")
```

## Benchmark

Detection quality on the synthetic fault benchmark — 20 seeds, exact ground truth (injected
faults), 95% bootstrap CIs, false-positive budget 2%:

| metric | value (95% CI) |
|---|---|
| precision | 0.973 [0.962, 0.984] |
| timely recall | 0.730 [0.714, 0.748] |
| detection rate (caught at all) | 0.878 [0.859, 0.897] |
| F1 | 0.833 [0.822, 0.845] |
| **false-positive rate** | **0.013 [0.007, 0.018]** (budget 0.02) |
| median lead-time | 2.2 steps before visible failure |

Per-fault *timely* recall: `latency_injection` 0.99, `induced_loop` 1.00, `retry_storm` 1.00,
`tool_misselection` 0.91, `cost_blowup` 0.41, `output_degradation` 0.07. The last two — subtle,
single-direction drifts on alternating steps against a tight visible-failure line — are honestly
hard for the static detector and are exactly where the self-improvement engine has room to show
compounding gains. Reproduce with `uv run python examples/benchmark.py`.

## Self-improvement & governance

The detection engine improves itself, under a governance layer that records every change.

- **Skill tier (working).** A closed loop — generator → evaluator → governance gate → archive —
  *mines* drift signatures from labeled incidents and promotes one only if the held-out
  benchmark shows a genuine recall gain with **no** false-positive or precision regression
  (the same conservative, CI-style gate). In the demo, a mined `cost_blowup` signature is
  promoted and lifts held-out timely recall **0.70 → 0.83 (+0.13)** with FP within the 2% budget
  and precision held; an `output_degradation` candidate is **correctly rejected** for an FP
  regression, and an already-solved fault is rejected for no gain. Every decision is written to a
  **tamper-evident, hash-chained audit log**. Reproduce with
  `uv run python examples/self_improve_demo.py`.
- **Anti-plateau (design doc 01).** A quality-diversity archive keyed by drift type (diverse by
  construction), a multi-objective gate (so "alert on everything" / "alert on nothing" both
  lose), and an explicit plateau detector on the held-out recall curve.
- **Triage explainer.** Turns an alert into a suspected cause and a suggested action (e.g.
  *"recognized cost-blowup drift … set a per-step token budget"*), with a hook for a local-LLM
  narrator (the prompt-tier optimization target).

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
| **1** | Shippable v1: SDK, feature extractors, baseline manager, calibrated detector ensemble, benchmark with CIs | **done** |
| **2** | Governance/audit + self-improvement (skill tier) + multi-objective gate + triage explainer | **done** |
| 3 | Topology self-improvement + MCP/OTel proxy (zero-code) + dashboard | next |
| 4 | Optional: QLoRA triage fine-tune, multi-agent/coordination drift, published benchmark | planned |

## Project layout

```
src/bellwether/
  schema.py            canonical, versioned AgentRun event
  stats.py             robust estimators, conformal p-values, bootstrap CIs
  ingest/              DuckDB store · OTel normalizer · PII redaction
  fixtures/            controllable synthetic agent + dial-able fault taxonomy
  features/            pure extractors: run -> FeatureObservation sequence
  baseline/            per-(agent, task_class, fingerprint) windowed baselines
  detect/              conformal detectors -> aggregator -> calibrated engine
  eval/                metrics + N-seed benchmark with bootstrap CIs
  sdk/                 the Bellwether SDK (@watch instrumentation)
  governance/          append-only, hash-chained, tamper-evident audit log
  improve/             skill-tier self-improvement: generator/gate/archive/loop
  triage/              alert -> human cause + suggested action (LLM-pluggable)
docs/design/           the four [BRAINSTORM REQUIRED] design decisions
examples/              quickstart · detect_demo · benchmark · self_improve_demo
tests/                 L0 unit + L1/L3 property + integration tests
```

## License

Apache-2.0.
