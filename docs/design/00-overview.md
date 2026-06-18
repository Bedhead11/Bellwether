# BELLWETHER — Design Decisions Index

BELLWETHER detects **behavioral drift** in AI agents: the silent degradation where an agent
stops crashing but starts looping, mis-selecting tools, getting slower, costlier, or quietly
producing worse output. It learns a per-agent behavioral baseline from observed runs, then
flags statistically significant deviations before they surface as failures.

The brief (§11) requires the four `[BRAINSTORM REQUIRED]` decisions be designed *before*
implementing their subsystems. Each document below presents 2–3 options with tradeoffs,
recommends one, justifies it, and names the riskiest unknown.

| # | Decision | Doc | Lands in |
|---|---|---|---|
| 1 | Anti-plateau self-improvement strategy (highest stakes) | [`01-anti-plateau-self-improvement.md`](01-anti-plateau-self-improvement.md) | Phase 2–3 |
| 2 | Testing & evaluation harness (highest stakes) | [`02-eval-harness.md`](02-eval-harness.md) | Phase 0–2 |
| 3 | Drift vs. intended-change disambiguation | [`03-drift-vs-intended-change.md`](03-drift-vs-intended-change.md) | Phase 0–1 |
| 4 | Baseline representation | [`04-baseline-representation.md`](04-baseline-representation.md) | Phase 1 |

## How the decisions interlock

- **#4 (baseline)** produces calibrated, attributable per-family drift sub-scores. Its
  estimators emit change-points consumed by **#3**.
- **#3 (intended-change)** keys baselines by config fingerprint and gates whether a
  step-change becomes an alert or a re-baseline.
- **#2 (eval)** is the ground-truth engine: the fixture-agent fault injector generates labeled
  traces, and the CI gate is the *same* gate that governs self-improvement.
- **#1 (self-improvement)** optimizes the prompt/skill/topology tiers, scored by **#2**,
  governed by the audit layer, kept honest by held-out curves and anti-collusion tripwires.

## Cross-cutting principles (from the brief)

- **Honesty in evals.** Confidence intervals, held-out sets, never single-run numbers; a
  rigorous negative result beats a hand-waved positive one.
- **Attribution is the product.** Every score names which signal drifted, on which agent,
  when, by how much — this drives the composable (not monolithic) baseline.
- **FP-rate is a first-class objective.** The project dies if it cries wolf; degenerate
  optima ("alert on nothing/everything") must lose on the multi-objective eval.
- **Local-first / cost-disciplined.** Local models by default; API only for hard reasoning
  via a cost-controlled router; PII redaction at ingest.
- **Governance is the moat.** Every alert, baseline update, and self-modification is logged
  append-only with diff + rationale + metric deltas.

See [`../../README.md`](../../README.md) for the phased build plan and quickstart.
