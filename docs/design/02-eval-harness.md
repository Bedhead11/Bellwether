# Design Decision #2 — Testing & Evaluation Harness

> Status: **Decided** (design). The L0/L1/L3 layers begin in Phase 0–1 (they generate the
> ground truth everything else needs); L2 and the CI gate harden in Phase 2.

## Problem

We must measure detection quality with **ground truth** despite double non-determinism (the
agent under observation is non-deterministic, and so is the detector's learned baseline).
The harness must (a) produce trustworthy precision/recall/lead-time/FP numbers, (b) gate
self-improvement (#1) and CI, and (c) run reproducibly. A weak eval makes the whole project
hand-waving — this is a credibility load-bearing component.

## Options considered

### Option A — Synthetic fault-injection only
Fixture agents dialed to exhibit known faults; ground truth is exact.
- **Pros:** deterministic labels → exact P/R/F1/lead-time; cheap; CI-friendly.
- **Cons:** realism gap — synthetic traces may not resemble production drift, so numbers
  can be optimistic. Necessary but insufficient alone.

### Option B — Replay corpus + fault injection only
Record real agent runs, inject faults into replays.
- **Pros:** realistic substrate; still has ground-truth labels (we know what we injected).
- **Cons:** needs a corpus (cold-start chicken-and-egg early); replay fidelity work; doesn't
  by itself test invariants or detector unit correctness.

### Option C — Layered harness L0–L4 (RECOMMENDED)
Combine synthetic fixtures, replay injection, metamorphic invariants, and a statistically
rigorous CI gate into one stack where each layer covers the others' blind spots.

## Recommendation — Option C, five layers

```
L0  Unit         deterministic tests of extractors & detectors (pure functions)
L1  Synthetic    fixture-agent fault injection → exact P/R/lead-time/FP with CIs
L2  Replay       real-trace corpus + injected faults → realism check
L3  Metamorphic  invariant properties that must ALWAYS hold (property-based)
L4  CI gate      statistical regression gate over held-out set; gates self-improvement
```

### L0 — Unit (deterministic)
Feature extractors and individual drift detectors are **pure functions** with fixed inputs
and asserted outputs. No statistics, no randomness. This is the TDD floor (brief §11) and
catches the majority of correctness bugs cheaply. Begins in Phase 0.

### L1 — Synthetic fault injection (the ground-truth engine)
Controllable **fixture agents** emit canonical `AgentRun` traces; a **fault injector**
applies a parameterized fault. Because the fault is *known*, labels are exact → deterministic
precision/recall/F1/lead-time. This is where headline numbers come from. Begins in Phase 0
(minimal) and is the data faucet for everything downstream.

### L2 — Replay corpus + injection (realism)
Record real runs (open frameworks / public OTel GenAI traces), normalize to `AgentRun`,
replay them, then inject the *same* taxonomy of faults. Closes the realism gap of L1 while
keeping ground-truth labels. Phase 2.

### L3 — Metamorphic invariants (property-based, via Hypothesis)
Properties that must hold for *any* input, testing the detector's logic rather than a point:

- **Monotonicity:** injecting a *strictly worse* fault must not *decrease* the drift score.
- **Benign stability:** a benign re-run within baseline variance must not alert
  (bounds FP-rate by construction).
- **Re-baseline clears:** after a *declared* config change + accept-new-normal, the alert
  for that change must clear (ties to drift-vs-intended-change, `03`).
- **Permutation invariance** where features are set-based (e.g., tool-frequency) and
  **order sensitivity** where they are sequential (e.g., loop detection).
- **Attribution soundness:** the family flagged as the dominant contributor must be the
  family the fault was injected into (for single-fault scenarios).

### L4 — CI regression gate (statistical, gates self-improvement)
Same gate used by the self-improvement governance step (#1). Runs in CI.

## Metric definitions (formulas)

Let an *episode* be one run with a known label (drifted / benign) and, for drifted runs, a
**visible-failure marker** `t_fail` (the step at which a human/SLA would notice). The
detector emits a **sustained alert** at the first step `t_alert` where the calibrated drift
score exceeds threshold for `p` consecutive steps (debounce against single-point noise).

- **TP** = drifted episode with `t_alert` defined and `t_alert ≤ t_fail`.
- **FN** = drifted episode with no sustained alert before `t_fail`.
- **FP** = benign episode with any sustained alert.
- **TN** = benign episode with no sustained alert.

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2·P·R / (P + R)
- **FP-rate** = FP / (FP + TN)  *(per-episode)*; also report **FP per 1k benign steps**
  for streaming realism. FP-rate is a **first-class objective**, not a derived afterthought
  — the project dies if it cries wolf.
- **Lead-time** = `t_fail − t_alert` (in steps and wall-clock), reported only over TPs.
  This is the value proposition; it is a first-class metric. Report the **distribution**
  (median + IQR), not just the mean — a few huge lead-times must not hide many near-zero
  ones.
- **Cost** = tokens + wall-clock of running detection per episode (keeps the optimizer
  from buying accuracy with unbounded compute).

## Variance & confidence intervals (statistical rigor)

Never report single-run numbers — agent evals have large run-to-run variance.

- Each configuration is evaluated over **N independent seeds** (`N ≥ 20` default; the seed
  controls fixture-agent RNG, baseline init, and detector sampling).
- Report **95% bootstrap CIs** (BCa) over the N seeds for every metric.
- The optimizer (#1) only ever sees the **training partition**; the **held-out partition**
  produces the reported curve; the **frozen canonical bank** is the untouched reference.

## CI gate spec

A candidate config `C` (a code change OR a self-improvement variant) is **promoted** iff,
evaluated on the **held-out set** across **N seeds**:

```
for each protected metric m in {precision, recall, lead_time}:
    lower_CI95(m[C]) >= m[baseline] - tol[m]
and  upper_CI95(fp_rate[C]) <= fp_rate[baseline] + tol[fp]
and  median(cost[C]) <= cost[baseline] * (1 + tol[cost])
```

Any violation → **reject**, and the governance/audit layer records the diff, the per-metric
deltas with CIs, and the rejection rationale. Tolerances `tol[*]` live in a versioned config
so the bar is explicit and auditable. The gate is intentionally **conservative on FP-rate
and asymmetric**: regressions must be *statistically* established (lower-CI), improvements
need not be — we make it hard to ship something worse, easy to ship something clearly better.

## Fixture-agent fault taxonomy (injectable faults)

Each is a parameterized operator over a clean trace; severity is dial-able so L3 monotonicity
holds and #1's generator can search the parameter space.

| Fault | What it injects | Primary feature family hit |
|---|---|---|
| `latency_injection` | added per-step / tail latency | temporal |
| `tool_misselection` | wrong tool chosen for task-class | tool usage |
| `induced_loop` | repeated tool+args cycles | structural |
| `retry_storm` | escalating retry counts | structural / economic |
| `output_degradation` | shorter / elided / lower-quality output | semantic |
| `refusal_spike` | increased refusal/hedge rate | semantic |
| `format_break` | broken output-format adherence | semantic |
| `context_overflow` | prompt-size growth → window pressure | context |
| `context_staleness` | stale RAG/memory hits | context |
| `model_version_swap` | simulated model swap (intended-change test, `03`) | all |
| `cost_blowup` | token/cost explosion per step | economic |
| `branching_explosion` | sub-agent spawn topology blowup | structural |

Compositions (multiple simultaneous faults at varying severity) test attribution and the
aggregator. `model_version_swap` is dual-purpose: with a deploy marker it must be treated as
*intended change* (no alert), without one it must alert — the discriminating test for `03`.

## Riskiest unknown

**Defining `t_fail` (the visible-failure marker) defensibly.** Lead-time is the headline
metric, and it is only as credible as the ground-truth point it measures against. For
synthetic faults `t_fail` is definable by construction (the severity crosses a declared
SLA-violation threshold). For replayed real traces it is harder and partly subjective; we
mitigate by deriving `t_fail` from explicit SLA-style thresholds (latency > X, cost > Y,
quality-proxy < Z) declared *per corpus*, logged, and held constant across configs so
comparisons stay fair even if the absolute number is debatable.
