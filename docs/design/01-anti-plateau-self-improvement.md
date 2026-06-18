# Design Decision #1 — Anti-Plateau Self-Improvement Strategy

> Status: **Decided** (design). Implementation lands in Phase 2 (prompt + skill tiers)
> and Phase 3 (topology tier). This document is the contract the implementation must honor.

## Problem

BELLWETHER improves itself through one closed loop (generator → evaluator → governance
gate → archive) applied to three change-targets: **prompt**, **skill** (drift-signature
library), and **topology** (detector-ensemble composition). The #1 documented failure mode
of self-improving systems is *premature convergence* — "in-benchmark optimization rather
than genuine, open-ended improvement." A naive optimizer will:

1. Hill-climb a scalar reward and lock onto a local optimum.
2. Overfit the fixed eval benchmark (Goodhart) and stop generalizing.
3. Discover a degenerate strategy ("alert on nothing" → FP-rate 0; "alert on everything"
   → recall 1) that wins the scalar but is useless.
4. Collude in co-evolution (fault-generator and detector settle into an easy mutual
   equilibrium that looks like progress but isn't).

The ask: design the strongest mechanism that keeps the loop *genuinely* improving, plus
an honest measurement of "are we still improving vs. overfitting."

## Options considered

### Option A — Multi-objective scalarized hill-climbing
Single archive entry ("best"), weighted-sum objective, accept-if-better.
- **Pros:** trivial to build; fast; low compute.
- **Cons:** every documented failure above applies. Collapses to one point; overfits the
  fixed benchmark; brittle to weight choice. Rejected as the *primary* strategy.

### Option B — Novelty / open-ended search only
Reward behavioral novelty (MAP-Elites / novelty search) without strong quality pressure.
- **Pros:** never plateaus on diversity; great exploration.
- **Cons:** no guarantee the explored detectors are *good*; can wander into useless regions;
  no Pareto pressure toward the metrics that decide adoption. Rejected as standalone.

### Option C — Quality-Diversity archive + adversarial co-evolution + Pareto + plateau-triggered exploration (RECOMMENDED)
A principled combination of four mechanisms, each covering a specific failure mode of the
others. This is the recommendation.

## Recommendation — Option C, four composed mechanisms

### 1. Quality-Diversity archive (MAP-Elites) — defeats single-point collapse
Maintain an archive of detector/ensemble variants binned by a **behavioral descriptor**,
not by scalar score. Descriptor dimensions (cheap to compute, behaviorally meaningful):

- `detector_family_mix` — which feature families the ensemble weights (structural / tool /
  temporal / economic / semantic / context), bucketed.
- `cost_bucket` — eval-time compute / token cost, bucketed (cheap → expensive).
- `fp_rate_bucket` — operating point on the FP axis (conservative → trigger-happy).
- `latency_regime` — streaming-light vs. batch-heavy detection.

Each archive cell keeps the **Pareto-best** variant *for that behavioral niche*. The
archive is the unit of search: mutation samples a parent from a cell, the gate places the
child in the cell its descriptor lands in. This structurally prevents collapse onto one
solution — we always retain behaviorally distinct strategies, which is also exactly the
material the live system needs for *per-(agent, task-class)* specialization.

### 2. Adversarial co-evolution of a fault generator — defeats benchmark saturation
A **fault generator** evolves *new* synthetic drift scenarios in competition with the
detector archive (self-play / moving target). The detector can never saturate a fixed
benchmark because the benchmark grows. The fault generator is rewarded for producing faults
that are **(a) valid, (b) realistic, and (c) hard for the current archive but not impossible
in principle.**

**Anti-collusion guardrails (this is the critical, easy-to-get-wrong part):**

- **Validity oracle.** Every generated fault must pass an independent validity check: a
  *reference oracle detector with full ground-truth information* (it can see the injected
  fault label and clean baseline) must still be able to detect it. This forbids the
  degenerate "undetectable noise" fault that would let the generator win by cheating.
- **Realism filter.** Faults must be expressible as one of the taxonomy operators in the
  eval harness (`docs/design/02-eval-harness.md`) applied to *replayed real traces*, not
  free-form noise. The generator searches *parameters and compositions* of real fault
  operators, not arbitrary distributions.
- **Frozen canonical bank.** A held-out, human-curated bank of canonical faults the
  generator may **never** modify or see during optimization. Detector fitness on this frozen
  bank is the anti-collusion tripwire: if archive fitness on co-evolved faults rises while
  fitness on the frozen bank stalls or falls, that is *collusion/overfitting*, and the run
  is flagged and exploration is boosted (mechanism 4).

### 3. Multi-objective Pareto fitness — makes every degenerate strategy lose
Fitness is a vector, never a scalar sum:
`(precision, recall, lead_time, −fp_rate, −cost)`. Promotion uses **Pareto dominance within
a behavioral cell**, not a weighted sum. This is what kills the degenerate optima:

- "Alert on nothing" → recall 0 → Pareto-dominated by anything with recall > 0 at equal
  FP-rate. Loses.
- "Alert on everything" → fp_rate maximal → Pareto-dominated on the −fp_rate axis. Loses.

There is no scalar weight to game; both axes are first-class objectives simultaneously.

### 4. Explicit plateau detection + exploration boost — defeats stalling
Monitor the **held-out best curve** (best Pareto hypervolume on the rotating held-out eval,
mechanism in §"Are we still improving?"). Declare a plateau when the hypervolume slope over
a window of `W` iterations is below `ε` *and* the bootstrap CIs of the last `K` iterations
overlap (no statistically distinguishable gain). On plateau:

1. Increase mutation rate / step size.
2. Random-restart from under-explored archive cells (novelty bias).
3. Inject new fault *types* from the generator's frontier into the eval.
4. If still flat after `M` boosts → log "converged" to the audit layer and stop burning
   compute (honest convergence is a valid terminal state, not a failure to hide).

## "Are we still improving vs. overfitting?" — the honest measurement

This is the deliverable the brief calls load-bearing. We never trust training-set numbers.

- **Three disjoint data partitions:**
  - *Training faults* — the optimizer may use freely.
  - *Held-out faults* — never seen by the optimizer; used only to plot the real curve.
  - *Frozen canonical bank* — never seen, never rotated; the absolute reference and
    collusion tripwire.
- **Rotation.** The held-out partition is periodically rotated/refreshed from the replay
  corpus so the optimizer cannot indirectly overfit it through repeated gate decisions.
  Rotation events are logged to the audit layer.
- **Improvement signal = held-out Pareto hypervolume** over iterations, with **bootstrap
  confidence intervals** computed across `N` independent eval seeds (per §6 statistical
  rigor — never single-run numbers).
- **Overfitting alarm:** training hypervolume rises while held-out hypervolume is flat/falls
  → overfitting; auto-reject the lineage and trigger exploration boost.
- **Genuine-improvement criterion:** held-out hypervolume lower-CI strictly exceeds the
  previous accepted best's upper-CI. Anything weaker is treated as noise, not progress.

## Riskiest unknown

**Productive vs. degenerate co-evolution.** The arms race is the strongest anti-plateau
lever *and* the most likely thing to fail silently (collusion → fake progress). The
validity-oracle + frozen-canonical-bank tripwire is the mitigation, but its sensitivity
(how fast it catches collusion) is unproven and must be validated empirically before the
topology tier is allowed to self-modify in Phase 3. Until then, topology changes are
human-confirmed through the governance gate.

## How this wires to the rest

- **Generator/Evaluator/Gate/Archive** are the shared substrate (brief §7); this document
  specifies the *search dynamics* that ride on top.
- **Evaluator** = the eval harness in `02-eval-harness.md` (same CI gate).
- **Governance gate** logs every promotion/rejection with diff + metric deltas + rationale
  to the append-only audit layer — including plateau declarations, rotation events, and
  collusion-tripwire firings.
