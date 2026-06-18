# BELLWETHER Benchmark

A reproducible, openly-published benchmark for behavioral-drift detection in AI agents. The
ground truth is **exact** because faults are *injected* by a controllable fixture harness, so
precision / recall / lead-time are measured, not estimated. Every number is reported over **N
independent seeds with 95% bootstrap confidence intervals** — never a single run (brief §6).

## Methodology

- **Fixture agents** emit canonical `AgentRun` traces; a **fault injector** applies a
  parameterized, severity-dialed fault. Because the fault is known, labels are exact.
- Per seed, **disjoint** splits: train (learn the baseline) · calibrate (set per-track
  thresholds to a 2% false-positive budget on benign hold-out) · eval (benign + faulted, scored).
  Nothing the threshold saw is scored.
- **Metrics** (formulas in [`design/02-eval-harness.md`](design/02-eval-harness.md)): precision,
  *timely* recall (detected at or before the visible-failure step), detection-rate (detected at
  all), F1, false-positive rate, lead-time (steps before visible failure). `t_alert` is the step
  at which the alert rule is actually satisfied — never back-dated — so lead-time is honest.
- **CIs**: percentile bootstrap across the N per-seed metric values.

Reproduce: `uv run python examples/benchmark.py` (or `bellwether benchmark`).

## Single-agent detection (20 seeds, FP budget 2%)

| metric | value (95% CI) |
|---|---|
| precision | 0.973 [0.962, 0.984] |
| timely recall | 0.730 [0.714, 0.748] |
| detection rate | 0.878 [0.859, 0.897] |
| F1 | 0.833 [0.822, 0.845] |
| **false-positive rate** | **0.013 [0.007, 0.018]** (budget 0.02) |
| median lead-time | ~2.2 steps before visible failure |

Per-fault *timely* recall: `latency_injection` 0.99 · `induced_loop` 1.00 · `retry_storm` 1.00 ·
`tool_misselection` 0.91 · `cost_blowup` 0.41 · `output_degradation` 0.07.

The last two are honestly hard for the *static* detector — subtle, single-direction drifts on
alternating steps against a tight visible-failure line. They are exactly the headroom the
self-improvement engine fills.

## Self-improvement (skill tier)

A drift signature is *mined* from labeled incidents and promoted only if the held-out benchmark
shows a genuine recall gain with no FP or precision regression (the governance gate). Result:

| | held-out timely recall | FP-rate | precision |
|---|---|---|---|
| generic baseline | 0.70 | 0.013 | 0.97 |
| + mined `cost_blowup` signature | **0.83 (+0.13)** | 0.015 (within budget) | held |

The `output_degradation` candidate is **rejected** (it would raise FP) and an already-solved fault
is rejected for no gain — the gate working in both directions. Reproduce:
`uv run python examples/self_improve_demo.py`.

## Topology self-improvement (MAP-Elites)

A quality-diversity search over the detector-ensemble configuration explores diverse behavioral
niches, the plateau detector fires, and it **confirms the hand-tuned default is near-optimal** in
this space — a rigorous negative result that validates the anti-plateau machinery (the brief
explicitly values negative results). Reproduce: `uv run python examples/topology_demo.py`.

## Multi-agent coordination drift

A coordination feature family (role balance, handoffs, ping-pong, cross-agent loops) feeds the
same machinery. Three coordination pathologies are detected and attributed via skill-tier
signatures **even though each agent's per-step behavior stays locally normal**, at a low benign
false-positive rate (≤ 0.12 in the test). Reproduce:
`uv run python examples/coordination_demo.py`.

## Honest limitations

- `output_degradation` and `cost_blowup` are hard for the static detector (the skill tier fixes
  `cost_blowup`; `output_degradation`'s mined signature is correctly rejected for raising FP —
  better semantic features are future work).
- Distinguishing a *pathological* emergent loop from a *healthy* cyclic multi-agent workflow needs
  a richer signal and is left as future work rather than faked.
- All results are on synthetic fixture traces (eval layer L1). The replay-corpus layer (L2, real
  traces) is designed but not yet populated — until then, treat absolute numbers as a controlled
  lower bound on realism, not a production guarantee.
