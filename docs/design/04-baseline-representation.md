# Design Decision #4 — Baseline Representation

> Status: **Decided** (design). v1 of the composable baseline lands in Phase 1.

## Problem

What is the strongest statistical representation of "normal behavior" for a *non-deterministic*
agent that supports (a) streaming/online update, (b) cold-start with little data,
(c) per-task-class conditioning, and (d) cheap drift scoring with calibrated confidence?

The hard part: the baseline itself is non-deterministic. You cannot diff two runs; you must
compare *distributions* over heterogeneous, partially-semantic, streaming signals.

## Options considered

### Option A — One monolithic multivariate model
A single joint density / autoencoder over all features.
- **Pros:** captures cross-feature correlations natively.
- **Cons:** brutal cold-start (needs lots of data before it's trustworthy); poor
  **attribution** ("which signal drifted?" is the product's core output, and a black-box
  joint model hides it); retraining cost; hard to update online per-feature. Rejected.

### Option C — Sequence transformer over the whole trace
A tiny transformer scoring trace likelihood / surprise.
- **Pros:** powerful on sequential structure.
- **Cons:** data-hungry (bad cold-start), expensive scoring, awkward on a 16 GB 4060 Ti for
  high-volume low-stakes scoring, weak attribution. Overkill for v1; a *single* sequence
  detector can live inside Option B instead. Rejected as the whole baseline.

### Option B — Composable per-family estimators fused by a calibrated aggregator (RECOMMENDED)
A set of small, online, per-feature/per-family estimators, each producing a calibrated drift
sub-score, fused by an aggregator into one calibrated score **with attribution**.

## Recommendation — Option B

Baseline is keyed by **(agent, task_class, config_fingerprint)** (see `03`) and composed of:

### Per-family estimators
- **Univariate streaming numerics** (step count, latencies, token counts, retry counts,
  error rates): online distribution estimators with built-in concept-drift detection —
  **ADWIN** and **Page-Hinkley** (`river`). Cheap, streaming, well-understood, give
  change-points for free (feeds `03`).
- **Categorical / distributional** (tool-frequency distribution, tool-selection conditioned
  on task-class, format-adherence): online histograms compared by a two-sample distance
  (population stability index / KS / chi-square).
- **Multivariate & semantic embeddings** (output-embedding centroid + covariance, joint
  feature vectors): kernel two-sample tests — **MMD** via `alibi-detect` — over a sliding
  window vs. the reference window. Track a running centroid + covariance for cheap Mahalanobis
  scoring between full tests.
- **Sequential** (tool-call sequences, loop structure): a small **n-gram / Markov** model
  scored by **perplexity/surprise**. Cheap, interpretable, good cold-start; an optional tiny
  transformer can replace it later if data justifies it (the Option C idea, contained).

This composition gives **attribution by construction** — each sub-score names its family, so
the aggregator can report "tool-selection drifted on agent X at time T by N σ," which is the
product's core output.

### Cold-start — hierarchical / empirical-Bayes shrinkage
With little data, each estimator **shrinks toward a global prior** pooled across task-classes
(and across agents for a brand-new agent), Empirical-Bayes style. As `n` grows, the estimate
relaxes from prior → agent-specific. Concretely:

- maintain a global/pooled prior per feature,
- per-(agent, task_class) estimate = shrinkage(local, prior, weight = f(n)),
- **warmup mode** while `n < n_min`: detectors score and *record* but **suppress alerts**
  (or widen thresholds), surfacing "baseline still warming up (n=…)" rather than crying wolf
  on noise. Ties to the L3 "benign stability" invariant in `02`.

### Calibration & confidence
Raw statistics are not comparable across families, so each sub-score is mapped to a
**calibrated scale**:

- Convert each detector statistic to a **p-value or empirical percentile** against its own
  benign reference window.
- The **aggregator** combines calibrated sub-scores into one drift score whose threshold maps
  to a **target FP-rate** measured on a benign held-out set (so the operating point is a
  *chosen* FP budget, per `02`, not an arbitrary number).
- Prefer **distribution-free calibration** (conformal / empirical-quantile) over Gaussian
  assumptions — agent feature distributions are heavy-tailed and non-normal.
- Confidence shrinks (CIs widen) when `n` is small — the system reports *how sure* it is, not
  just a point score.

### Cheap drift scoring
Per-step scoring is O(features): update online estimators, compute Mahalanobis / percentile
sub-scores (cheap), and run the expensive kernel two-sample test (MMD) only on a **sliding
window cadence**, not every step. This keeps streaming detection within the local-compute
ceiling.

## Why composable beats monolithic here

1. **Attribution** is the product. Composability gives it for free; a joint model hides it.
2. **Cold-start** is graceful per-feature with shrinkage; a joint model is all-or-nothing.
3. **Online update** is native to `river` estimators; joint models need batched retraining.
4. **The ensemble is a self-improvement target** (`01` topology tier) — add/remove/reweight
   detectors per (agent, task_class). That only works if detectors are separable units.

## Riskiest unknown

**Cross-feature correlations a per-family decomposition can miss.** Some drifts only show up
jointly (e.g., latency *and* token count move together in a way neither alone flags). The MMD
detector over the *joint* feature vector is the backstop for this, but its window-cadence and
kernel-bandwidth choice trade off detection power vs. cost/false-positives, and the right
setting is agent-dependent — which is precisely why the topology tier (`01`) must be able to
tune it per (agent, task_class) rather than us hard-coding one global value.
