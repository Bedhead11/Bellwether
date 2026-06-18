"""The drift scoring engine: baseline + detectors + aggregator + sustained-alert + calibration.

``DriftScorer`` scores a run's observation sequence against its baseline and produces a
:class:`DriftReport`. A run alerts if *any* detection track fires:

- **critical** — a single step whose generic drift score alone clears the critical threshold
  (fast path for overwhelming drift). Yields ``t_alert``.
- **step (k-of-w)** — at least ``k`` of the last ``w`` steps exceed the step threshold. k-of-w
  (not k-*consecutive*) catches faults that perturb only alternating steps while still
  suppressing single-point noise. Yields ``t_alert`` and therefore **lead-time**.
- **run-summary** — the run-summary observation vs the run threshold; catches purely aggregate
  drift no single step reveals.
- **signature tracks** (skill tier, optional) — one focused, direction-filtered track per drift
  signature; a single step matching the signature's sharp focused score fires it.

Every track is **calibrated independently** on a benign hold-out to its own share of the
false-positive budget, because the tracks have different score distributions. The total budget
is fixed and split across tracks, so adding signatures cannot blow it — the anti-collapse guard.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from bellwether.baseline import Baseline, BaselineManager
from bellwether.detect.aggregator import aggregate, family_contributions
from bellwether.detect.detectors import score_categorical, score_numeric
from bellwether.detect.report import (
    Alert,
    DriftReport,
    ObservationScore,
    SignatureProvider,
    Thresholds,
)
from bellwether.features import FeatureObservation, extract_observations
from bellwether.schema import AgentRun

# Sentinel "never a hit / never alerts at any threshold" (real drift scores live in [0, 1]).
_NEVER = -1.0

# One step's contribution to a track: (step_index, score, warmup).
_StepPoint = tuple[int, float, bool]


@dataclass(frozen=True)
class ScoringConfig:
    """Tunables for the detection engine (a topology-tier self-improvement target later)."""

    min_samples: int = 30  # per-feature warmup threshold
    laplace_alpha: float = 1.0  # categorical smoothing
    window_w: int = 3  # sliding window size for the k-of-w sustained rule
    min_hits_k: int = 2  # required hits within the window
    # Shares of the FP budget across the generic tracks (critical / sustained-step / run);
    # summed and normalized so they need not add to exactly 1.
    crit_budget_frac: float = 0.34
    step_budget_frac: float = 0.33
    run_budget_frac: float = 0.33
    # Share of the *total* budget reserved for the signature tracks collectively (split equally).
    signature_budget_frac: float = 0.4
    default_threshold: float = 0.999


# --- track primitives (operate on a per-step score sequence) -----------------------------


def _critical_scan(seq: Sequence[_StepPoint], threshold: float) -> tuple[bool, int | None]:
    for idx, score, warm in seq:
        if not warm and score >= threshold:
            return (True, idx)
    return (False, None)


def _critical_level(seq: Sequence[_StepPoint]) -> float:
    return max((s for _, s, w in seq if not w), default=_NEVER)


def _sustained_scan(
    seq: Sequence[_StepPoint], threshold: float, w: int, k: int
) -> tuple[bool, int | None]:
    """Fire at the step where k-of-w becomes satisfied (the honest, operational alert time)."""
    window: list[bool] = []
    for idx, score, warm in seq:
        window.append((not warm) and score >= threshold)
        if len(window) > w:
            window.pop(0)
        if sum(window) >= k:
            return (True, idx)
    return (False, None)


def _sustained_level(seq: Sequence[_StepPoint], w: int, k: int) -> float:
    level = _NEVER
    window: list[float] = []
    for _, score, warm in seq:
        window.append(_NEVER if warm else score)
        if len(window) > w:
            window.pop(0)
        if len(window) >= k:
            level = max(level, sorted(window, reverse=True)[k - 1])
    return level


class DriftScorer:
    """Scores runs against baselines and raises calibrated drift alerts."""

    def __init__(
        self, config: ScoringConfig | None = None, library: SignatureProvider | None = None
    ) -> None:
        self.config = config or ScoringConfig()
        self.library = library

    # --- observation-level scoring -------------------------------------------------------

    def _score_observation(self, obs: FeatureObservation, baseline: Baseline) -> ObservationScore:
        subs = []
        for nobs in obs.numerics:
            s = score_numeric(nobs, baseline, min_samples=self.config.min_samples)
            if s is not None:
                subs.append(s)
        for cobs in obs.categoricals:
            s = score_categorical(
                cobs,
                baseline,
                min_samples=self.config.min_samples,
                laplace_alpha=self.config.laplace_alpha,
            )
            if s is not None:
                subs.append(s)
        base = aggregate(obs.kind, obs.step_index, subs)
        if self.library is not None and not base.warmup:
            sig_scores = self.library.signature_scores(base.sub_scores)
            return ObservationScore(
                base.kind,
                base.step_index,
                base.drift_score,
                base.sub_scores,
                base.warmup,
                signature_scores=sig_scores,
            )
        return base

    def score_run(self, run: AgentRun, baseline: Baseline) -> list[ObservationScore]:
        """Score every observation of a run (per-step + run-summary) against the baseline."""
        return [self._score_observation(o, baseline) for o in extract_observations(run)]

    # --- sequence builders --------------------------------------------------------------

    @staticmethod
    def _generic_step_seq(scores: Sequence[ObservationScore]) -> list[_StepPoint]:
        return [
            (o.step_index if o.step_index is not None else -1, o.drift_score, o.warmup)
            for o in scores
            if o.kind == "step"
        ]

    @staticmethod
    def _signature_step_seq(scores: Sequence[ObservationScore], name: str) -> list[_StepPoint]:
        return [
            (
                o.step_index if o.step_index is not None else -1,
                o.signature_scores.get(name, 0.0),
                o.warmup,
            )
            for o in scores
            if o.kind == "step"
        ]

    @staticmethod
    def _run_summary_score(scores: Sequence[ObservationScore]) -> ObservationScore | None:
        return next((o for o in scores if o.kind == "run_summary"), None)

    # --- calibration levels --------------------------------------------------------------

    def critical_level(self, scores: Sequence[ObservationScore]) -> float:
        return _critical_level(self._generic_step_seq(scores))

    def step_alert_level(self, scores: Sequence[ObservationScore]) -> float:
        return _sustained_level(
            self._generic_step_seq(scores), self.config.window_w, self.config.min_hits_k
        )

    def run_summary_level(self, scores: Sequence[ObservationScore]) -> float:
        rs = self._run_summary_score(scores)
        if rs is None or rs.warmup:
            return _NEVER
        return rs.drift_score

    def signature_level(self, scores: Sequence[ObservationScore], name: str) -> float:
        return _critical_level(self._signature_step_seq(scores, name))

    # --- run-level verdict ---------------------------------------------------------------

    def evaluate(
        self, run: AgentRun, baseline: Baseline, thresholds: Thresholds | None = None
    ) -> DriftReport:
        d = self.config.default_threshold
        thr = thresholds or Thresholds(critical=d, step=d, run=d)
        scores = self.score_run(run, baseline)
        gen_seq = self._generic_step_seq(scores)

        # Generic step-level tracks.
        crit_trig, t_crit = _critical_scan(gen_seq, thr.critical)
        sust_trig, t_sust = _sustained_scan(
            gen_seq, thr.step, self.config.window_w, self.config.min_hits_k
        )
        step_times = [t for t in (t_crit, t_sust) if t is not None]
        triggered = crit_trig or sust_trig

        # Signature tracks (skill tier).
        fired_signature: str | None = None
        if self.library is not None:
            for name in self.library.names:
                sig_thr = thr.signatures.get(name, d)
                sig_trig, t_sig = _critical_scan(self._signature_step_seq(scores, name), sig_thr)
                if sig_trig:
                    triggered = True
                    if t_sig is not None:
                        step_times.append(t_sig)
                    if fired_signature is None:
                        fired_signature = name

        t_alert = min(step_times) if step_times else None

        rs = self._run_summary_score(scores)
        if not triggered and rs is not None and not rs.warmup and rs.drift_score >= thr.run:
            triggered, t_alert = True, None  # detected only at run end

        # Attribution: the most-surprising sub-score across the whole run.
        primary = None
        for obs in scores:
            if obs.primary is not None and (primary is None or obs.primary.score > primary.score):
                primary = obs.primary

        alert = Alert(
            triggered=triggered,
            step_index=t_alert,
            drift_score=(primary.score if primary else 0.0) if triggered else 0.0,
            primary_family=primary.family if (triggered and primary) else None,
            primary_feature=primary.feature if (triggered and primary) else None,
            detail=primary.detail if (triggered and primary) else "",
            signature=fired_signature if triggered else None,
        )
        return DriftReport(
            run_id=run.run_id,
            agent_id=run.agent_id,
            baseline_key=run.baseline_key,
            thresholds=thr,
            observation_scores=tuple(scores),
            alert=alert,
            family_contributions=family_contributions(scores),
        )

    def evaluate_with_manager(
        self, run: AgentRun, manager: BaselineManager, thresholds: Thresholds | None = None
    ) -> DriftReport:
        baseline = manager.baseline_for(run)
        if baseline is None:
            raise KeyError(f"no baseline learned for {run.baseline_key}")
        return self.evaluate(run, baseline, thresholds)


def _smallest_threshold_within_budget(levels: list[float], target_fp: float) -> float:
    """Smallest threshold whose benign FP-rate (fraction of levels ≥ T) ≤ target.

    FP-rate is monotone non-increasing in T, so the first qualifying ascending candidate is the
    most sensitive operating point that still respects the budget.
    """
    if not levels:
        return 1.0 + 1e-9
    n = len(levels)
    candidates = sorted({lvl for lvl in levels if lvl > _NEVER})
    ceiling = (max(candidates) if candidates else 0.0) + 1e-9
    for t in candidates:
        fp = sum(1 for lvl in levels if lvl >= t) / n
        if fp <= target_fp:
            return t
    return ceiling


def calibrate_threshold(
    scorer: DriftScorer,
    baseline: Baseline,
    benign_runs: Iterable[AgentRun],
    *,
    target_fp_rate: float,
) -> Thresholds:
    """Calibrate every track's threshold so the combined benign FP-rate stays within budget.

    The budget is split (union bound): a configurable share to the signature tracks collectively
    (split equally), the remainder to the generic tracks by their configured fractions. Each
    track's threshold is the most sensitive value respecting its share.
    """
    runs = list(benign_runs)
    scored = [scorer.score_run(r, baseline) for r in runs]
    cfg = scorer.config

    names = scorer.library.names if scorer.library is not None else []
    sig_frac = cfg.signature_budget_frac if names else 0.0
    generic_total = target_fp_rate * (1.0 - sig_frac)
    gen_norm = cfg.crit_budget_frac + cfg.step_budget_frac + cfg.run_budget_frac

    crit_levels = [scorer.critical_level(s) for s in scored]
    step_levels = [scorer.step_alert_level(s) for s in scored]
    run_levels = [scorer.run_summary_level(s) for s in scored]

    signatures: dict[str, float] = {}
    if names:
        per_sig_target = target_fp_rate * sig_frac / len(names)
        for name in names:
            sig_levels = [scorer.signature_level(s, name) for s in scored]
            signatures[name] = _smallest_threshold_within_budget(sig_levels, per_sig_target)

    return Thresholds(
        critical=_smallest_threshold_within_budget(
            crit_levels, generic_total * cfg.crit_budget_frac / gen_norm
        ),
        step=_smallest_threshold_within_budget(
            step_levels, generic_total * cfg.step_budget_frac / gen_norm
        ),
        run=_smallest_threshold_within_budget(
            run_levels, generic_total * cfg.run_budget_frac / gen_norm
        ),
        signatures=signatures,
    )
