"""The drift scoring engine: baseline + detectors + aggregator + sustained-alert + calibration.

``DriftScorer`` scores a run's observation sequence against its baseline and produces a
:class:`DriftReport`. Two detection tracks run in parallel and a run alerts if *either* fires:

- **step track** — a *k-of-w* sustained rule over per-step observations: at least ``k`` of the
  last ``w`` steps exceed the step threshold. k-of-w (rather than k-*consecutive*) is essential
  because several faults only perturb alternating steps (e.g. a cost blowup hits LLM steps but
  not the tool steps between them); it still suppresses single-point noise. This track yields
  ``t_alert`` and therefore **lead-time**.
- **run-summary track** — the single run-summary observation vs the run threshold; catches
  purely aggregate drifts (total cost, depth) that no single step reveals.

Each track is **calibrated independently** on a benign hold-out to its own share of the
false-positive budget, because the two tracks have different score distributions (the run
summary aggregates ~20 features, so its multiplicity penalty differs from a step's). A single
global threshold would let one track's noise suppress the other's signal.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from bellwether.baseline import Baseline, BaselineManager
from bellwether.detect.aggregator import aggregate, family_contributions
from bellwether.detect.detectors import score_categorical, score_numeric
from bellwether.detect.report import Alert, DriftReport, ObservationScore, Thresholds
from bellwether.features import FeatureObservation, extract_observations
from bellwether.schema import AgentRun

# Sentinel "never a hit / never alerts at any threshold" (real drift scores live in [0, 1]).
_NEVER = -1.0


@dataclass(frozen=True)
class ScoringConfig:
    """Tunables for the detection engine (a topology-tier self-improvement target later)."""

    min_samples: int = 30  # per-feature warmup threshold
    laplace_alpha: float = 1.0  # categorical smoothing
    window_w: int = 3  # sliding window size for the k-of-w sustained rule
    min_hits_k: int = 2  # required hits within the window
    # Shares of the FP budget across the three tracks (critical / sustained-step / run); summed
    # and normalized, so they need not add to exactly 1.
    crit_budget_frac: float = 0.34
    step_budget_frac: float = 0.33
    run_budget_frac: float = 0.33
    default_threshold: float = 0.999


class DriftScorer:
    """Scores runs against baselines and raises calibrated drift alerts."""

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig()

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
        return aggregate(obs.kind, obs.step_index, subs)

    def score_run(self, run: AgentRun, baseline: Baseline) -> list[ObservationScore]:
        """Score every observation of a run (per-step + run-summary) against the baseline."""
        return [self._score_observation(o, baseline) for o in extract_observations(run)]

    # --- critical track: single overwhelming step ---------------------------------------

    @staticmethod
    def _critical_alert(
        scores: Sequence[ObservationScore], threshold: float
    ) -> tuple[bool, int | None]:
        """Fire at the first step whose drift score alone clears the critical threshold."""
        for obs in scores:
            if obs.kind != "step" or obs.warmup:
                continue
            if obs.drift_score >= threshold:
                return (True, obs.step_index)
        return (False, None)

    @staticmethod
    def critical_level(scores: Sequence[ObservationScore]) -> float:
        """Largest critical threshold at which a single step still alerts (max step score)."""
        level = _NEVER
        for obs in scores:
            if obs.kind == "step" and not obs.warmup:
                level = max(level, obs.drift_score)
        return level

    # --- step track: k-of-w sustained rule ----------------------------------------------

    def _step_sustained(
        self, scores: Sequence[ObservationScore], threshold: float
    ) -> tuple[bool, int | None]:
        """Return (triggered, t_alert).

        ``t_alert`` is the step at which the k-of-w rule becomes satisfied — i.e. when the alert
        would actually fire and an operator would be notified. We deliberately do *not* back-date
        it to the earliest hit in the window: that would overstate lead-time. This is the honest,
        operationally meaningful detection time.
        """
        w = self.config.window_w
        k = self.config.min_hits_k
        window: list[bool] = []  # is_hit for the last w steps
        for obs in scores:
            if obs.kind != "step":
                continue
            is_hit = (not obs.warmup) and obs.drift_score >= threshold
            window.append(is_hit)
            if len(window) > w:
                window.pop(0)
            if sum(window) >= k:
                return (True, obs.step_index)
        return (False, None)

    def step_alert_level(self, scores: Sequence[ObservationScore]) -> float:
        """Largest step threshold at which the step track still alerts (for calibration).

        Within any window of ``w`` steps, the highest threshold that still yields ``k`` hits is
        the ``k``-th largest drift score in the window; the run's level is the max over windows.
        """
        w = self.config.window_w
        k = self.config.min_hits_k
        level = _NEVER
        window: list[float] = []
        for obs in scores:
            if obs.kind != "step":
                continue
            window.append(_NEVER if obs.warmup else obs.drift_score)
            if len(window) > w:
                window.pop(0)
            if len(window) >= k:
                kth_largest = sorted(window, reverse=True)[k - 1]
                level = max(level, kth_largest)
        return level

    # --- run-summary track ---------------------------------------------------------------

    @staticmethod
    def _run_summary_score(scores: Sequence[ObservationScore]) -> ObservationScore | None:
        return next((o for o in scores if o.kind == "run_summary"), None)

    def run_summary_level(self, scores: Sequence[ObservationScore]) -> float:
        rs = self._run_summary_score(scores)
        if rs is None or rs.warmup:
            return _NEVER
        return rs.drift_score

    # --- run-level verdict ---------------------------------------------------------------

    def evaluate(
        self, run: AgentRun, baseline: Baseline, thresholds: Thresholds | None = None
    ) -> DriftReport:
        d = self.config.default_threshold
        thr = thresholds or Thresholds(critical=d, step=d, run=d)
        scores = self.score_run(run, baseline)

        # Step-level detection: a single critical step OR a sustained k-of-w streak. The run's
        # step alert time is the earlier of the two.
        crit_trig, t_crit = self._critical_alert(scores, thr.critical)
        sust_trig, t_sust = self._step_sustained(scores, thr.step)
        step_times = [t for t in (t_crit, t_sust) if t is not None]
        triggered = crit_trig or sust_trig
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
    """Calibrate per-track thresholds so the combined benign FP-rate stays within budget.

    The budget is split between the two tracks (union bound) per
    ``ScoringConfig.step_budget_frac``; each track's threshold is the most sensitive value
    respecting its share.
    """
    runs = list(benign_runs)
    scored = [scorer.score_run(r, baseline) for r in runs]
    crit_levels = [scorer.critical_level(s) for s in scored]
    step_levels = [scorer.step_alert_level(s) for s in scored]
    run_levels = [scorer.run_summary_level(s) for s in scored]

    cfg = scorer.config
    total = cfg.crit_budget_frac + cfg.step_budget_frac + cfg.run_budget_frac
    crit_target = target_fp_rate * cfg.crit_budget_frac / total
    step_target = target_fp_rate * cfg.step_budget_frac / total
    run_target = target_fp_rate * cfg.run_budget_frac / total
    return Thresholds(
        critical=_smallest_threshold_within_budget(crit_levels, crit_target),
        step=_smallest_threshold_within_budget(step_levels, step_target),
        run=_smallest_threshold_within_budget(run_levels, run_target),
    )
