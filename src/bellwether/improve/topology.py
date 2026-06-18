"""Topology-tier self-improvement: evolve the detector-ensemble configuration.

The "topology" is the ensemble's shape: window size and hit count of the sustained rule, how the
false-positive budget is split across the critical / step / run / signature tracks, and the
warmup threshold. This module searches that space with **MAP-Elites** quality-diversity (design
doc 01): rather than hill-climbing one scalar, it keeps an archive of the best configuration in
each *behavioral niche*, so the search keeps exploring structurally distinct ensembles instead of
collapsing onto one local optimum.

Each candidate is evaluated on the held-out benchmark (the eval harness) and accepted into its
niche only if it Pareto-improves — a higher F1 with no false-positive or precision regression —
the same multi-objective discipline that makes degenerate strategies lose. A plateau detector on
the global-best curve triggers exploration boosts, and every accepted change is audited.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from bellwether.detect import ScoringConfig
from bellwether.detect.report import SignatureProvider
from bellwether.eval.harness import BenchmarkConfig, run_benchmark
from bellwether.fixtures import FixtureAgent
from bellwether.governance import AuditLog, EventType
from bellwether.improve.gate import MetricVector
from bellwether.improve.loop import metric_vector_from_report


def mutate_config(config: ScoringConfig, rng: random.Random) -> ScoringConfig:
    """Perturb one topology parameter, keeping the result valid."""
    choice = rng.choice(["window", "hits", "budget", "signature_budget", "min_samples"])
    if choice == "window":
        w = min(6, max(2, config.window_w + rng.choice([-1, 1])))
        k = min(config.min_hits_k, w)
        return replace(config, window_w=w, min_hits_k=max(1, k))
    if choice == "hits":
        k = min(config.window_w, max(1, config.min_hits_k + rng.choice([-1, 1])))
        return replace(config, min_hits_k=k)
    if choice == "budget":
        # Perturb one generic-track fraction and renormalize the three.
        fracs = [config.crit_budget_frac, config.step_budget_frac, config.run_budget_frac]
        i = rng.randrange(3)
        fracs[i] = max(0.05, fracs[i] + rng.choice([-0.1, 0.1]))
        total = sum(fracs)
        fracs = [f / total for f in fracs]
        return replace(
            config, crit_budget_frac=fracs[0], step_budget_frac=fracs[1], run_budget_frac=fracs[2]
        )
    if choice == "signature_budget":
        sb = min(0.6, max(0.0, config.signature_budget_frac + rng.choice([-0.1, 0.1])))
        return replace(config, signature_budget_frac=sb)
    ms = min(60, max(15, config.min_samples + rng.choice([-5, 5])))
    return replace(config, min_samples=ms)


def descriptor(config: ScoringConfig) -> tuple[int, str]:
    """Behavioral descriptor for MAP-Elites binning: (window size, dominant generic track)."""
    fracs = {
        "critical": config.crit_budget_frac,
        "step": config.step_budget_frac,
        "run": config.run_budget_frac,
    }
    dominant = max(fracs, key=lambda k: fracs[k])
    return (config.window_w, dominant)


@dataclass(frozen=True)
class Elite:
    config: ScoringConfig
    metrics: MetricVector


@dataclass(frozen=True)
class TopologyStep:
    iteration: int
    accepted: bool
    descriptor: tuple[int, str]
    best_f1: float
    best_recall: float
    best_fp: float
    plateau: bool = False


@dataclass
class TopologyHistory:
    steps: list[TopologyStep]
    archive: dict[tuple[int, str], Elite]
    best: Elite

    def best_f1_curve(self) -> list[float]:
        return [s.best_f1 for s in self.steps]

    def render(self) -> str:
        lines = [
            "BELLWETHER topology self-improvement (MAP-Elites over ensemble configs)",
            "  iter  accepted  niche                 best F1   best recall   best fp",
            "  " + "-" * 68,
        ]
        for s in self.steps:
            acc = "  yes " if s.accepted else "  no  "
            plat = "  [plateau]" if s.plateau else ""
            niche = f"w={s.descriptor[0]},{s.descriptor[1]}"
            lines.append(
                f"  {s.iteration:>3}  {acc}    {niche:<20} {s.best_f1:.3f}    "
                f"{s.best_recall:.3f}        {s.best_fp:.3f}{plat}"
            )
        b = self.best.config
        lines += [
            "  " + "-" * 68,
            f"  archive niches filled: {len(self.archive)}",
            f"  best config: window_w={b.window_w} min_hits_k={b.min_hits_k} "
            f"budget(crit/step/run)={b.crit_budget_frac:.2f}/{b.step_budget_frac:.2f}/"
            f"{b.run_budget_frac:.2f} sig_budget={b.signature_budget_frac:.2f}",
            f"  best F1={self.best.metrics.f1[0]:.3f} recall={self.best.metrics.recall[0]:.3f} "
            f"fp={self.best.metrics.fp_rate[0]:.3f}",
        ]
        return "\n".join(lines)


@dataclass
class TopologySearch:
    """MAP-Elites search over detector-ensemble configurations, eval-gated and audited."""

    agent: FixtureAgent
    eval_config: BenchmarkConfig
    audit: AuditLog
    library: SignatureProvider | None = None
    fp_budget: float = 0.02
    fp_tol: float = 0.015
    precision_tol: float = 0.06
    seed: int = 0
    plateau_window: int = 5
    plateau_epsilon: float = 0.005

    def _evaluate(self, config: ScoringConfig) -> MetricVector:
        cfg = replace(self.eval_config, scoring=config, library=self.library)
        return metric_vector_from_report(run_benchmark(self.agent, cfg))

    def _accepts(self, incumbent: Elite | None, candidate: MetricVector) -> bool:
        """Pareto acceptance within a niche: higher F1, no FP/precision regression."""
        fp_ok = candidate.fp_rate[2] <= self.fp_budget + self.fp_tol
        if not fp_ok:
            return False
        if incumbent is None:
            return True
        precision_ok = candidate.precision[1] >= incumbent.metrics.precision[0] - self.precision_tol
        return candidate.f1[0] > incumbent.metrics.f1[0] and precision_ok

    def _global_best(self, archive: dict[tuple[int, str], Elite]) -> Elite:
        return max(archive.values(), key=lambda e: e.metrics.f1[0])

    def run(
        self, iterations: int = 20, seed_config: ScoringConfig | None = None
    ) -> TopologyHistory:
        rng = random.Random(self.seed)
        base = seed_config or self.eval_config.scoring_config()
        archive: dict[tuple[int, str], Elite] = {}
        steps: list[TopologyStep] = []

        # Seed the archive with the starting configuration.
        base_metrics = self._evaluate(base)
        archive[descriptor(base)] = Elite(base, base_metrics)
        best = archive[descriptor(base)]

        best_f1_history = [best.metrics.f1[0]]
        mutation_burst = 1

        for it in range(1, iterations + 1):
            # Sample a parent from the archive (random niche) and mutate.
            parent = rng.choice(list(archive.values()))
            child_cfg = parent.config
            for _ in range(mutation_burst):
                child_cfg = mutate_config(child_cfg, rng)

            child_metrics = self._evaluate(child_cfg)
            cell = descriptor(child_cfg)
            incumbent = archive.get(cell)
            accepted = self._accepts(incumbent, child_metrics)
            if accepted:
                archive[cell] = Elite(child_cfg, child_metrics)
                best = self._global_best(archive)
                self.audit.append(
                    EventType.PROMOTION,
                    f"topology:{cell[0]},{cell[1]}",
                    f"accepted ensemble config; F1={child_metrics.f1[0]:.3f}",
                    {
                        "config": {
                            "window_w": child_cfg.window_w,
                            "min_hits_k": child_cfg.min_hits_k,
                            "crit": round(child_cfg.crit_budget_frac, 3),
                            "step": round(child_cfg.step_budget_frac, 3),
                            "run": round(child_cfg.run_budget_frac, 3),
                            "sig": round(child_cfg.signature_budget_frac, 3),
                        }
                    },
                )

            # Plateau detection on the global-best F1; boost exploration when flat.
            best_f1_history.append(best.metrics.f1[0])
            plateau = False
            if len(best_f1_history) > self.plateau_window:
                gain = best_f1_history[-1] - best_f1_history[-self.plateau_window - 1]
                if gain < self.plateau_epsilon:
                    plateau = True
                    mutation_burst = min(3, mutation_burst + 1)  # explore harder
                    self.audit.append(
                        EventType.PLATEAU,
                        "topology",
                        f"global-best F1 gain < {self.plateau_epsilon} over {self.plateau_window} iters; "
                        f"mutation_burst={mutation_burst}",
                        {"best_f1": best.metrics.f1[0]},
                    )
                else:
                    mutation_burst = 1

            steps.append(
                TopologyStep(
                    it,
                    accepted,
                    cell,
                    best.metrics.f1[0],
                    best.metrics.recall[0],
                    best.metrics.fp_rate[0],
                    plateau,
                )
            )

        return TopologyHistory(steps, archive, best)
