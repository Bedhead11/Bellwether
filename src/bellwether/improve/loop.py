"""The self-improvement loop: generator → evaluator → governance gate → archive.

Phase 2 wires the skill tier through the shared substrate (brief §7):

1. **Generator** mines a candidate drift signature from training incidents.
2. **Evaluator** scores the candidate library on a held-out benchmark (the eval harness) →
   a multi-objective :class:`MetricVector` with bootstrap CIs.
3. **Gate** promotes or rejects, logging the diff + metric deltas + rationale to the audit log.
4. **Archive** keeps the full lineage of every variant (enables rollback + open-ended search).

**Anti-plateau (design doc 01):** signatures are kept in a quality-diversity archive keyed by
the drift type they target (diverse by construction, never collapsing to one), the gate is
multi-objective (so degenerate strategies lose), and an explicit plateau detector watches the
held-out recall curve and is logged when improvement stalls. The held-out recall curve with CIs
is the honest "are we still improving?" measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer
from bellwether.eval.harness import BenchmarkConfig, BenchmarkReport, run_benchmark
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run
from bellwether.governance import AuditLog, EventType
from bellwether.improve.gate import GovernanceGate, MetricVector
from bellwether.improve.generator import mine_signature
from bellwether.improve.skills import SkillLibrary

# Seed namespaces for the loop's own data, kept far from the benchmark's internal seeds
# ((si+1)*1_000_000 + offsets) so mining never overlaps held-out evaluation.
_MINE_BASELINE_BASE = 900_000_000
_MINE_INCIDENT_BASE = 950_000_000


def _triple(ci: object) -> tuple[float, float, float]:
    return (ci.point, ci.lo, ci.hi)  # type: ignore[attr-defined]


def metric_vector_from_report(report: BenchmarkReport) -> MetricVector:
    m = report.metrics
    return MetricVector(
        recall=_triple(m["recall"]),
        fp_rate=_triple(m["fp_rate"]),
        precision=_triple(m["precision"]),
        detection_rate=_triple(m["detection_rate"]),
        f1=_triple(m["f1"]),
        median_lead_time=_triple(m["median_lead_time"]),
    )


@dataclass(frozen=True)
class ArchiveEntry:
    round_index: int
    variant_id: str
    target: str
    library: SkillLibrary
    metrics: MetricVector
    promoted: bool


@dataclass
class Archive:
    """Lineage of every evaluated variant (promoted or not) — enables rollback + QD search."""

    entries: list[ArchiveEntry] = field(default_factory=list)

    def add(self, entry: ArchiveEntry) -> None:
        self.entries.append(entry)

    @property
    def promoted(self) -> list[ArchiveEntry]:
        return [e for e in self.entries if e.promoted]

    def best_by_recall(self) -> ArchiveEntry | None:
        promoted = self.promoted
        return max(promoted, key=lambda e: e.metrics.recall[0]) if promoted else None


@dataclass(frozen=True)
class Round:
    index: int
    target: str
    promoted: bool
    recall: tuple[float, float, float]
    fp_rate: tuple[float, float, float]
    detection_rate: tuple[float, float, float]
    median_lead_time: tuple[float, float, float]
    plateau: bool = False


@dataclass
class ImprovementHistory:
    rounds: list[Round]
    final_library: SkillLibrary
    archive: Archive

    def recall_curve(self) -> list[float]:
        return [r.recall[0] for r in self.rounds]

    def render(self) -> str:
        lines = [
            "BELLWETHER self-improvement (skill tier) — held-out recall curve",
            "  round  target               promoted  recall (95% CI)            fp-rate   lead",
            "  " + "-" * 78,
        ]
        for r in self.rounds:
            flag = "  yes  " if r.promoted else "  no   "
            plat = "  [plateau]" if r.plateau else ""
            lines.append(
                f"  {r.index:>2}    {r.target:<18} {flag}  "
                f"{r.recall[0]:.3f} [{r.recall[1]:.3f}, {r.recall[2]:.3f}]   "
                f"{r.fp_rate[0]:.3f}     {r.median_lead_time[0]:.2f}{plat}"
            )
        first, last = self.rounds[0].recall[0], self.rounds[-1].recall[0]
        lines.append("  " + "-" * 78)
        lines.append(
            f"  held-out timely recall: {first:.3f} -> {last:.3f}  "
            f"(+{last - first:.3f}) over {len(self.rounds) - 1} rounds; "
            f"{len(self.final_library.names)} signatures learned"
        )
        return "\n".join(lines)


@dataclass
class SelfImprovementLoop:
    """Drives skill-tier self-improvement and records the held-out improvement curve."""

    agent: FixtureAgent
    eval_config: BenchmarkConfig
    audit: AuditLog
    gate: GovernanceGate = field(default_factory=GovernanceGate)
    severity: float = 0.7
    onset_step: int = 1
    n_mine_baseline: int = 120
    n_mine_incidents: int = 15
    plateau_window: int = 3
    plateau_epsilon: float = 0.01

    def __post_init__(self) -> None:
        mgr = BaselineManager()
        for s in range(_MINE_BASELINE_BASE, _MINE_BASELINE_BASE + self.n_mine_baseline):
            mgr.learn(self.agent.clean_run(seed=s))
        baseline = mgr.baseline_for(self.agent.clean_run(seed=_MINE_BASELINE_BASE))
        assert baseline is not None
        self._mining_baseline = baseline
        self._mining_scorer = DriftScorer(self.eval_config.scoring_config())

    def _mine_incidents(self, drift_type: str) -> list:  # type: ignore[type-arg]
        base = _MINE_INCIDENT_BASE + (abs(hash(drift_type)) % 1000) * 10_000
        return [
            faulted_run(self.agent, base + j, FaultSpec(drift_type, self.severity, self.onset_step))
            for j in range(self.n_mine_incidents)
        ]

    def _evaluate(self, library: SkillLibrary) -> MetricVector:
        cfg = replace(self.eval_config, library=library)
        return metric_vector_from_report(run_benchmark(self.agent, cfg))

    def _is_plateau(self, rounds: list[Round]) -> bool:
        if len(rounds) <= self.plateau_window:
            return False
        recent = rounds[-self.plateau_window :]
        gain = recent[-1].recall[0] - rounds[-self.plateau_window - 1].recall[0]
        return gain < self.plateau_epsilon

    def run(self, target_drift_types: list[str]) -> ImprovementHistory:
        library = SkillLibrary()
        archive = Archive()
        rounds: list[Round] = []

        current = self._evaluate(library)
        archive.add(ArchiveEntry(0, "generic", "baseline", library, current, promoted=True))
        rounds.append(
            Round(
                0,
                "baseline",
                True,
                current.recall,
                current.fp_rate,
                current.detection_rate,
                current.median_lead_time,
            )
        )

        for i, drift_type in enumerate(target_drift_types, start=1):
            candidate = mine_signature(
                drift_type,
                self._mine_incidents(drift_type),
                self._mining_baseline,
                self._mining_scorer,
            )
            if candidate is None:
                self.audit.append(EventType.NOTE, drift_type, "no signature could be mined")
                rounds.append(
                    Round(
                        i,
                        drift_type,
                        False,
                        current.recall,
                        current.fp_rate,
                        current.detection_rate,
                        current.median_lead_time,
                    )
                )
                continue

            cand_library = library.with_added(candidate)
            cand_metrics = self._evaluate(cand_library)
            decision = self.gate.decide(current, cand_metrics)
            diff = {
                "add_signature": candidate.name,
                "features": [list(fd) for fd in candidate.feature_directions],
            }

            if decision.promote:
                self.audit.record_promotion(
                    f"signature:{candidate.name}",
                    "; ".join(decision.reasons),
                    diff,
                    decision.metric_deltas,
                )
                self.audit.append(
                    EventType.SIGNATURE_ADDED,
                    f"signature:{candidate.name}",
                    candidate.description,
                    diff,
                )
                library = cand_library
                current = cand_metrics
                promoted = True
            else:
                self.audit.record_rejection(
                    f"signature:{candidate.name}",
                    "; ".join(decision.reasons),
                    diff,
                    decision.metric_deltas,
                )
                promoted = False

            archive.add(ArchiveEntry(i, f"v{i}", drift_type, library, current, promoted))
            r = Round(
                i,
                drift_type,
                promoted,
                current.recall,
                current.fp_rate,
                current.detection_rate,
                current.median_lead_time,
            )
            rounds.append(r)

            if self._is_plateau(rounds):
                self.audit.append(
                    EventType.PLATEAU,
                    "self-improvement",
                    f"held-out recall gain < {self.plateau_epsilon} over {self.plateau_window} rounds",
                    {"recall": current.recall[0]},
                )
                rounds[-1] = replace(r, plateau=True)

        return ImprovementHistory(rounds, library, archive)
