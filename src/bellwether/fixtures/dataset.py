"""Labeled dataset generation for the eval harness.

Produces reproducible mixes of benign and faulted runs from a :class:`FixtureAgent`. Each run
carries its ground-truth label (``injected_fault``), so downstream metric computation
(precision/recall/lead-time/FP-rate, doc 02) is exact. Generation is seeded end-to-end so a
given ``(agent, config, seed)`` always yields the same dataset — required for CI reproducibility
and the N-seed confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

from bellwether.fixtures.agent import FixtureAgent
from bellwether.fixtures.faults import FAULT_INJECTORS, inject_fault
from bellwether.schema import AgentRun


@dataclass
class FaultSpec:
    """A request to generate a faulted run of a given type/severity."""

    fault_type: str
    severity: float
    onset_step: int = 1


def benign_run(agent: FixtureAgent, seed: int) -> AgentRun:
    return agent.clean_run(seed, run_id=f"{agent.agent_id}-benign-{seed}")


def faulted_run(agent: FixtureAgent, seed: int, spec: FaultSpec) -> AgentRun:
    plan = agent.plan(seed)
    drifted, fault = inject_fault(
        plan,
        fault_type=spec.fault_type,
        severity=spec.severity,
        onset_step=spec.onset_step,
        seed=seed,
    )
    rid = f"{agent.agent_id}-{spec.fault_type}-{int(spec.severity * 100)}-{seed}"
    return agent.materialize(drifted, run_id=rid, injected_fault=fault)


def generate_dataset(
    agent: FixtureAgent,
    *,
    n_benign: int = 20,
    fault_specs: list[FaultSpec] | None = None,
    n_per_fault: int = 5,
    seed_start: int = 0,
) -> list[AgentRun]:
    """Generate a labeled dataset: ``n_benign`` clean runs plus faulted runs per spec.

    If ``fault_specs`` is None, generate one moderate-severity instance of every fault type.
    """
    if fault_specs is None:
        fault_specs = [FaultSpec(ft, severity=0.7) for ft in sorted(FAULT_INJECTORS)]

    runs: list[AgentRun] = []
    seed = seed_start
    for _ in range(n_benign):
        runs.append(benign_run(agent, seed))
        seed += 1
    for spec in fault_specs:
        for _ in range(n_per_fault):
            runs.append(faulted_run(agent, seed, spec))
            seed += 1
    return runs
