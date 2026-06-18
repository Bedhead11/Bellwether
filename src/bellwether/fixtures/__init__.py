"""Fixture-agent harness: the Phase 0 data faucet and L1 ground-truth engine (eval doc 02)."""

from bellwether.fixtures.agent import FixtureAgent, StepSpec
from bellwether.fixtures.dataset import (
    FaultSpec,
    benign_run,
    faulted_run,
    generate_dataset,
)
from bellwether.fixtures.faults import (
    FAULT_INJECTORS,
    FAULT_PRIMARY_FAMILY,
    inject_fault,
)

__all__ = [
    "FixtureAgent",
    "StepSpec",
    "FaultSpec",
    "benign_run",
    "faulted_run",
    "generate_dataset",
    "FAULT_INJECTORS",
    "FAULT_PRIMARY_FAMILY",
    "inject_fault",
]
