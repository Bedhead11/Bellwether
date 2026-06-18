"""Fault injectors — the taxonomy of known, dial-able drifts (eval design doc 02).

Each injector transforms a clean plan into a drifted plan and returns the ground-truth
``InjectedFault`` describing what was done (type, severity, onset, visible-failure step). Faults
operate on step plans (not materialized runs) so timestamps re-flow consistently on
materialization.

Two invariants every injector must respect (enforced by the L3 metamorphic tests):
1. Severity is monotone — a strictly larger severity must produce a strictly "worse" trace on
   the affected feature family, so the detector's drift score must not *decrease* with severity.
2. Onset/visible-failure are well-defined so lead-time is measurable.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

from bellwether.fixtures.agent import StepSpec
from bellwether.schema import InjectedFault, SpanKind, SpanStatus

# A fault injector takes (plan, severity, onset_step, seed) and returns the drifted plan.
FaultInjector = Callable[[list[StepSpec], float, int, int], list[StepSpec]]

_LOOP_COST_K = 1.0  # severity-to-effect gains, kept explicit for monotonicity reasoning
_LATENCY_K = 4.0
_COST_K = 4.0


def _visible_failure_step(n_steps: int, onset: int, severity: float) -> int:
    """Higher severity becomes visible sooner; lower severity takes longer to surface.

    Returns a step index in ``[onset, n_steps-1]``. Used as ``t_fail`` for lead-time.
    """
    tail = max(1, n_steps - onset)
    offset = max(1, round((1.0 - severity) * tail))
    return min(n_steps - 1, onset + offset)


def _clone(plan: list[StepSpec]) -> list[StepSpec]:
    return copy.deepcopy(plan)


# --- individual injectors ---------------------------------------------------------------


def latency_injection(
    plan: list[StepSpec], severity: float, onset: int, seed: int
) -> list[StepSpec]:
    out = _clone(plan)
    factor = 1.0 + severity * _LATENCY_K
    for step in out[onset:]:
        step.duration_s *= factor
    return out


def cost_blowup(plan: list[StepSpec], severity: float, onset: int, seed: int) -> list[StepSpec]:
    out = _clone(plan)
    factor = 1.0 + severity * _COST_K
    for step in out[onset:]:
        if step.kind == SpanKind.LLM and step.input_tokens is not None:
            step.input_tokens = round(step.input_tokens * factor)
            out_tok = step.output_tokens or 0
            step.cost_usd = round(step.input_tokens / 1000 * 0.00015 + out_tok / 1000 * 0.00060, 8)
    return out


def output_degradation(
    plan: list[StepSpec], severity: float, onset: int, seed: int
) -> list[StepSpec]:
    out = _clone(plan)
    keep = max(0.05, 1.0 - severity * 0.85)  # shrink output length toward elision
    for step in out[onset:]:
        if step.kind == SpanKind.LLM and step.output_tokens is not None:
            step.output_tokens = max(1, round(step.output_tokens * keep))
    return out


def tool_misselection(
    plan: list[StepSpec], severity: float, onset: int, seed: int
) -> list[StepSpec]:
    import random

    rng = random.Random(seed + 7)
    out = _clone(plan)
    # Replace a severity-controlled fraction of tool steps with an off-profile tool.
    for step in out[onset:]:
        if step.kind == SpanKind.TOOL and rng.random() < severity:
            step.tool_name = "deprecated_tool"
            step.name = step.name.replace("tool.", "tool.deprecated_tool.")
    return out


def induced_loop(plan: list[StepSpec], severity: float, onset: int, seed: int) -> list[StepSpec]:
    out = _clone(plan)
    if not out:
        return out
    onset = min(onset, len(out) - 1)
    template = out[onset]
    # Insert repeated identical (tool, args) cycles — the canonical loop signature.
    extra = max(1, round(severity * 8 * _LOOP_COST_K))
    looped = [copy.deepcopy(template) for _ in range(extra)]
    for j, step in enumerate(looped):
        step.name = f"{template.name}.loop{j}"
    return out[: onset + 1] + looped + out[onset + 1 :]


def retry_storm(plan: list[StepSpec], severity: float, onset: int, seed: int) -> list[StepSpec]:
    out = _clone(plan)
    if not out:
        return out
    onset = min(onset, len(out) - 1)
    retries = max(1, round(severity * 6))
    inserts: list[StepSpec] = []
    for r in range(retries):
        retry = copy.deepcopy(out[onset])
        retry.status = SpanStatus.ERROR
        retry.error = "transient failure, retrying"
        retry.name = f"{out[onset].name}.retry{r}"
        inserts.append(retry)
    return out[:onset] + inserts + out[onset:]


# --- registry ---------------------------------------------------------------------------

FAULT_INJECTORS: dict[str, FaultInjector] = {
    "latency_injection": latency_injection,
    "cost_blowup": cost_blowup,
    "output_degradation": output_degradation,
    "tool_misselection": tool_misselection,
    "induced_loop": induced_loop,
    "retry_storm": retry_storm,
}

# Maps each fault to the feature family it primarily perturbs (for attribution tests, doc 02).
FAULT_PRIMARY_FAMILY: dict[str, str] = {
    "latency_injection": "temporal",
    "cost_blowup": "economic",
    "output_degradation": "semantic",
    "tool_misselection": "tool",
    "induced_loop": "structural",
    "retry_storm": "structural",
}


def inject_fault(
    plan: list[StepSpec],
    *,
    fault_type: str,
    severity: float,
    onset_step: int = 0,
    seed: int = 0,
) -> tuple[list[StepSpec], InjectedFault]:
    """Apply a named fault to a plan, returning the drifted plan and its ground-truth label."""
    if fault_type not in FAULT_INJECTORS:
        raise KeyError(f"unknown fault_type {fault_type!r}; known: {sorted(FAULT_INJECTORS)}")
    if not (0.0 <= severity <= 1.0):
        raise ValueError("severity must be in [0, 1]")
    if math.isnan(severity):
        raise ValueError("severity must not be NaN")

    onset = max(0, min(onset_step, len(plan) - 1)) if plan else 0
    drifted = FAULT_INJECTORS[fault_type](plan, severity, onset, seed)
    fault = InjectedFault(
        fault_type=fault_type,
        severity=severity,
        onset_step=onset,
        visible_failure_step=_visible_failure_step(len(drifted), onset, severity),
        params={"severity": severity},
    )
    return drifted, fault
