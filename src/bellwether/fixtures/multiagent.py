"""Multi-agent system fixture + coordination fault injectors (the named unsolved problem).

A :class:`MultiAgentSystem` simulates several agents (roles) collaborating via handoffs, emitting
a canonical ``AgentRun`` whose step spans carry an ``agent_role`` attribute. Its "normal"
coordination profile is a forward-biased pipeline (planner → researcher → writer → reviewer) with
moderate handoffs and little ping-pong.

Coordination faults perturb the *interaction structure* while each individual agent may still look
locally normal — exactly the emergent/silent degradation single-agent metrics miss (brief §1):
ping-pong oscillation, role collapse (one agent dominates), and handoff storms.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bellwether.schema import (
    AgentRun,
    ConfigFingerprint,
    InjectedFault,
    RunStatus,
    Span,
    SpanKind,
    SpanStatus,
)

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass
class MAStep:
    role: str
    duration_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str


@dataclass
class MultiAgentSystem:
    agent_id: str = "multi-agent-system"
    task_class: str = "default"
    roles: tuple[str, ...] = ("planner", "researcher", "writer", "reviewer")
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    temperature: float = 0.2
    prompt_version: str = "v1"
    n_steps_mean: float = 14.0
    n_steps_jitter: float = 2.0
    base_latency_s: float = 0.4
    latency_jitter_s: float = 0.08
    base_in_tokens: float = 420.0
    base_out_tokens: float = 130.0

    def fingerprint(self) -> ConfigFingerprint:
        return ConfigFingerprint(
            model_id=self.model,
            provider=self.provider,
            temperature=self.temperature,
            prompt_version=self.prompt_version,
            framework="multiagent-fixture",
            framework_version="0",
            agent_version="1",
        )

    def _next_role(self, idx: int, rng: random.Random) -> int:
        """Clean forward pipeline: an agent either continues or hands off to the next role; at the
        end of the pipeline it restarts a fresh cycle. No back-steps, so the normal profile has
        ~zero ping-pong and stable, balanced handoffs — the baseline a fault must deviate from."""
        last = len(self.roles) - 1
        if idx == last:
            return idx if rng.random() < 0.5 else 0  # stay or start a new cycle
        return idx if rng.random() < 0.5 else idx + 1  # stay or advance

    def role_sequence(self, seed: int, n: int) -> list[str]:
        rng = random.Random(seed)
        idx = 0
        seq = [self.roles[idx]]
        for _ in range(n - 1):
            idx = self._next_role(idx, rng)
            seq.append(self.roles[idx])
        return seq

    def plan(self, seed: int) -> list[MAStep]:
        rng = random.Random(seed * 7919 + 1)
        n = max(4, round(rng.gauss(self.n_steps_mean, self.n_steps_jitter)))
        roles = self.role_sequence(seed, n)
        steps: list[MAStep] = []
        for role in roles:
            in_tok = max(1, round(rng.gauss(self.base_in_tokens, 50)))
            out_tok = max(1, round(rng.gauss(self.base_out_tokens, 25)))
            latency = max(0.01, rng.gauss(self.base_latency_s, self.latency_jitter_s))
            cost = round(in_tok / 1000 * 0.00015 + out_tok / 1000 * 0.0006, 8)
            steps.append(MAStep(role, latency, in_tok, out_tok, cost, self.model))
        return steps

    def materialize(
        self, plan: list[MAStep], *, run_id: str, injected_fault: InjectedFault | None = None
    ) -> AgentRun:
        start = _EPOCH
        cursor = start
        root_id = f"{run_id}-root"
        children: list[Span] = []
        for idx, step in enumerate(plan):
            s_start = cursor
            s_end = s_start + timedelta(seconds=step.duration_s)
            children.append(
                Span(
                    span_id=f"{run_id}-{idx}",
                    parent_span_id=root_id,
                    name=f"{step.role}.act.{idx}",
                    kind=SpanKind.LLM,
                    start_time=s_start,
                    end_time=s_end,
                    status=SpanStatus.OK,
                    input_tokens=step.input_tokens,
                    output_tokens=step.output_tokens,
                    cost_usd=step.cost_usd,
                    model=step.model,
                    attributes={"agent_role": step.role},
                )
            )
            cursor = s_end
        root = Span(
            span_id=root_id,
            parent_span_id=None,
            name=f"orchestrator.{self.agent_id}",
            kind=SpanKind.AGENT,
            start_time=start,
            end_time=cursor,
        )
        return AgentRun(
            run_id=run_id,
            agent_id=self.agent_id,
            task_class=self.task_class,
            start_time=start,
            end_time=cursor,
            status=RunStatus.OK,
            spans=[root, *children],
            config_fingerprint=self.fingerprint(),
            injected_fault=injected_fault,
        )

    def clean_run(self, seed: int, *, run_id: str | None = None) -> AgentRun:
        rid = run_id or f"{self.agent_id}-clean-{seed}"
        return self.materialize(self.plan(seed), run_id=rid)


# --- coordination fault injectors -------------------------------------------------------

CoordFault = Callable[[list[MAStep], float, int], list[MAStep]]


def ping_pong(plan: list[MAStep], severity: float, seed: int) -> list[MAStep]:
    """Inject A↔B oscillation from the onset: two agents bounce work back and forth."""
    out = copy.deepcopy(plan)
    distinct = list(dict.fromkeys(s.role for s in out))
    if len(distinct) < 2:
        return out
    a, b = distinct[0], distinct[1]
    onset = len(out) // 3
    rng = random.Random(seed + 31)
    for i in range(onset, len(out)):
        if rng.random() < severity:
            out[i].role = a if (i - onset) % 2 == 0 else b
    return out


def role_collapse(plan: list[MAStep], severity: float, seed: int) -> list[MAStep]:
    """One agent takes over a contiguous severity-fraction of the work (division of labor breaks
    down). Contiguous (not scattered) so it raises imbalance without incidentally creating
    ping-pong, keeping the fault types cleanly separable."""
    out = copy.deepcopy(plan)
    dominant = out[0].role
    n = len(out)
    k = max(1, round(severity * n))
    for step in out[n - k :]:
        step.role = dominant
    return out


def handoff_storm(plan: list[MAStep], severity: float, seed: int) -> list[MAStep]:
    """Excessive churn: a different agent on (almost) every step, avoiding immediate returns so it
    is high-handoff but not ping-pong and not a fixed loop."""
    out = copy.deepcopy(plan)
    roles = sorted({s.role for s in out}) or ["a"]
    rng = random.Random(seed + 23)
    prev: str | None = None
    prev2: str | None = None
    for step in out:
        if rng.random() < severity:
            choices = (
                [r for r in roles if r != prev and r != prev2]
                or [r for r in roles if r != prev]
                or roles
            )
            step.role = rng.choice(choices)
        prev2, prev = prev, step.role
    return out


COORDINATION_FAULTS: dict[str, CoordFault] = {
    "ping_pong": ping_pong,
    "role_collapse": role_collapse,
    "handoff_storm": handoff_storm,
}

# NOTE (honest limitation): a fourth pathology — a pathological *emergent loop* (a fixed
# cross-agent cycle repeating) — is hard to separate from a *healthy* cyclic workflow when the
# normal coordination profile is itself cyclic (our pipeline restarts), because `cross_agent_loop`
# is high in both. Distinguishing the two needs a richer signal (e.g. cycle length vs. the normal
# cycle, or agent exclusion) and is left as future work rather than faked here.


def inject_coordination_fault(
    plan: list[MAStep], *, fault_type: str, severity: float, seed: int = 0
) -> tuple[list[MAStep], InjectedFault]:
    if fault_type not in COORDINATION_FAULTS:
        raise KeyError(f"unknown coordination fault {fault_type!r}")
    drifted = COORDINATION_FAULTS[fault_type](plan, severity, seed)
    onset = len(plan) // 3
    fault = InjectedFault(
        fault_type=fault_type,
        severity=severity,
        onset_step=onset,
        visible_failure_step=min(len(drifted) - 1, onset + max(1, round((1 - severity) * 3))),
        params={"severity": severity},
    )
    return drifted, fault


@dataclass
class CoordinationFaultSpec:
    fault_type: str
    severity: float = 0.8


def coordination_faulted_run(
    system: MultiAgentSystem, seed: int, spec: CoordinationFaultSpec
) -> AgentRun:
    plan = system.plan(seed)
    drifted, fault = inject_coordination_fault(
        plan, fault_type=spec.fault_type, severity=spec.severity, seed=seed
    )
    rid = f"{system.agent_id}-{spec.fault_type}-{int(spec.severity * 100)}-{seed}"
    return system.materialize(drifted, run_id=rid, injected_fault=fault)
