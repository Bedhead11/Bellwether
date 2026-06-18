"""Fixture agent: a controllable synthetic agent that emits canonical ``AgentRun`` traces.

This is the Phase 0 data faucet and the L1 ground-truth engine from the eval design
(``docs/design/02-eval-harness.md``). A fixture agent produces a *plan* (an ordered list of
``StepSpec``) from a seed; faults transform the plan; ``materialize`` turns a plan into an
``AgentRun`` with correctly re-flowed timestamps. Operating on plans (not materialized runs)
keeps timestamp bookkeeping in one place, so injected faults never produce inconsistent traces.

Everything is deterministic given a seed, so the eval harness can run N reproducible seeds and
report confidence intervals (per the brief's statistical-rigor requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

# Fixed epoch so materialized runs are byte-stable across machines/timezones.
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass
class StepSpec:
    """One planned step before materialization into a span."""

    kind: SpanKind
    name: str
    duration_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, object] = field(default_factory=dict)
    status: SpanStatus = SpanStatus.OK
    error: str | None = None


@dataclass
class FixtureAgent:
    """A synthetic agent with a stable behavioral profile.

    Default behavior: a sequence of (LLM call, tool call) pairs. Latencies and token counts are
    drawn from per-agent distributions; tool selection follows a task-class-conditioned weight
    vector. This stable profile is exactly what a baseline should learn — faults perturb it.
    """

    agent_id: str = "fixture-agent"
    task_class: str = "default"
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    temperature: float = 0.2
    prompt_version: str = "v1"

    tools: tuple[str, ...] = ("search", "calculator", "db_query", "summarize")
    # Selection weights over ``tools`` (task-class conditioning); normalized internally.
    tool_weights: tuple[float, ...] = (0.5, 0.2, 0.2, 0.1)

    n_steps_mean: float = 4.0
    n_steps_jitter: float = 1.0
    base_latency_s: float = 0.40
    latency_jitter_s: float = 0.08
    base_in_tokens: float = 420.0
    in_tokens_jitter: float = 60.0
    base_out_tokens: float = 130.0
    out_tokens_jitter: float = 30.0
    cost_per_1k_in: float = 0.00015
    cost_per_1k_out: float = 0.00060

    def fingerprint(self) -> ConfigFingerprint:
        return ConfigFingerprint(
            model_id=self.model,
            provider=self.provider,
            temperature=self.temperature,
            prompt_version=self.prompt_version,
            framework="fixture",
            framework_version="0",
            agent_version="1",
        )

    # --- planning -----------------------------------------------------------------------

    def _llm_cost(self, in_tok: int, out_tok: int) -> float:
        return round(in_tok / 1000 * self.cost_per_1k_in + out_tok / 1000 * self.cost_per_1k_out, 8)

    def plan(self, seed: int) -> list[StepSpec]:
        """Produce a deterministic clean plan for the given seed."""
        import random

        rng = random.Random(seed)
        n_steps = max(1, round(rng.gauss(self.n_steps_mean, self.n_steps_jitter)))

        total_w = sum(self.tool_weights)
        weights = [w / total_w for w in self.tool_weights]

        steps: list[StepSpec] = []
        for i in range(n_steps):
            in_tok = max(1, round(rng.gauss(self.base_in_tokens, self.in_tokens_jitter)))
            out_tok = max(1, round(rng.gauss(self.base_out_tokens, self.out_tokens_jitter)))
            llm_latency = max(0.01, rng.gauss(self.base_latency_s, self.latency_jitter_s))
            steps.append(
                StepSpec(
                    kind=SpanKind.LLM,
                    name=f"llm.call.{i}",
                    duration_s=llm_latency,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=self._llm_cost(in_tok, out_tok),
                    model=self.model,
                )
            )
            tool = rng.choices(self.tools, weights=weights, k=1)[0]
            tool_latency = max(0.01, rng.gauss(self.base_latency_s * 0.6, self.latency_jitter_s))
            steps.append(
                StepSpec(
                    kind=SpanKind.TOOL,
                    name=f"tool.{tool}.{i}",
                    duration_s=tool_latency,
                    tool_name=tool,
                    tool_args={"q": f"step-{i}"},
                )
            )
        return steps

    # --- materialization ----------------------------------------------------------------

    def materialize(
        self,
        plan: list[StepSpec],
        *,
        run_id: str,
        injected_fault: InjectedFault | None = None,
        start: datetime | None = None,
    ) -> AgentRun:
        """Turn a plan into an ``AgentRun`` with sequential, consistent timestamps.

        A single root ``agent`` span wraps sequential child spans. Child end-times re-flow from
        the plan's durations, so any fault that changed a duration yields a consistent trace.
        """
        start = start or _EPOCH
        children: list[Span] = []
        cursor = start
        root_id = f"{run_id}-root"

        for idx, step in enumerate(plan):
            s_start = cursor
            s_end = s_start + timedelta(seconds=step.duration_s)
            children.append(
                Span(
                    span_id=f"{run_id}-{idx}",
                    parent_span_id=root_id,
                    name=step.name,
                    kind=step.kind,
                    start_time=s_start,
                    end_time=s_end,
                    status=step.status,
                    input_tokens=step.input_tokens,
                    output_tokens=step.output_tokens,
                    cost_usd=step.cost_usd,
                    model=step.model,
                    tool_name=step.tool_name,
                    tool_args=dict(step.tool_args),
                    error=step.error,
                )
            )
            cursor = s_end

        root = Span(
            span_id=root_id,
            parent_span_id=None,
            name=f"agent.{self.agent_id}",
            kind=SpanKind.AGENT,
            start_time=start,
            end_time=cursor,
            status=(
                SpanStatus.ERROR
                if any(c.status == SpanStatus.ERROR for c in children)
                else SpanStatus.OK
            ),
        )

        status = RunStatus.ERROR if root.status == SpanStatus.ERROR else RunStatus.OK
        return AgentRun(
            run_id=run_id,
            agent_id=self.agent_id,
            task_class=self.task_class,
            start_time=start,
            end_time=cursor,
            status=status,
            spans=[root, *children],
            config_fingerprint=self.fingerprint(),
            injected_fault=injected_fault,
        )

    def clean_run(self, seed: int, *, run_id: str | None = None) -> AgentRun:
        """Convenience: plan + materialize a benign run."""
        rid = run_id or f"{self.agent_id}-clean-{seed}"
        return self.materialize(self.plan(seed), run_id=rid)
