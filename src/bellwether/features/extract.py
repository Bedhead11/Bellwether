"""Pure feature extractors: ``AgentRun`` -> ordered ``FeatureObservation`` sequence.

Per-step features cover every fault in the taxonomy (eval doc 02) at the cadence needed for
lead-time:

| feature           | family     | catches                              |
|-------------------|------------|--------------------------------------|
| step_latency      | temporal   | latency_injection                    |
| step_input_tokens | context    | cost_blowup (prompt growth)          |
| step_output_tokens| semantic   | output_degradation (elision)         |
| step_cost         | economic   | cost_blowup                          |
| is_error          | structural | retry_storm                          |
| repeat_count      | structural | induced_loop (repeated tool+args)    |
| tool_id (categ.)  | tool       | tool_misselection (novel tool)       |

Run-summary features add per-run breadth (depth, branching, distinct tools, totals, variability)
scored once at run end.
"""

from __future__ import annotations

import json
import math
from collections import Counter

from bellwether.features.types import (
    GLOBAL_CONTEXT,
    RUN_CONTEXT,
    CategoricalObs,
    FeatureFamily,
    FeatureObservation,
    NumericObs,
)
from bellwether.schema import AgentRun, Span, SpanKind, SpanStatus

F = FeatureFamily


def _tool_signature(span: Span) -> str:
    """Stable identity of a tool call (name + sorted args) for loop/recurrence detection."""
    return json.dumps([span.tool_name, span.tool_args], sort_keys=True, default=str)


def _ordered_steps(run: AgentRun) -> list[Span]:
    """Non-root spans in temporal order — the run's steps."""
    steps = [s for s in run.spans if s.parent_span_id is not None]
    steps.sort(key=lambda s: (s.start_time, s.span_id))
    return steps


def step_observations(run: AgentRun) -> list[FeatureObservation]:
    """One observation per step. ``repeat_count`` is recurrence *within this run so far*."""
    out: list[FeatureObservation] = []
    seen_tool_sig: dict[str, int] = {}

    for i, span in enumerate(_ordered_steps(run)):
        ctx = span.kind.value
        numerics: list[NumericObs] = [
            NumericObs("step_latency", ctx, span.duration_s, F.TEMPORAL),
            NumericObs(
                "is_error",
                GLOBAL_CONTEXT,
                1.0 if span.status == SpanStatus.ERROR else 0.0,
                F.STRUCTURAL,
            ),
        ]
        if span.input_tokens is not None:
            numerics.append(
                NumericObs("step_input_tokens", ctx, float(span.input_tokens), F.CONTEXT)
            )
        if span.output_tokens is not None:
            numerics.append(
                NumericObs("step_output_tokens", ctx, float(span.output_tokens), F.SEMANTIC)
            )
        if span.cost_usd is not None:
            numerics.append(NumericObs("step_cost", ctx, float(span.cost_usd), F.ECONOMIC))

        categoricals: list[CategoricalObs] = []
        if span.tool_name is not None:
            sig = _tool_signature(span)
            rc = seen_tool_sig.get(sig, 0)
            # Only tool calls (with concrete args) get a recurrence signal; benign tool calls
            # carry distinct args per step, so repeat_count stays 0 unless a loop repeats them.
            numerics.append(NumericObs("repeat_count", GLOBAL_CONTEXT, float(rc), F.STRUCTURAL))
            seen_tool_sig[sig] = rc + 1
            categoricals.append(CategoricalObs("tool_id", GLOBAL_CONTEXT, span.tool_name, F.TOOL))

        out.append(FeatureObservation("step", i, tuple(numerics), tuple(categoricals)))
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _cv(xs: list[float]) -> float:
    """Coefficient of variation; 0 when mean is ~0 to avoid blow-ups."""
    m = _mean(xs)
    return _std(xs) / m if abs(m) > 1e-9 else 0.0


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _agent_roles(run: AgentRun) -> list[str]:
    """The ordered sequence of agent roles across steps (multi-agent runs only).

    A role is read from ``span.attributes['agent_role']``; sub-agent spans fall back to their
    name. Runs with fewer than two distinct roles are single-agent and yield no coordination
    features.
    """
    roles: list[str] = []
    for s in _ordered_steps(run):
        role = s.attributes.get("agent_role")
        if role is None and s.kind == SpanKind.SUB_AGENT:
            role = s.name
        if role is not None:
            roles.append(str(role))
    return roles


def coordination_features(run: AgentRun) -> list[NumericObs]:
    """Multi-agent coordination features — the named unsolved problem (brief §1).

    Captures *emergent* behavior across agents that each individual agent's metrics miss: role
    balance, handoff structure, ping-pong, and cross-agent loops. Empty for single-agent runs.
    """
    roles = _agent_roles(run)
    if not roles:
        return []  # not a multi-agent run (no agent_role attributes)
    distinct = set(roles)

    n = len(roles)
    counts = Counter(roles)
    shares = [c / n for c in counts.values()]
    role_entropy = -sum(s * math.log(s) for s in shares)
    role_imbalance = max(shares)

    transitions = [(roles[i], roles[i + 1]) for i in range(n - 1) if roles[i] != roles[i + 1]]
    handoff_count = len(transitions)
    handoff_rate = handoff_count / (n - 1) if n > 1 else 0.0

    pingpong = sum(
        1
        for i in range(len(transitions) - 1)
        if transitions[i + 1] == (transitions[i][1], transitions[i][0])
    )
    pingpong_rate = pingpong / handoff_count if handoff_count else 0.0

    trans_counts = Counter(transitions)
    cross_agent_loop = float(max(trans_counts.values()) - 1) if trans_counts else 0.0
    tt = sum(trans_counts.values())
    trans_entropy = -sum((c / tt) * math.log(c / tt) for c in trans_counts.values()) if tt else 0.0

    max_consec, cur = 1, 1
    for i in range(1, n):
        if roles[i] == roles[i - 1]:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 1

    def c(name: str, value: float) -> NumericObs:
        return NumericObs(name, RUN_CONTEXT, value, F.COORDINATION)

    return [
        c("n_agents", float(len(distinct))),
        c("role_entropy", role_entropy),
        c("role_imbalance", role_imbalance),
        c("handoff_count", float(handoff_count)),
        c("handoff_rate", handoff_rate),
        c("pingpong_rate", pingpong_rate),
        c("cross_agent_loop", cross_agent_loop),
        c("role_transition_entropy", trans_entropy),
        c("max_consecutive_same_role", float(max_consec)),
    ]


def run_summary_observation(run: AgentRun) -> FeatureObservation:
    """Per-run aggregate features, scored once at run end (context ``"run"``)."""
    steps = _ordered_steps(run)
    latencies = [s.duration_s for s in steps]
    in_tokens = [float(s.input_tokens) for s in steps if s.input_tokens is not None]
    out_tokens = [float(s.output_tokens) for s in steps if s.output_tokens is not None]
    costs = [float(s.cost_usd) for s in steps if s.cost_usd is not None]
    tool_steps = [s for s in steps if s.tool_name is not None]
    tool_names = [s.tool_name for s in tool_steps if s.tool_name is not None]
    errors = sum(1 for s in steps if s.status == SpanStatus.ERROR)

    # Loop score: max within-run recurrence of any (tool, args) signature.
    sig_counts: dict[str, int] = {}
    for s in tool_steps:
        sig = _tool_signature(s)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
    loop_score = float(max(sig_counts.values()) - 1) if sig_counts else 0.0

    # Tool-distribution entropy (nats).
    entropy = 0.0
    if tool_names:
        counts: dict[str, int] = {}
        for t in tool_names:
            counts[t] = counts.get(t, 0) + 1
        total = len(tool_names)
        entropy = -sum((c / total) * math.log(c / total) for c in counts.values())

    def n(name: str, value: float, fam: FeatureFamily) -> NumericObs:
        return NumericObs(name, RUN_CONTEXT, value, fam)

    numerics: tuple[NumericObs, ...] = (
        # structural
        n("step_count", float(len(steps)), F.STRUCTURAL),
        n("max_depth", float(run.max_depth()), F.STRUCTURAL),
        n("retry_count", float(errors), F.STRUCTURAL),
        n("loop_score", loop_score, F.STRUCTURAL),
        n("sub_agent_count", float(len(run.spans_of_kind(SpanKind.SUB_AGENT))), F.STRUCTURAL),
        # tool
        n("tool_call_count", float(len(tool_steps)), F.TOOL),
        n("distinct_tools", float(len(set(tool_names))), F.TOOL),
        n("tool_entropy", entropy, F.TOOL),
        # temporal
        n("total_duration_s", run.duration_s, F.TEMPORAL),
        n("mean_step_latency", _mean(latencies), F.TEMPORAL),
        n("p95_step_latency", _percentile(latencies, 0.95), F.TEMPORAL),
        n("latency_cv", _cv(latencies), F.TEMPORAL),
        # economic
        n("total_input_tokens", sum(in_tokens), F.ECONOMIC),
        n("total_cost_usd", sum(costs), F.ECONOMIC),
        n("mean_cost_per_step", _mean(costs), F.ECONOMIC),
        # semantic
        n("mean_output_tokens", _mean(out_tokens), F.SEMANTIC),
        n("min_output_tokens", float(min(out_tokens)) if out_tokens else 0.0, F.SEMANTIC),
        n("output_tokens_cv", _cv(out_tokens), F.SEMANTIC),
        # context
        n("max_input_tokens", float(max(in_tokens)) if in_tokens else 0.0, F.CONTEXT),
        n("mean_input_tokens", _mean(in_tokens), F.CONTEXT),
    )
    # Append coordination features for multi-agent runs (empty otherwise).
    numerics = (*numerics, *coordination_features(run))
    return FeatureObservation("run_summary", None, numerics, ())


def extract_observations(run: AgentRun) -> list[FeatureObservation]:
    """Full observation sequence for a run: per-step observations + final run-summary."""
    return [*step_observations(run), run_summary_observation(run)]
