"""L2 eval layer: inject faults into *recorded* runs (replay corpus realism).

The L1 fixture injectors operate on synthetic plans. This module injects the same fault taxonomy
into an arbitrary recorded ``AgentRun`` — one captured via the SDK, the OTLP receiver, or a public
trace corpus — so detection quality can be measured against *realistic* traces while keeping exact
ground-truth labels (eval design doc 02, L2). It targets the common flat trace shape (a root span
wrapping sequential children) and re-flows timestamps so the result stays schema-valid.
"""

from __future__ import annotations

import copy
from datetime import timedelta

from bellwether.schema import AgentRun, InjectedFault, RunStatus, Span, SpanKind, SpanStatus

_LATENCY_K = 4.0
_COST_K = 4.0


def _children(run: AgentRun) -> list[Span]:
    kids = [s for s in run.spans if s.parent_span_id is not None]
    kids.sort(key=lambda s: (s.start_time, s.span_id))
    return kids


def _llm_cost(in_tok: int, out_tok: int) -> float:
    return round(in_tok / 1000 * 0.00015 + out_tok / 1000 * 0.0006, 8)


def _apply(
    children: list[Span], fault_type: str, severity: float, onset: int, seed: int
) -> tuple[list[tuple[Span, float]], dict[str, float]]:
    """Return (span, duration) pairs after injecting the fault, plus extra label params."""
    import random

    rng = random.Random(seed + 17)
    out: list[tuple[Span, float]] = [(s, s.duration_s) for s in children]

    if fault_type == "latency_injection":
        factor = 1.0 + severity * _LATENCY_K
        out = [(s, d * factor if i >= onset else d) for i, (s, d) in enumerate(out)]
    elif fault_type == "cost_blowup":
        factor = 1.0 + severity * _COST_K
        new = []
        for i, (s, d) in enumerate(out):
            if i >= onset and s.kind == SpanKind.LLM and s.input_tokens is not None:
                in_tok = round(s.input_tokens * factor)
                s = s.model_copy(
                    update={
                        "input_tokens": in_tok,
                        "cost_usd": _llm_cost(in_tok, s.output_tokens or 0),
                    }
                )
            new.append((s, d))
        out = new
    elif fault_type == "output_degradation":
        keep = max(0.05, 1.0 - severity * 0.85)
        out = [
            (
                s.model_copy(update={"output_tokens": max(1, round(s.output_tokens * keep))})
                if (i >= onset and s.kind == SpanKind.LLM and s.output_tokens is not None)
                else s,
                d,
            )
            for i, (s, d) in enumerate(out)
        ]
    elif fault_type == "tool_misselection":
        out = [
            (
                s.model_copy(
                    update={"tool_name": "deprecated_tool", "name": "tool.deprecated_tool"}
                )
                if (i >= onset and s.tool_name is not None and rng.random() < severity)
                else s,
                d,
            )
            for i, (s, d) in enumerate(out)
        ]
    elif fault_type == "induced_loop":
        if out:
            onset = min(onset, len(out) - 1)
            template, t_dur = out[onset]
            extra = max(1, round(severity * 6))
            loop = [(template.model_copy(), t_dur) for _ in range(extra)]
            out = out[: onset + 1] + loop + out[onset + 1 :]
    elif fault_type == "retry_storm":
        if out:
            onset = min(onset, len(out) - 1)
            template, t_dur = out[onset]
            retries = max(1, round(severity * 5))
            inserts = [
                (template.model_copy(update={"status": SpanStatus.ERROR, "error": "retry"}), t_dur)
                for _ in range(retries)
            ]
            out = out[:onset] + inserts + out[onset:]
    else:
        raise KeyError(f"unknown fault_type {fault_type!r}")

    return out, {"severity": severity}


def inject_into_run(
    run: AgentRun,
    *,
    fault_type: str,
    severity: float,
    onset_step: int | None = None,
    seed: int = 0,
    run_id: str | None = None,
) -> AgentRun:
    """Inject a fault into a recorded run, re-flowing timestamps and attaching ground truth.

    The run must have a single root wrapping sequential children (the canonical shape). The result
    is schema-valid and labelled with an :class:`InjectedFault` for exact eval scoring.
    """
    roots = run.root_spans()
    if len(roots) != 1:
        raise ValueError("replay injection targets a single-root, flat trace")
    root = roots[0]
    kids = _children(run)
    if not kids:
        raise ValueError("run has no child spans to perturb")

    onset = onset_step if onset_step is not None else max(1, len(kids) // 3)
    injected, params = _apply(copy.deepcopy(kids), fault_type, severity, onset, seed)

    # Re-flow: lay the (possibly inserted) children out sequentially from the root's start.
    cursor = root.start_time
    new_children: list[Span] = []
    for idx, (span, dur) in enumerate(injected):
        start = cursor
        end = start + timedelta(seconds=max(dur, 0.0))
        new_children.append(
            span.model_copy(
                update={"span_id": f"{span.span_id}#{idx}", "start_time": start, "end_time": end}
            )
        )
        cursor = end

    status = (
        SpanStatus.ERROR
        if any(c.status == SpanStatus.ERROR for c in new_children)
        else SpanStatus.OK
    )
    new_root = root.model_copy(update={"end_time": cursor, "status": status})
    # Re-parent children to the (unchanged) root id.
    new_children = [c.model_copy(update={"parent_span_id": new_root.span_id}) for c in new_children]

    n = len(new_children)
    fault = InjectedFault(
        fault_type=fault_type,
        severity=severity,
        onset_step=min(onset, n - 1),
        visible_failure_step=min(n - 1, onset + max(1, round((1 - severity) * 3))),
        params=params,
    )
    return run.model_copy(
        update={
            "run_id": run_id or f"{run.run_id}-{fault_type}-{int(severity * 100)}",
            "start_time": new_root.start_time,
            "end_time": cursor,
            "status": RunStatus.OK if status == SpanStatus.OK else RunStatus.ERROR,
            "spans": [new_root, *new_children],
            "injected_fault": fault,
        }
    )
