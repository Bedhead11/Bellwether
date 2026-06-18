"""Tests for the SDK: trace assembly, sink routing, redaction, deploy markers, errors."""

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer
from bellwether.ingest import RedactionConfig, RunStore
from bellwether.schema import ConfigFingerprint, RunStatus, SpanKind
from bellwether.sdk import Bellwether


def test_watch_assembles_span_tree() -> None:
    bw = Bellwether(agent_id="demo", task_class="qa")
    with bw.watch() as handle:
        with bw.llm(model="gpt-4o-mini") as call:
            call.set_tokens(input=400, output=120).set_cost(0.001)
        with bw.tool("search", args={"q": "weather"}):
            pass

    run = handle.run
    assert run is not None
    assert run.agent_id == "demo" and run.task_class == "qa"
    kinds = sorted(s.kind for s in run.spans)
    assert SpanKind.AGENT in kinds and SpanKind.LLM in kinds and SpanKind.TOOL in kinds

    # The LLM and tool spans are children of the root agent span.
    root = run.root_spans()[0]
    children = run.children_of(root.span_id)
    assert len(children) == 2

    llm = next(s for s in run.spans if s.kind == SpanKind.LLM)
    assert llm.input_tokens == 400 and llm.output_tokens == 120 and llm.cost_usd == 0.001


def test_nested_subagent_spans_form_tree() -> None:
    bw = Bellwether(agent_id="demo")
    with bw.watch() as handle, bw.agent("planner"), bw.llm(model="m"):
        pass
    run = handle.run
    assert run is not None
    sub = next(s for s in run.spans if s.kind == SpanKind.SUB_AGENT)
    llm = next(s for s in run.spans if s.kind == SpanKind.LLM)
    assert llm.parent_span_id == sub.span_id  # nested under the sub-agent


def test_store_sink_persists_run() -> None:
    with RunStore() as store:
        bw = Bellwether(agent_id="demo", store=store)
        with bw.watch(), bw.llm(model="m"):
            pass
        assert store.count() == 1


def test_redaction_applied_at_ingest() -> None:
    bw = Bellwether(agent_id="demo", redaction=RedactionConfig())
    with bw.watch() as handle, bw.tool("lookup", args={"email": "user@corp.com"}):
        pass
    run = handle.run
    assert run is not None
    tool = next(s for s in run.spans if s.kind == SpanKind.TOOL)
    assert "REDACTED" in str(tool.tool_args["email"])


def test_deploy_marker_attaches_to_next_run() -> None:
    bw = Bellwether(agent_id="demo")
    bw.mark_deploy(version="2.0", note="new prompt")
    with bw.watch() as h1, bw.llm(model="m"):
        pass
    assert h1.run is not None and h1.run.deploy_marker is not None
    assert h1.run.deploy_marker.version == "2.0"

    # Marker is one-shot: the following run does not carry it.
    with bw.watch() as h2, bw.llm(model="m"):
        pass
    assert h2.run is not None and h2.run.deploy_marker is None


def test_exception_marks_run_error_and_propagates() -> None:
    bw = Bellwether(agent_id="demo")
    handle_box = {}
    try:
        with bw.watch() as handle:
            handle_box["h"] = handle
            with bw.tool("bad"):
                raise ValueError("boom")
    except ValueError:
        pass
    run = handle_box["h"].run
    assert run is not None
    assert run.status == RunStatus.ERROR
    tool = next(s for s in run.spans if s.kind == SpanKind.TOOL)
    assert tool.error is not None and "boom" in tool.error


def test_learn_mode_updates_baseline() -> None:
    fp = ConfigFingerprint(model_id="m", temperature=0.0)
    mgr = BaselineManager()
    bw = Bellwether(agent_id="demo", fingerprint=fp, manager=mgr, learn=True)
    for _ in range(5):
        with bw.watch(), bw.llm(model="m", input_tokens=400, output_tokens=120):
            pass
    # A baseline lineage now exists for this agent/fingerprint.
    assert len(mgr.keys) == 1
    key = mgr.keys[0]
    assert mgr._baselines[key].numeric_count("step_latency", "llm") == 5


def test_monitor_mode_produces_report() -> None:
    fp = ConfigFingerprint(model_id="m", temperature=0.0)
    mgr = BaselineManager()

    # Learn a baseline first.
    learner = Bellwether(agent_id="demo", fingerprint=fp, manager=mgr, learn=True)
    for _ in range(40):
        with learner.watch(), learner.llm(model="m", input_tokens=400, output_tokens=120):
            pass

    reports = []
    monitor = Bellwether(
        agent_id="demo",
        fingerprint=fp,
        manager=mgr,
        scorer=DriftScorer(),
        on_report=reports.append,
    )
    with monitor.watch(), monitor.llm(model="m", input_tokens=400, output_tokens=120):
        pass
    assert len(reports) == 1  # a report was generated and delivered to the callback
