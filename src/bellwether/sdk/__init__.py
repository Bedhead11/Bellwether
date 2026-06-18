"""The BELLWETHER SDK: instrument an agent in minutes and get drift detection.

Wrap a run with ``watch`` and record LLM/tool calls with ``llm``/``tool`` context managers; the
SDK assembles a canonical ``AgentRun``, redacts PII at ingest, and routes it to sinks (a store
and/or a live drift scorer). This is the adoption wedge from the brief (§9): one decorator or
``with`` block to get value.

Example::

    from bellwether.sdk import Bellwether

    bw = Bellwether(agent_id="my-agent", store=RunStore("runs.duckdb"))

    with bw.watch(task_class="qa"):
        with bw.llm(model="gpt-4o-mini") as call:
            ...  # do the LLM call
            call.set_tokens(input=420, output=130)
        with bw.tool("search", args={"q": "weather"}):
            ...  # run the tool
"""

from bellwether.sdk.watch import Bellwether, RunHandle, SpanHandle

__all__ = ["Bellwether", "RunHandle", "SpanHandle"]
