"""Optional MCP server exposing a :class:`DriftMonitor` as queryable tools.

This is the zero-code form factor's control surface (brief §5 v2): an MCP client can ask
BELLWETHER for an agent's drift status, recent alerts, baselines, and overall status as tools.
``fastmcp`` is an optional dependency (``pip install bellwether[mcp]``); it is imported lazily so
the rest of the package has no hard MCP dependency.
"""

from __future__ import annotations

from typing import Any

from bellwether.monitor import DriftMonitor


def build_mcp_server(monitor: DriftMonitor, *, name: str = "bellwether") -> Any:
    """Build a FastMCP server wrapping ``monitor``'s query surface.

    Raises a clear error if ``fastmcp`` is not installed.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "FastMCP is not installed. Install the MCP extra: pip install 'bellwether[mcp]'"
        ) from exc

    mcp = FastMCP(name)

    @mcp.tool()
    def drift_status(agent_id: str) -> dict[str, object]:
        """Latest drift verdict for an agent (ok / drift / unknown) with attribution."""
        return monitor.drift_status(agent_id)

    @mcp.tool()
    def recent_alerts(n: int = 10) -> list[dict[str, object]]:
        """The most recent drift alerts recorded in the audit log."""
        return monitor.recent_alerts(n)

    @mcp.tool()
    def baselines() -> list[dict[str, object]]:
        """The learned baselines (agent, task_class, fingerprint, observation count)."""
        return monitor.baselines()

    @mcp.tool()
    def status() -> dict[str, object]:
        """Overall monitor status: baseline/agent/alert counts and audit integrity."""
        return monitor.status()

    return mcp
