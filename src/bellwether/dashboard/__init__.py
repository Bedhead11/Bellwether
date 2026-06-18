"""Dashboard: render BELLWETHER's state as a single self-contained HTML file.

Zero dependencies and zero server — inline SVG and CSS only, so the output opens in any browser
and is trivial to ship in CI artifacts or a README GIF. It renders the headline visuals from the
brief (§5): the drift timeline, the self-improvement curve, the audit log, and the evolving
detector-ensemble topology.
"""

from bellwether.dashboard.html import render_dashboard

__all__ = ["render_dashboard"]
