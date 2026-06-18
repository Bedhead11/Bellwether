"""Self-contained HTML dashboard rendering (inline SVG + CSS, no dependencies)."""

from __future__ import annotations

import html
from collections.abc import Sequence

from bellwether.detect import DriftReport
from bellwether.governance import AuditLog
from bellwether.improve.loop import ImprovementHistory
from bellwether.improve.topology import TopologyHistory

_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0d1117;
  color: #c9d1d9; }
.wrap { max-width: 1000px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; } h2 { font-size: 16px; border-bottom: 1px solid #30363d; padding-bottom: 6px;
  margin-top: 32px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px;
  margin: 12px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #21262d; }
th { color: #8b949e; font-weight: 600; }
.drift { color: #f85149; font-weight: 600; } .ok { color: #3fb950; }
.muted { color: #8b949e; } .mono { font-family: ui-monospace, monospace; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 11px;
  background: #21262d; }
"""


def _esc(s: object) -> str:
    return html.escape(str(s))


def _svg_line(
    series: Sequence[float],
    *,
    width: int = 760,
    height: int = 160,
    lo: float | None = None,
    hi: float | None = None,
    color: str = "#58a6ff",
    label: str = "",
) -> str:
    if not series:
        return "<p class='muted'>no data</p>"
    lo = min(series) if lo is None else lo
    hi = max(series) if hi is None else hi
    span = (hi - lo) or 1.0
    pad = 24
    n = len(series)

    def xs(i: int) -> float:
        return pad + (i * (width - 2 * pad) / max(n - 1, 1))

    def ys(v: float) -> float:
        return height - pad - ((v - lo) / span) * (height - 2 * pad)

    pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(series))
    dots = "".join(
        f"<circle cx='{xs(i):.1f}' cy='{ys(v):.1f}' r='3' fill='{color}'/>"
        for i, v in enumerate(series)
    )
    axis = (
        f"<line x1='{pad}' y1='{height - pad}' x2='{width - pad}' y2='{height - pad}' "
        f"stroke='#30363d'/><line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height - pad}' stroke='#30363d'/>"
    )
    ticks = (
        f"<text x='2' y='{ys(hi):.0f}' fill='#8b949e' font-size='10'>{hi:.2f}</text>"
        f"<text x='2' y='{ys(lo):.0f}' fill='#8b949e' font-size='10'>{lo:.2f}</text>"
    )
    cap = (
        f"<text x='{pad}' y='14' fill='#8b949e' font-size='11'>{_esc(label)}</text>"
        if label
        else ""
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%'>{axis}{ticks}{cap}"
        f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{pts}'/>{dots}</svg>"
    )


def _drift_timeline(reports: Sequence[DriftReport]) -> str:
    rows = []
    for r in reports:
        a = r.alert
        if a.triggered:
            what = (
                f"signature:{a.signature}"
                if a.signature
                else f"{a.primary_family}/{a.primary_feature}"
            )
            where = f"step {a.step_index}" if a.step_index is not None else "run-end"
            verdict = "<span class='drift'>DRIFT</span>"
            detail = f"{_esc(what)} @ {where}"
        else:
            verdict = "<span class='ok'>ok</span>"
            detail = "<span class='muted'>—</span>"
        rows.append(
            f"<tr><td class='mono'>{_esc(r.run_id)}</td><td>{verdict}</td>"
            f"<td>{detail}</td><td class='mono'>{r.max_drift:.3f}</td></tr>"
        )
    spark = _svg_line([r.max_drift for r in reports], label="max drift per run", lo=0.0, hi=1.0)
    return (
        f"<div class='card'>{spark}</div>"
        "<table><tr><th>run</th><th>verdict</th><th>attribution</th><th>max drift</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _improvement_curve(history: ImprovementHistory) -> str:
    curve = history.recall_curve()
    rows = []
    for r in history.rounds:
        flag = (
            "<span class='ok'>promoted</span>"
            if r.promoted
            else "<span class='muted'>rejected</span>"
        )
        plat = " <span class='tag'>plateau</span>" if r.plateau else ""
        rows.append(
            f"<tr><td>{r.index}</td><td>{_esc(r.target)}</td><td>{flag}{plat}</td>"
            f"<td class='mono'>{r.recall[0]:.3f} [{r.recall[1]:.3f}, {r.recall[2]:.3f}]</td>"
            f"<td class='mono'>{r.fp_rate[0]:.3f}</td></tr>"
        )
    chart = _svg_line(curve, label="held-out timely recall over rounds", color="#3fb950")
    return (
        f"<div class='card'>{chart}</div>"
        "<table><tr><th>round</th><th>target</th><th>decision</th><th>recall (95% CI)</th>"
        "<th>fp-rate</th></tr>" + "".join(rows) + "</table>"
    )


def _topology(history: TopologyHistory) -> str:
    chart = _svg_line(
        history.best_f1_curve(), label="global-best F1 over iterations", color="#d29922"
    )
    cells = "".join(
        f"<tr><td class='mono'>w={d[0]}, {d[1]}</td><td class='mono'>{e.metrics.f1[0]:.3f}</td>"
        f"<td class='mono'>{e.metrics.recall[0]:.3f}</td><td class='mono'>{e.metrics.fp_rate[0]:.3f}</td></tr>"
        for d, e in sorted(history.archive.items())
    )
    return (
        f"<div class='card'>{chart}</div>"
        "<table><tr><th>niche</th><th>F1</th><th>recall</th><th>fp-rate</th></tr>"
        + cells
        + "</table>"
    )


def _audit(audit: AuditLog) -> str:
    rows = [
        f"<tr><td>{e.seq}</td><td><span class='tag'>{_esc(e.event_type)}</span></td>"
        f"<td class='mono'>{_esc(e.subject)}</td><td>{_esc(e.rationale)}</td></tr>"
        for e in audit
    ]
    status = "verified ✓" if audit.verify() else "TAMPERED ✗"
    return (
        f"<p class='muted'>hash-chained, append-only — integrity: <b>{status}</b> "
        f"({len(audit)} events)</p>"
        "<table><tr><th>#</th><th>type</th><th>subject</th><th>rationale</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def render_dashboard(
    *,
    title: str = "BELLWETHER",
    reports: Sequence[DriftReport] | None = None,
    improvement: ImprovementHistory | None = None,
    topology: TopologyHistory | None = None,
    audit: AuditLog | None = None,
) -> str:
    """Render a self-contained HTML dashboard from whichever artifacts are provided."""
    sections = [f"<div class='wrap'><h1>{_esc(title)} — drift & self-improvement</h1>"]
    if reports:
        sections.append("<h2>Drift timeline</h2>" + _drift_timeline(reports))
    if improvement is not None:
        sections.append("<h2>Self-improvement (skill tier)</h2>" + _improvement_curve(improvement))
    if topology is not None:
        sections.append("<h2>Detector-ensemble topology</h2>" + _topology(topology))
    if audit is not None:
        sections.append("<h2>Governance / audit log</h2>" + _audit(audit))
    sections.append("</div>")
    body = "".join(sections)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{_esc(title)}</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )
