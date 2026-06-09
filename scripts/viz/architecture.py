"""Architecture diagram for the C9-ALS analysis pipeline.

Single figure built with Plotly shapes + annotations.  Each station's
glyph is the shape of the data the station emits — text reserved for
naming, the geometry carries the meaning.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Engineering palette — disciplined, no decorative chrome.
INK = "#1a1a1a"            # primary text / line
MID = "#666"               # secondary text / muted line
LIGHT = "#b8b8b8"          # tertiary
WHITE = "#ffffff"
PAPER = "#fafafa"
ACCENT = "#0a3b80"         # routes the eye through the active path
ACCENT_SOFT = "#dce8f5"    # accent at low intensity (cell fills, light highlights)

MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def fill_color(intensity: float) -> str:
    """Interpolate white → ACCENT by intensity ∈ [0, 1]."""
    r = int(255 - (255 - 10) * intensity)
    g = int(255 - (255 - 59) * intensity)
    b = int(255 - (255 - 128) * intensity)
    return f"rgb({r},{g},{b})"


def build_pipeline_fig() -> go.Figure:
    """Five-station dataflow where each station's glyph IS its data shape.

    The visual story is dimensionality collapse: matrix → vector → graph
    → per-anchor bullseye → ranked bars.  Text is reserved for naming
    (station name, data dim, short caption) and for the verb on each
    arrow.  No operation blocks, no title bars — the geometry carries
    the meaning."""
    FIG_W = 1380
    FIG_H = 460
    M_L, M_R, M_T, M_B = 24, 24, 72, 44
    plot_w_px = FIG_W - M_L - M_R
    plot_h_px = FIG_H - M_T - M_B
    aspect = plot_h_px / plot_w_px

    n_slots = 5
    slot_w = 1.0 / n_slots
    slot_cxs = [(i + 0.5) * slot_w for i in range(n_slots)]
    cy = 0.51

    shapes: list[dict] = []
    annotations: list[dict] = []
    scatter_specs: list[dict] = []

    fig = go.Figure()

    def station_label(cx: float, top_y: float, bot_y: float,
                      name: str, dim_text: str, caption: str) -> None:
        annotations.append(dict(
            xref="paper", yref="paper",
            x=cx, y=top_y + 0.055,
            text=(f"<b style='font-family:{SANS};font-size:13px;color:{INK};"
                  f"letter-spacing:1.4px'>{name}</b>"),
            showarrow=False, align="center",
        ))
        annotations.append(dict(
            xref="paper", yref="paper",
            x=cx, y=bot_y - 0.05,
            text=(f"<span style='font-family:{MONO};font-size:11px;"
                  f"color:{ACCENT};font-weight:600'>{dim_text}</span>"),
            showarrow=False, align="center",
        ))
        annotations.append(dict(
            xref="paper", yref="paper",
            x=cx, y=bot_y - 0.095,
            text=(f"<span style='font-family:{SANS};font-size:9.5px;"
                  f"color:{MID};font-style:italic'>{caption}</span>"),
            showarrow=False, align="center",
        ))

    def add_arrow(x_from: float, x_to: float, y: float, verb: str) -> None:
        x0 = x_from + 0.006
        x1 = x_to - 0.006
        shapes.append(dict(
            type="line", xref="paper", yref="paper",
            x0=x0, x1=x1, y0=y, y1=y,
            line=dict(color=MID, width=1.0),
        ))
        ah_x = 0.005
        ah_y = 0.013
        shapes.append(dict(
            type="path", xref="paper", yref="paper",
            path=(f"M {x1},{y} "
                  f"L {x1 - ah_x * 1.7},{y + ah_y} "
                  f"L {x1 - ah_x * 1.7},{y - ah_y} Z"),
            fillcolor=MID, line=dict(color=MID, width=0),
        ))
        annotations.append(dict(
            xref="paper", yref="paper",
            x=(x_from + x_to) / 2, y=y + 0.042,
            text=(f"<span style='font-family:{SANS};font-size:10.5px;"
                  f"color:{MID};font-style:italic'>{verb}</span>"),
            showarrow=False, align="center",
        ))

    # ------------------------------------------------------------------
    # Glyph 1 — MEASURE: abundance matrix (wide gridded rectangle)
    # ------------------------------------------------------------------
    cx = slot_cxs[0]
    gw, gh = 0.110, 0.30
    g_x0, g_x1 = cx - gw / 2, cx + gw / 2
    g_y0, g_y1 = cy - gh / 2, cy + gh / 2
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=g_x0, x1=g_x1, y0=g_y0, y1=g_y1,
        line=dict(color=INK, width=1.3), fillcolor="#f6f6f6",
    ))
    n_c, n_r = 5, 10
    for k in range(1, n_c):
        xk = g_x0 + gw * k / n_c
        shapes.append(dict(
            type="line", xref="paper", yref="paper",
            x0=xk, x1=xk, y0=g_y0, y1=g_y1,
            line=dict(color="#d8d8d8", width=0.4),
        ))
    for k in range(1, n_r):
        yk = g_y0 + gh * k / n_r
        shapes.append(dict(
            type="line", xref="paper", yref="paper",
            x0=g_x0, x1=g_x1, y0=yk, y1=yk,
            line=dict(color="#d8d8d8", width=0.4),
        ))
    rng = np.random.default_rng(3)
    for _ in range(14):
        ci = int(rng.integers(0, n_c))
        ri = int(rng.integers(0, n_r))
        intensity = float(rng.uniform(0.25, 0.85))
        cell_x0 = g_x0 + ci * gw / n_c
        cell_x1 = g_x0 + (ci + 1) * gw / n_c
        cell_y0 = g_y0 + ri * gh / n_r
        cell_y1 = g_y0 + (ri + 1) * gh / n_r
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=cell_x0, x1=cell_x1, y0=cell_y0, y1=cell_y1,
            line=dict(color="rgba(0,0,0,0)", width=0),
            fillcolor=fill_color(intensity * 0.7),
        ))
    annotations.append(dict(
        xref="paper", yref="paper",
        x=cx, y=g_y1 + 0.018,
        text=(f"<span style='font-family:{MONO};font-size:8.5px;"
              f"color:{MID}'>samples →</span>"),
        showarrow=False,
    ))
    annotations.append(dict(
        xref="paper", yref="paper",
        x=g_x0 - 0.013, y=cy, textangle=-90,
        text=(f"<span style='font-family:{MONO};font-size:8.5px;"
              f"color:{MID}'>proteins</span>"),
        showarrow=False,
    ))
    station_label(
        cx, g_y1 + 0.020, g_y0,
        "MEASURE", "3,264 × 436",
        "abundance matrix · C9·25  spor·294  hc·91",
    )

    # ------------------------------------------------------------------
    # Arrow 1 → 2
    # ------------------------------------------------------------------
    add_arrow(g_x1, slot_cxs[1] - 0.014, cy, "regress")

    # ------------------------------------------------------------------
    # Glyph 2 — TEST: per-protein effect vector (thin tall column)
    # ------------------------------------------------------------------
    cx = slot_cxs[1]
    gw, gh = 0.028, 0.30
    g_x0, g_x1 = cx - gw / 2, cx + gw / 2
    g_y0, g_y1 = cy - gh / 2, cy + gh / 2
    n_cells = 24
    rng = np.random.default_rng(11)
    raw_t = np.abs(rng.standard_normal(n_cells))
    intensities_vec = (raw_t / raw_t.max()) ** 0.95
    for k, intensity in enumerate(intensities_vec):
        yk0 = g_y0 + gh * k / n_cells
        yk1 = g_y0 + gh * (k + 1) / n_cells
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=g_x0, x1=g_x1, y0=yk0, y1=yk1,
            line=dict(color="rgba(0,0,0,0)", width=0),
            fillcolor=fill_color(float(intensity) * 0.92),
        ))
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=g_x0, x1=g_x1, y0=g_y0, y1=g_y1,
        line=dict(color=INK, width=1.3), fillcolor="rgba(0,0,0,0)",
    ))
    annotations.append(dict(
        xref="paper", yref="paper",
        x=g_x1 + 0.008, y=cy, textangle=-90,
        text=(f"<span style='font-family:{MONO};font-size:8.5px;"
              f"color:{MID}'>|t|</span>"),
        showarrow=False,
    ))
    station_label(
        cx, g_y1 + 0.020, g_y0,
        "TEST", "3,264 × 3",
        "effect size · per protein · per comparison",
    )

    # ------------------------------------------------------------------
    # Arrow 2 → 3
    # ------------------------------------------------------------------
    add_arrow(g_x1 + 0.008, slot_cxs[2] - 0.075, cy, "join graph")

    # ------------------------------------------------------------------
    # Glyph 3 — CONNECT: small regulatory subgraph (anchor + neighbors)
    # ------------------------------------------------------------------
    cx = slot_cxs[2]
    r1y, r2y = 0.080, 0.140
    r1x, r2x = r1y * aspect, r2y * aspect
    inner_ang = np.linspace(0, 2 * np.pi, 6, endpoint=False) + 0.4
    inner_xs = cx + r1x * np.cos(inner_ang)
    inner_ys = cy + r1y * np.sin(inner_ang)
    outer_ang = np.array([0.5, 1.6, 2.8, 3.9, 5.1])
    outer_xs = cx + r2x * np.cos(outer_ang)
    outer_ys = cy + r2y * np.sin(outer_ang)
    for ix, iy in zip(inner_xs, inner_ys):
        shapes.append(dict(
            type="line", xref="paper", yref="paper",
            x0=cx, x1=ix, y0=cy, y1=iy,
            line=dict(color="#a0a0a0", width=0.7),
        ))
    for ix, iy, ox, oy in zip(inner_xs[:5], inner_ys[:5], outer_xs, outer_ys):
        shapes.append(dict(
            type="line", xref="paper", yref="paper",
            x0=ix, x1=ox, y0=iy, y1=oy,
            line=dict(color="#a0a0a0", width=0.6),
        ))
    scatter_specs.append(dict(
        x=[cx], y=[cy],
        marker=dict(symbol="circle", size=13, color=ACCENT,
                    line=dict(color=ACCENT, width=1.5)),
    ))
    scatter_specs.append(dict(
        x=inner_xs.tolist(), y=inner_ys.tolist(),
        marker=dict(symbol="circle", size=9, color="white",
                    line=dict(color=INK, width=1.2)),
    ))
    scatter_specs.append(dict(
        x=outer_xs.tolist(), y=outer_ys.tolist(),
        marker=dict(symbol="circle", size=7, color="white",
                    line=dict(color=MID, width=1.0)),
    ))
    g_y_top = cy + r2y + 0.020
    g_y_bot = cy - r2y - 0.018
    g_x_right = cx + r2x
    station_label(
        cx, g_y_top, g_y_bot,
        "CONNECT", "~129K edges",
        "regulatory adjacency · INDRA",
    )

    # ------------------------------------------------------------------
    # Arrow 3 → 4
    # ------------------------------------------------------------------
    add_arrow(g_x_right + 0.005, slot_cxs[3] - 0.075, cy, "shell test")

    # ------------------------------------------------------------------
    # Glyph 4 — CONCENTRATE: anchor + two rings + tiny slope readout
    # ------------------------------------------------------------------
    cx = slot_cxs[3]
    r1y, r2y = 0.062, 0.115
    r1x, r2x = r1y * aspect, r2y * aspect
    theta = np.linspace(0, 2 * np.pi, 100)
    scatter_specs.append(dict(
        x=(cx + r1x * np.cos(theta)).tolist(),
        y=(cy + r1y * np.sin(theta)).tolist(),
        mode="lines",
        line=dict(color=LIGHT, width=0.7, dash="dot"),
    ))
    scatter_specs.append(dict(
        x=(cx + r2x * np.cos(theta)).tolist(),
        y=(cy + r2y * np.sin(theta)).tolist(),
        mode="lines",
        line=dict(color=LIGHT, width=0.7, dash="dot"),
    ))
    scatter_specs.append(dict(
        x=[cx], y=[cy],
        marker=dict(symbol="circle", size=11, color=ACCENT,
                    line=dict(color=ACCENT, width=1.5)),
    ))
    inner_ang = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    inner_xs = cx + r1x * np.cos(inner_ang)
    inner_ys = cy + r1y * np.sin(inner_ang)
    inner_intensities = [0.85, 0.70, 0.95, 0.60, 0.80]
    scatter_specs.append(dict(
        x=inner_xs.tolist(), y=inner_ys.tolist(),
        marker=dict(
            symbol="circle", size=10,
            color=[fill_color(c * 0.95) for c in inner_intensities],
            line=dict(color=INK, width=0.8),
        ),
    ))
    outer_ang = np.linspace(0, 2 * np.pi, 8, endpoint=False) + 0.39
    outer_xs = cx + r2x * np.cos(outer_ang)
    outer_ys = cy + r2y * np.sin(outer_ang)
    outer_intensities = [0.30, 0.25, 0.40, 0.35, 0.20, 0.45, 0.30, 0.25]
    scatter_specs.append(dict(
        x=outer_xs.tolist(), y=outer_ys.tolist(),
        marker=dict(
            symbol="circle", size=8,
            color=[fill_color(c * 0.85) for c in outer_intensities],
            line=dict(color=MID, width=0.6),
        ),
    ))
    # Slope readout — to the right of the bullseye
    inset_x0 = cx + r2x + 0.012
    inset_y0 = cy - 0.015
    p1 = (inset_x0, inset_y0 + 0.020)
    p2 = (inset_x0 + 0.028, inset_y0 - 0.005)
    scatter_specs.append(dict(
        x=[p1[0], p2[0]], y=[p1[1], p2[1]],
        mode="lines+markers",
        line=dict(color=ACCENT, width=1.6),
        marker=dict(size=5, color=ACCENT, line=dict(width=0)),
    ))
    annotations.append(dict(
        xref="paper", yref="paper",
        x=(p1[0] + p2[0]) / 2, y=inset_y0 - 0.030,
        text=(f"<span style='font-family:{MONO};font-size:9px;"
              f"color:{ACCENT};font-weight:600'>slope</span>"),
        showarrow=False,
    ))
    g_y_top = cy + r2y + 0.020
    g_y_bot = cy - r2y - 0.035
    g_x_right = max(cx + r2x, p2[0])
    station_label(
        cx, g_y_top, g_y_bot,
        "CONCENTRATE", "3,257 slopes × 3",
        "per anchor · per comparison",
    )

    # ------------------------------------------------------------------
    # Arrow 4 → 5
    # ------------------------------------------------------------------
    add_arrow(g_x_right + 0.012, slot_cxs[4] - 0.080, cy, "rank pathways")

    # ------------------------------------------------------------------
    # Glyph 5 — RANK: pathway bars, top three above the family-wise line
    # ------------------------------------------------------------------
    cx = slot_cxs[4]
    n_bars = 10
    bar_h = 0.022
    bar_gap = 0.004
    total_h = n_bars * bar_h + (n_bars - 1) * bar_gap
    g_y_top = cy + total_h / 2
    g_y_bot = cy - total_h / 2
    widths = np.array(
        [0.140, 0.135, 0.128, 0.108, 0.100, 0.075, 0.063, 0.052, 0.045, 0.038]
    )
    pass_n = 3
    g_x_left = cx - widths.max() / 2
    for k in range(n_bars):
        bar_y_top = g_y_top - k * (bar_h + bar_gap)
        bar_y_bot = bar_y_top - bar_h
        color = ACCENT if k < pass_n else "#c2c2c2"
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=g_x_left, x1=g_x_left + widths[k],
            y0=bar_y_bot, y1=bar_y_top,
            line=dict(color="rgba(0,0,0,0)", width=0),
            fillcolor=color,
        ))
    shapes.append(dict(
        type="line", xref="paper", yref="paper",
        x0=g_x_left, x1=g_x_left,
        y0=g_y_bot - 0.005, y1=g_y_top + 0.005,
        line=dict(color=INK, width=1.0),
    ))
    annotations.append(dict(
        xref="paper", yref="paper",
        x=g_x_left + widths.max() + 0.010,
        y=g_y_top - bar_h * pass_n / 2,
        text=(f"<span style='font-family:{MONO};font-size:9px;"
              f"color:{ACCENT};font-weight:600'>PASS</span>"),
        showarrow=False, xanchor="left",
    ))
    annotations.append(dict(
        xref="paper", yref="paper",
        x=g_x_left + widths.max() + 0.010,
        y=g_y_top - (pass_n + (n_bars - pass_n) / 2) * (bar_h + bar_gap)
            + bar_gap / 2,
        text=(f"<span style='font-family:{MONO};font-size:9px;"
              f"color:{MID}'>fail</span>"),
        showarrow=False, xanchor="left",
    ))
    station_label(
        cx, g_y_top + 0.020, g_y_bot,
        "RANK", "8 × 3",
        "pathway PASS / fail",
    )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    for spec in scatter_specs:
        spec.setdefault("mode", "markers")
        fig.add_trace(go.Scatter(
            hoverinfo="skip", showlegend=False, **spec
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b style='font-family:{SANS};font-size:18px;color:{INK}'>"
                "Analysis pipeline</b>  "
                f"<span style='font-family:{SANS};font-size:12px;color:{MID}'>"
                "each glyph is the shape of the data the station emits"
                "</span>"
            ),
            x=0.025, xanchor="left",
        ),
        xaxis=dict(visible=False, range=[0, 1], fixedrange=True),
        yaxis=dict(visible=False, range=[0, 1], fixedrange=True),
        shapes=shapes,
        annotations=annotations,
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        margin=dict(l=M_L, r=M_R, t=M_T, b=M_B),
        width=FIG_W, height=FIG_H,
        showlegend=False,
    )
    return fig


if __name__ == "__main__":
    from pathlib import Path
    out_dir = Path(__file__).resolve().parents[2] / "output" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_pipeline_fig()
    out_path = out_dir / "arch_pipeline.html"
    fig.write_html(
        str(out_path),
        include_plotlyjs="cdn", full_html=True,
        config=dict(displaylogo=False),
    )
    print(f"wrote {out_path}")
