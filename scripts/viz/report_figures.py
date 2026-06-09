"""Static SVG figures for the standalone C9 triangulation report.

Each function builds a self-contained SVG string for one figure slot
in output/c9_triangulation_report.html.  Pure geometry — no Plotly,
no JS — keeps the report a single shareable file.

Conventions:
- Palette matches the report CSS (ink, muted, accent, cluster colors).
- Glyphs ARE their data shape; text is reserved for one-word station
  labels and figcaptions (in the report).
- viewBox sizes are in design units; SVG scales to container via
  width="100%".
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

# ---------------------------------------------------------------------
# Palette — matches output/c9_triangulation_report.html CSS variables
# ---------------------------------------------------------------------
INK         = "#1a1a1a"
TEXT        = "#2a2a2a"
MUTED       = "#6b6b6b"
FAINT       = "#a8a8a8"
LIGHT       = "#d8d8d8"
PAPER       = "#faf9f6"
CARD        = "#ffffff"
RULE        = "#e8e6df"
ACCENT      = "#0a3b80"
ACCENT_SOFT = "#cad9ef"

SANS = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, "
        "Arial, sans-serif")
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"


def _fill(intensity: float) -> str:
    """Interpolate white → ACCENT by intensity ∈ [0, 1]."""
    r = int(255 - (255 - 10) * intensity)
    g = int(255 - (255 - 59) * intensity)
    b = int(255 - (255 - 128) * intensity)
    return f"rgb({r},{g},{b})"


# =====================================================================
# Figure 1 — Analysis pipeline
# =====================================================================

def build_fig1_pipeline() -> str:
    """Five-station dataflow.  Each glyph IS the shape of the data it
    emits; the eye reads dimensionality collapse across stations.

    No data dimensions, no operation captions, no arrow verbs — the
    figcaption in the report carries verbal load.  One-word station
    labels above each glyph."""
    W, H = 720, 230
    GLYPH_CY = 130
    LABEL_Y = 50
    STATION_CXS = [80, 220, 360, 504, 644]
    STATIONS = ["MEASURE", "TEST", "CONNECT", "CONCENTRATE", "RANK"]

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="100%" height="auto" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'role="img" aria-label="Analysis pipeline: five-station dataflow '
        f'from abundance matrix to ranked pathway outcomes.">'
    )
    # Background — slight transparency so it inherits page tone
    parts.append(
        f'<rect width="{W}" height="{H}" fill="{CARD}"/>'
    )

    # ---- Station labels ----
    for cx, name in zip(STATION_CXS, STATIONS):
        parts.append(
            f'<text x="{cx}" y="{LABEL_Y}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="12" font-weight="700" '
            f'fill="{INK}" letter-spacing="1.4">{name}</text>'
        )

    # ---- Glyphs ----
    edges: list[tuple[float, float]] = []  # (left, right) per station
    cx0 = STATION_CXS[0]
    edges.append(_glyph_matrix(parts, cx0, GLYPH_CY))
    cx1 = STATION_CXS[1]
    edges.append(_glyph_vector(parts, cx1, GLYPH_CY))
    cx2 = STATION_CXS[2]
    edges.append(_glyph_network(parts, cx2, GLYPH_CY))
    cx3 = STATION_CXS[3]
    edges.append(_glyph_concentrate(parts, cx3, GLYPH_CY))
    cx4 = STATION_CXS[4]
    edges.append(_glyph_rank(parts, cx4, GLYPH_CY))

    # ---- Arrows between stations ----
    for i in range(4):
        right_i = edges[i][1]
        left_next = edges[i + 1][0]
        _arrow(parts, right_i, left_next, GLYPH_CY)

    parts.append('</svg>')
    return "\n".join(parts)


def _arrow(parts: list[str], x0: float, x1: float, y: float) -> None:
    start = x0 + 6
    end = x1 - 8
    parts.append(
        f'<line x1="{start}" y1="{y}" x2="{end}" y2="{y}" '
        f'stroke="{MUTED}" stroke-width="1"/>'
    )
    # Triangular arrowhead
    parts.append(
        f'<polygon points="{end},{y} {end - 6},{y - 4} {end - 6},{y + 4}" '
        f'fill="{MUTED}" stroke="none"/>'
    )


def _glyph_matrix(parts: list[str], cx: float, cy: float) -> tuple[float, float]:
    """MEASURE — wide gridded rectangle with some filled cells."""
    w, h = 60, 100
    x0 = cx - w / 2
    y0 = cy - h / 2
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" '
        f'fill="#f5f4ef" stroke="{INK}" stroke-width="1.2"/>'
    )
    n_cols, n_rows = 4, 8
    for k in range(1, n_cols):
        xk = x0 + w * k / n_cols
        parts.append(
            f'<line x1="{xk}" y1="{y0}" x2="{xk}" y2="{y0 + h}" '
            f'stroke="{LIGHT}" stroke-width="0.4"/>'
        )
    for k in range(1, n_rows):
        yk = y0 + h * k / n_rows
        parts.append(
            f'<line x1="{x0}" y1="{yk}" x2="{x0 + w}" y2="{yk}" '
            f'stroke="{LIGHT}" stroke-width="0.4"/>'
        )
    # Filled cells (deterministic pattern, suggests real data)
    cells = [
        (0, 1, 0.55), (1, 3, 0.7), (2, 0, 0.4), (3, 2, 0.65),
        (0, 5, 0.5), (2, 6, 0.75), (3, 7, 0.45), (1, 4, 0.6),
        (0, 7, 0.35), (2, 2, 0.55), (3, 5, 0.7),
    ]
    cw = w / n_cols
    ch = h / n_rows
    for ci, ri, intensity in cells:
        cx_ = x0 + ci * cw
        cy_ = y0 + ri * ch
        parts.append(
            f'<rect x="{cx_}" y="{cy_}" width="{cw}" height="{ch}" '
            f'fill="{_fill(intensity * 0.75)}" stroke="none"/>'
        )
    # Re-stroke grid (over filled cells)
    for k in range(1, n_cols):
        xk = x0 + w * k / n_cols
        parts.append(
            f'<line x1="{xk}" y1="{y0}" x2="{xk}" y2="{y0 + h}" '
            f'stroke="{LIGHT}" stroke-width="0.4" opacity="0.6"/>'
        )
    for k in range(1, n_rows):
        yk = y0 + h * k / n_rows
        parts.append(
            f'<line x1="{x0}" y1="{yk}" x2="{x0 + w}" y2="{yk}" '
            f'stroke="{LIGHT}" stroke-width="0.4" opacity="0.6"/>'
        )
    return x0, x0 + w


def _glyph_vector(parts: list[str], cx: float, cy: float) -> tuple[float, float]:
    """TEST — thin tall column with intensity-shaded cells."""
    w, h = 16, 100
    x0 = cx - w / 2
    y0 = cy - h / 2
    intensities = [
        0.6, 0.3, 0.85, 0.5, 0.4, 0.75, 0.25, 0.55, 0.9, 0.4,
        0.3, 0.65, 0.5, 0.8, 0.45, 0.35, 0.7, 0.55, 0.25, 0.6,
    ]
    n = len(intensities)
    ch = h / n
    for k, intensity in enumerate(intensities):
        yk = y0 + k * ch
        parts.append(
            f'<rect x="{x0}" y="{yk}" width="{w}" height="{ch}" '
            f'fill="{_fill(intensity * 0.9)}" stroke="none"/>'
        )
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" '
        f'fill="none" stroke="{INK}" stroke-width="1.2"/>'
    )
    return x0, x0 + w


def _glyph_network(parts: list[str], cx: float, cy: float) -> tuple[float, float]:
    """CONNECT — small graph: accent anchor + 6 inner neighbors + 4 outer."""
    inner_r = 24
    outer_r = 44
    inner_angles = [0.3, 1.4, 2.5, 3.6, 4.7, 5.8]
    outer_angles = [0.8, 2.2, 3.5, 5.0]
    inner = [(cx + inner_r * math.cos(a), cy + inner_r * math.sin(a))
             for a in inner_angles]
    outer = [(cx + outer_r * math.cos(a), cy + outer_r * math.sin(a))
             for a in outer_angles]
    # Edges anchor → inner
    for ix, iy in inner:
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{ix}" y2="{iy}" '
            f'stroke="#9a9a9a" stroke-width="0.8"/>'
        )
    # Edges inner → outer (closest pairing)
    for ox, oy in outer:
        nn = min(inner, key=lambda p: (p[0] - ox) ** 2 + (p[1] - oy) ** 2)
        parts.append(
            f'<line x1="{nn[0]}" y1="{nn[1]}" x2="{ox}" y2="{oy}" '
            f'stroke="#9a9a9a" stroke-width="0.7"/>'
        )
    # Anchor (accent, distinctive)
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="6.5" fill="{ACCENT}" '
        f'stroke="{ACCENT}" stroke-width="1"/>'
    )
    # Inner nodes (white-filled, dark outline)
    for ix, iy in inner:
        parts.append(
            f'<circle cx="{ix}" cy="{iy}" r="4.5" fill="{CARD}" '
            f'stroke="{INK}" stroke-width="1.1"/>'
        )
    # Outer nodes (smaller, lighter outline)
    for ox, oy in outer:
        parts.append(
            f'<circle cx="{ox}" cy="{oy}" r="3.5" fill="{CARD}" '
            f'stroke="{MUTED}" stroke-width="0.9"/>'
        )
    return cx - outer_r, cx + outer_r


def _glyph_concentrate(parts: list[str], cx: float, cy: float) -> tuple[float, float]:
    """CONCENTRATE — bullseye: anchor + 2 dashed rings + |t|-shaded dots."""
    r1, r2 = 20, 36
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r1}" fill="none" '
        f'stroke="{LIGHT}" stroke-width="0.7" stroke-dasharray="2,3"/>'
    )
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r2}" fill="none" '
        f'stroke="{LIGHT}" stroke-width="0.7" stroke-dasharray="2,3"/>'
    )
    # Anchor
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="6" fill="{ACCENT}" '
        f'stroke="{ACCENT}" stroke-width="1"/>'
    )
    # Inner ring nodes — high |t|, mostly dark accent
    inner_data = [(0.10, 0.85), (1.40, 0.70), (2.60, 0.95),
                  (3.90, 0.60), (5.20, 0.80)]
    for ang, intensity in inner_data:
        ix = cx + r1 * math.cos(ang)
        iy = cy + r1 * math.sin(ang)
        parts.append(
            f'<circle cx="{ix}" cy="{iy}" r="4.5" '
            f'fill="{_fill(intensity * 0.9)}" '
            f'stroke="{INK}" stroke-width="0.7"/>'
        )
    # Outer ring nodes — low |t|, lighter
    outer_data = [(0.20, 0.30), (1.00, 0.25), (1.80, 0.40),
                  (2.70, 0.35), (3.50, 0.20), (4.30, 0.45),
                  (5.10, 0.30), (5.90, 0.25)]
    for ang, intensity in outer_data:
        ox = cx + r2 * math.cos(ang)
        oy = cy + r2 * math.sin(ang)
        parts.append(
            f'<circle cx="{ox}" cy="{oy}" r="3.5" '
            f'fill="{_fill(intensity * 0.85)}" '
            f'stroke="{MUTED}" stroke-width="0.5"/>'
        )
    return cx - r2, cx + r2


# =====================================================================
# Figure 3 — Shell-statistic geometry (anchor + rings, projection,
#            slope readout)
# =====================================================================

def build_fig3_shell() -> str:
    """Two coupled panels showing the per-anchor concentration test.

    Layout grid (explicit zones; nothing crosses boundaries):
      viewBox: 720 × 380
      MASTHEAD            y ∈ [0, 70]    (panel headers at y=42)
      CONTENT             y ∈ [70, 320]  (anchors at y=200; rings, chart)
      AXIS CAPTIONS       y ∈ [320, 360]
      BOTTOM MARGIN       y ∈ [360, 380]

      LEFT PANEL          x ∈ [20, 320]  (anchor at x=170; r1=50, r2=85)
      CONNECTOR           x ∈ [320, 400]
      RIGHT PANEL         x ∈ [400, 700] (chart x=480-690)

    All text widths estimated against zone width before placement.
    The slope-line text lives in the empty upper-right quadrant of
    the chart, where neither data points nor axes contest the space.
    The excluded path goes upper-left from the anchor, with its
    caption above the far node where the wide left margin holds it."""
    W, H = 720, 380

    ring1_t = [2.85, 2.30, 3.10, 1.95, 2.55]
    ring2_t = [1.10, 0.70, 1.30, 0.90, 0.55, 1.05, 0.85]
    max_t = 3.5

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="100%" height="auto" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'role="img" aria-label="Shell-statistic geometry: regulatory '
        f'neighborhood with anchor and rings of measured neighbors, '
        f'projected onto ring number versus mean t-statistic with the '
        f'slope shown as a line.">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="{CARD}"/>')

    # =================================================================
    # LEFT PANEL — regulatory neighborhood
    # =================================================================
    L_cx, L_cy = 170, 200
    r1, r2 = 50, 85

    parts.append(
        f'<text x="{L_cx}" y="42" text-anchor="middle" '
        f'font-family="{SANS}" font-size="12" font-weight="700" '
        f'fill="{INK}" letter-spacing="1.4">REGULATORY NEIGHBORHOOD</text>'
    )

    # Dashed ring contours
    for r in (r1, r2):
        parts.append(
            f'<circle cx="{L_cx}" cy="{L_cy}" r="{r}" fill="none" '
            f'stroke="{LIGHT}" stroke-width="0.8" stroke-dasharray="2,3"/>'
        )

    # Ring 1 node positions (5 evenly spaced)
    ring1_pos: list[tuple[float, float]] = []
    for i in range(len(ring1_t)):
        ang = (i / len(ring1_t)) * 2 * math.pi + 0.42
        ring1_pos.append((
            L_cx + r1 * math.cos(ang),
            L_cy + r1 * math.sin(ang),
        ))

    # Ring 2 node positions — angles chosen to LEAVE the upper-left
    # quadrant open for the excluded-path callout
    ring2_angles = [0.20, 0.95, 1.70, 2.45, 3.20, 4.20, 5.40]
    ring2_pos = [
        (L_cx + r2 * math.cos(a), L_cy + r2 * math.sin(a))
        for a in ring2_angles
    ]

    # Solid edges: anchor → ring 1
    for x, y in ring1_pos:
        parts.append(
            f'<line x1="{L_cx}" y1="{L_cy}" x2="{x:.2f}" y2="{y:.2f}" '
            f'stroke="#8a8a8a" stroke-width="1"/>'
        )
    # Solid edges: ring 1 → ring 2
    for x2, y2 in ring2_pos:
        nn = min(ring1_pos,
                 key=lambda p: (p[0] - x2) ** 2 + (p[1] - y2) ** 2)
        parts.append(
            f'<line x1="{nn[0]:.2f}" y1="{nn[1]:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#a0a0a0" stroke-width="0.8"/>'
        )

    # Excluded path: anchor → unmeasured (X) → far node, dashed
    # Direction: upper-LEFT (open quadrant). Caption goes ABOVE the
    # far node, centered, with the 80-px-wide caption fitting in
    # the 60-130 px x-range comfortably.
    via_angle = math.pi * 1.22   # upper-left
    via_x = L_cx + 0.62 * r2 * math.cos(via_angle)
    via_y = L_cy + 0.62 * r2 * math.sin(via_angle)
    far_x = L_cx + 1.20 * r2 * math.cos(math.pi * 1.20)
    far_y = L_cy + 1.20 * r2 * math.sin(math.pi * 1.20)
    parts.append(
        f'<line x1="{L_cx}" y1="{L_cy}" x2="{via_x:.2f}" y2="{via_y:.2f}" '
        f'stroke="#bcbcbc" stroke-width="1.3" stroke-dasharray="4,3"/>'
    )
    parts.append(
        f'<line x1="{via_x:.2f}" y1="{via_y:.2f}" '
        f'x2="{far_x:.2f}" y2="{far_y:.2f}" '
        f'stroke="#bcbcbc" stroke-width="1.3" stroke-dasharray="4,3"/>'
    )
    # Unmeasured intermediate — white circle with accent X overlay
    parts.append(
        f'<circle cx="{via_x:.2f}" cy="{via_y:.2f}" r="7" '
        f'fill="{CARD}" stroke="{LIGHT}" stroke-width="1.3"/>'
    )
    arm = 4.5
    parts.append(
        f'<line x1="{via_x - arm:.2f}" y1="{via_y - arm:.2f}" '
        f'x2="{via_x + arm:.2f}" y2="{via_y + arm:.2f}" '
        f'stroke="{ACCENT}" stroke-width="2.4"/>'
    )
    parts.append(
        f'<line x1="{via_x - arm:.2f}" y1="{via_y + arm:.2f}" '
        f'x2="{via_x + arm:.2f}" y2="{via_y - arm:.2f}" '
        f'stroke="{ACCENT}" stroke-width="2.4"/>'
    )
    # Far node — faint, smaller
    parts.append(
        f'<circle cx="{far_x:.2f}" cy="{far_y:.2f}" r="5.5" '
        f'fill="{CARD}" stroke="{LIGHT}" stroke-width="1.1"/>'
    )
    # Caption ABOVE the far node (centered).  "EXCLUDED" ~62px,
    # "via unmeasured" ~78px — both fit within left-panel zone
    # when centered at far_x.
    parts.append(
        f'<text x="{far_x:.2f}" y="{far_y - 18:.2f}" text-anchor="middle" '
        f'font-family="{SANS}" font-size="11" font-weight="700" '
        f'fill="{ACCENT}" letter-spacing="0.9">EXCLUDED</text>'
    )
    parts.append(
        f'<text x="{far_x:.2f}" y="{far_y - 5:.2f}" text-anchor="middle" '
        f'font-family="{SANS}" font-size="9.5" fill="{MUTED}" '
        f'font-style="italic">via unmeasured</text>'
    )

    # Anchor — black square with white "A"
    a_size = 16
    parts.append(
        f'<rect x="{L_cx - a_size / 2}" y="{L_cy - a_size / 2}" '
        f'width="{a_size}" height="{a_size}" fill="{INK}" stroke="none"/>'
    )
    parts.append(
        f'<text x="{L_cx}" y="{L_cy + 4}" text-anchor="middle" '
        f'font-family="{SANS}" font-size="11" font-weight="700" '
        f'fill="white">A</text>'
    )

    # Ring 1 nodes
    for (x, y), t in zip(ring1_pos, ring1_t):
        intensity = (t / max_t) * 0.95
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" '
            f'fill="{_fill(intensity)}" '
            f'stroke="{INK}" stroke-width="1"/>'
        )
    # Ring 2 nodes
    for (x, y), t in zip(ring2_pos, ring2_t):
        intensity = (t / max_t) * 0.92
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" '
            f'fill="{_fill(intensity)}" '
            f'stroke="{MUTED}" stroke-width="0.9"/>'
        )

    # Ring labels — placed in the lower-right empty arc
    parts.append(
        f'<text x="{L_cx + r1 * 0.65:.2f}" y="{L_cy + r1 + 8:.2f}" '
        f'text-anchor="middle" font-family="{MONO}" font-size="10" '
        f'fill="{MUTED}" letter-spacing="0.5">ring 1</text>'
    )
    parts.append(
        f'<text x="{L_cx + r2 * 0.20:.2f}" y="{L_cy + r2 + 12:.2f}" '
        f'text-anchor="middle" font-family="{MONO}" font-size="10" '
        f'fill="{MUTED}" letter-spacing="0.5">ring 2</text>'
    )
    # Anchor label
    parts.append(
        f'<text x="{L_cx + a_size / 2 + 4:.2f}" y="{L_cy + 4:.2f}" '
        f'text-anchor="start" font-family="{SANS}" font-size="10" '
        f'fill="{MUTED}" font-style="italic">anchor</text>'
    )

    # =================================================================
    # CONNECTOR — arrow with verb
    # =================================================================
    arrow_y = L_cy
    a_x0, a_x1 = 330, 400
    parts.append(
        f'<line x1="{a_x0}" y1="{arrow_y}" x2="{a_x1 - 7}" y2="{arrow_y}" '
        f'stroke="{MUTED}" stroke-width="1"/>'
    )
    parts.append(
        f'<polygon points="{a_x1 - 7},{arrow_y} '
        f'{a_x1 - 13},{arrow_y - 4.5} '
        f'{a_x1 - 13},{arrow_y + 4.5}" '
        f'fill="{MUTED}" stroke="none"/>'
    )
    parts.append(
        f'<text x="{(a_x0 + a_x1) / 2}" y="{arrow_y - 12}" '
        f'text-anchor="middle" font-family="{SANS}" font-size="11" '
        f'fill="{MUTED}" font-style="italic">project to (ring, |t|)</text>'
    )

    # =================================================================
    # RIGHT PANEL — shell concentration readout
    # =================================================================
    R_x0, R_x1 = 405, 700
    parts.append(
        f'<text x="{(R_x0 + R_x1) / 2}" y="42" text-anchor="middle" '
        f'font-family="{SANS}" font-size="12" font-weight="700" '
        f'fill="{INK}" letter-spacing="1.4">SHELL CONCENTRATION READOUT</text>'
    )

    # Chart bounds (safe zone for axes + data)
    chart_x0 = 470
    chart_x1 = 685
    chart_y0 = 305   # bottom
    chart_y1 = 115   # top
    x_data_min, x_data_max = 0.55, 2.55
    y_data_min, y_data_max = 0.0, 3.6

    def d2p(xd: float, yd: float) -> tuple[float, float]:
        px = chart_x0 + (xd - x_data_min) / (x_data_max - x_data_min) \
            * (chart_x1 - chart_x0)
        py = chart_y0 - (yd - y_data_min) / (y_data_max - y_data_min) \
            * (chart_y0 - chart_y1)
        return px, py

    # Axes
    parts.append(
        f'<line x1="{chart_x0}" y1="{chart_y0}" x2="{chart_x1}" y2="{chart_y0}" '
        f'stroke="{INK}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{chart_x0}" y1="{chart_y0}" x2="{chart_x0}" y2="{chart_y1}" '
        f'stroke="{INK}" stroke-width="1.2"/>'
    )
    # X ticks 1, 2
    for ring_n in (1, 2):
        px, _ = d2p(ring_n, 0)
        parts.append(
            f'<line x1="{px:.2f}" y1="{chart_y0}" '
            f'x2="{px:.2f}" y2="{chart_y0 + 6}" '
            f'stroke="{INK}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{chart_y0 + 20}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="11" fill="{INK}">{ring_n}</text>'
        )
    # Y ticks 0..3
    for yt in (0, 1, 2, 3):
        _, py = d2p(0, yt)
        parts.append(
            f'<line x1="{chart_x0 - 6}" y1="{py:.2f}" '
            f'x2="{chart_x0}" y2="{py:.2f}" '
            f'stroke="{INK}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{chart_x0 - 10}" y="{py + 4:.2f}" text-anchor="end" '
            f'font-family="{MONO}" font-size="10" fill="{INK}">{yt}</text>'
        )
    # X-axis label
    parts.append(
        f'<text x="{(chart_x0 + chart_x1) / 2}" y="{chart_y0 + 42}" '
        f'text-anchor="middle" font-family="{SANS}" font-size="11" '
        f'fill="{INK}">ring number</text>'
    )
    # Y-axis label — group with translate+rotate so it renders vertical
    y_pivot_x = chart_x0 - 28
    y_pivot_y = (chart_y0 + chart_y1) / 2
    parts.append(
        f'<g transform="translate({y_pivot_x}, {y_pivot_y}) rotate(-90)">'
        f'<text x="0" y="0" text-anchor="middle" font-family="{SANS}" '
        f'font-size="11" fill="{INK}">|t|</text>'
        f'</g>'
    )

    # Scatter (deterministic jitter)
    ring1_jitter = [0.10, -0.07, 0.14, -0.12, 0.05]
    for (t, jit) in zip(ring1_t, ring1_jitter):
        intensity = (t / max_t) * 0.95
        px, py = d2p(1 + jit, t)
        parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.5" '
            f'fill="{_fill(intensity)}" '
            f'stroke="{INK}" stroke-width="0.9"/>'
        )
    ring2_jitter = [0.09, -0.13, 0.06, -0.05, 0.14, -0.08, 0.11]
    for (t, jit) in zip(ring2_t, ring2_jitter):
        intensity = (t / max_t) * 0.92
        px, py = d2p(2 + jit, t)
        parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.5" '
            f'fill="{_fill(intensity)}" '
            f'stroke="{MUTED}" stroke-width="0.7"/>'
        )

    # Per-ring means + slope line
    mean1 = sum(ring1_t) / len(ring1_t)
    mean2 = sum(ring2_t) / len(ring2_t)
    slope = mean2 - mean1
    px1, py1 = d2p(1, mean1)
    px2, py2 = d2p(2, mean2)
    parts.append(
        f'<line x1="{px1:.2f}" y1="{py1:.2f}" '
        f'x2="{px2:.2f}" y2="{py2:.2f}" '
        f'stroke="{ACCENT}" stroke-width="2.6"/>'
    )
    for px, py in ((px1, py1), (px2, py2)):
        s = 7
        parts.append(
            f'<polygon points="{px:.2f},{py - s:.2f} '
            f'{px + s:.2f},{py:.2f} '
            f'{px:.2f},{py + s:.2f} '
            f'{px - s:.2f},{py:.2f}" '
            f'fill="{ACCENT}" stroke="{INK}" stroke-width="1.1"/>'
        )
    # Mean readouts — fit-budgeted: "2.55" ~28px to the left of diamond 1,
    # "0.92" ~28px to the right of diamond 2.  Both within chart bounds.
    parts.append(
        f'<text x="{px1 - 12:.2f}" y="{py1 - 10:.2f}" text-anchor="end" '
        f'font-family="{MONO}" font-size="11" font-weight="700" '
        f'fill="{ACCENT}">{mean1:.2f}</text>'
    )
    parts.append(
        f'<text x="{px2 + 12:.2f}" y="{py2 - 10:.2f}" text-anchor="start" '
        f'font-family="{MONO}" font-size="11" font-weight="700" '
        f'fill="{ACCENT}">{mean2:.2f}</text>'
    )

    # Slope readout — placed in the empty upper-right quadrant of the
    # chart.  Width budget: "slope = -1.63" at 14pt bold mono ≈ 108px;
    # caption at 10pt italic sans ≈ 110px / 80px on two lines.
    # Anchor at x=580, leaving 105px to chart_x1=685.  Three lines:
    sx = 580
    parts.append(
        f'<text x="{sx}" y="148" text-anchor="start" font-family="{MONO}" '
        f'font-size="14" font-weight="700" fill="{ACCENT}">'
        f'slope = {slope:+.2f}</text>'
    )
    parts.append(
        f'<text x="{sx}" y="163" text-anchor="start" font-family="{SANS}" '
        f'font-size="10" fill="{MUTED}" font-style="italic">'
        f'perturbation falls off</text>'
    )
    parts.append(
        f'<text x="{sx}" y="176" text-anchor="start" font-family="{SANS}" '
        f'font-size="10" fill="{MUTED}" font-style="italic">'
        f'with regulatory distance</text>'
    )

    parts.append('</svg>')
    return "\n".join(parts)


# =====================================================================
# Figure 2 — Cohort composition (donor dot grid)
# =====================================================================

def build_fig2_cohort() -> str:
    """Three donor-dot blocks, stacked vertically; 1 dot = 1 donor.

    Each block's row-count is proportional to N at a fixed row width,
    so block height carries the imbalance: C9 is one short row;
    sporadic spans ten.  Color singles C9 out (accent) against neutral
    grays for the other two groups — the methodological constraint
    (C9 is rare) becomes the perceptual headline."""
    W, H = 720, 340
    DOTS_PER_ROW = 30
    DOT_R = 4
    CELL_PITCH = 12  # horizontal and vertical
    BLOCK_W = DOTS_PER_ROW * CELL_PITCH  # 360
    BLOCK_X0 = (W - BLOCK_W) / 2          # 180

    groups = [
        ("C9-ALS",        25,  ACCENT,
         "carries a confirmed C9orf72 mutation or repeat expansion ≥ 30"),
        ("Healthy",       91,  "#a8a8a8",
         "no ALS or related neurodegenerative diagnosis"),
        ("Sporadic ALS", 294,  "#6b6b6b",
         "ALS without a known causal mutation"),
    ]

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="100%" height="auto" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'role="img" aria-label="Cohort composition: 25 C9-ALS, 91 healthy, '
        f'294 sporadic ALS donors; one dot per donor.">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="{CARD}"/>')

    y = 32
    for name, n, color, desc in groups:
        # Block label: NAME · N donors  ·  short descriptor (muted)
        parts.append(
            f'<text x="{BLOCK_X0}" y="{y}" font-family="{SANS}" '
            f'font-size="13" font-weight="700" fill="{color}" '
            f'letter-spacing="0.6">'
            f'{name.upper()}'
            f'<tspan fill="{INK}" font-weight="700" dx="10" '
            f'font-size="13" letter-spacing="0">·  {n} donors</tspan>'
            f'<tspan fill="{MUTED}" font-weight="400" dx="10" '
            f'font-size="11.5" letter-spacing="0" font-style="italic">'
            f'{desc}</tspan>'
            f'</text>'
        )
        # Dot grid
        dots_top = y + 14
        for i in range(n):
            row = i // DOTS_PER_ROW
            col = i % DOTS_PER_ROW
            cx = BLOCK_X0 + col * CELL_PITCH + CELL_PITCH / 2
            cy_dot = dots_top + row * CELL_PITCH + CELL_PITCH / 2
            parts.append(
                f'<circle cx="{cx}" cy="{cy_dot}" r="{DOT_R}" '
                f'fill="{color}" stroke="none"/>'
            )
        n_rows = (n + DOTS_PER_ROW - 1) // DOTS_PER_ROW
        block_h = n_rows * CELL_PITCH
        y = dots_top + block_h + 20

    # Bottom monospace summary
    parts.append(
        f'<text x="{BLOCK_X0}" y="{H - 14}" font-family="{MONO}" '
        f'font-size="11" fill="{MUTED}" letter-spacing="0.4">'
        f'410 donors  ·  436 PBMC samples  ·  3,264 proteins'
        f'</text>'
    )

    parts.append('</svg>')
    return "\n".join(parts)


def _glyph_rank(parts: list[str], cx: float, cy: float) -> tuple[float, float]:
    """RANK — sorted horizontal bars, top 3 in accent."""
    n_bars = 10
    bar_h = 7
    bar_gap = 2.5
    total_h = n_bars * bar_h + (n_bars - 1) * bar_gap
    y_start = cy - total_h / 2
    widths = [60, 56, 52, 40, 35, 25, 21, 17, 14, 11]
    x_left = cx - widths[0] / 2
    pass_n = 3
    for i, w in enumerate(widths):
        y_top = y_start + i * (bar_h + bar_gap)
        color = ACCENT if i < pass_n else "#c2c2c2"
        parts.append(
            f'<rect x="{x_left}" y="{y_top}" width="{w}" '
            f'height="{bar_h}" fill="{color}" stroke="none"/>'
        )
    # Y-axis baseline (rank axis)
    parts.append(
        f'<line x1="{x_left}" y1="{y_start - 3}" x2="{x_left}" '
        f'y2="{y_start + total_h + 3}" stroke="{INK}" stroke-width="1"/>'
    )
    return x_left, x_left + widths[0]


# =====================================================================
# Figure 4 — Bonferroni-8 confirmatory matrix
#   (HTML + CSS + JS; interactive tooltips)
# =====================================================================

def build_fig4_matrix() -> dict[str, str]:
    """The headline figure: 8 pre-registered cluster pathway terms × 3
    group comparisons, with per-cell NES + colored by -log10(raw p) +
    thick black border on family-wise passes.

    Returns dict with three pieces that get embedded into the report:
      - 'html': matrix + legend markup (replaces #fig-matrix placeholder)
      - 'css':  matrix-specific styles (appended to report <style>)
      - 'js':   tooltip hover handler (appended before </body>)

    Built as inline HTML/CSS/JS rather than Plotly/SVG so it inherits
    the report's design tokens directly and stays single-file standalone."""

    # ---- Data: term metadata + per-contrast results --------------------
    TERMS = [
        ("Splicing",  "mRNA Splicing",
         "mRNA Splicing"),
        ("Splicing",  "Processing Capped Pre-mRNA",
         "Processing of Capped Intron-Containing Pre-mRNA"),
        ("Splicing",  "mRNA splicing, via spliceosome",
         "mRNA splicing, via spliceosome"),
        ("Chromatin", "chromosome",
         "chromosome"),
        ("Chromatin", "chromatin",
         "chromatin"),
        ("Transport", "nucleocytoplasmic transport",
         "nucleocytoplasmic transport"),
        ("Transport", "nuclear pore",
         "nuclear pore"),
        ("Transport", "Vpr-mediated nuclear import",
         "Vpr-mediated nuclear import of PICs"),
    ]
    # Transcribed from output/wave_24l_confirmatory.md
    DATA = {
        "C9 vs Sporadic": {
            "mRNA Splicing":                  (2.41, 0.0010),
            "Processing Capped Pre-mRNA":     (2.51, 0.0010),
            "mRNA splicing, via spliceosome": (2.38, 0.0010),
            "chromosome":                     (2.64, 0.0010),
            "chromatin":                      (2.48, 0.0010),
            "nucleocytoplasmic transport":    (2.10, 0.0010),
            "nuclear pore":                   (1.82, 0.0055),
            "Vpr-mediated nuclear import":    (1.63, 0.0116),
        },
        "C9 vs Healthy": {
            "mRNA Splicing":                  (2.08, 0.0010),
            "Processing Capped Pre-mRNA":     (2.17, 0.0010),
            "mRNA splicing, via spliceosome": (2.04, 0.0010),
            "chromosome":                     (3.14, 0.0010),
            "chromatin":                      (2.95, 0.0010),
            "nucleocytoplasmic transport":    (1.83, 0.0010),
            "nuclear pore":                   (1.78, 0.0210),
            "Vpr-mediated nuclear import":    (1.34, 0.1388),
        },
        "Sporadic vs Healthy": {
            "mRNA Splicing":                  (1.29, 0.1136),
            "Processing Capped Pre-mRNA":     (1.35, 0.0652),
            "mRNA splicing, via spliceosome": (1.03, 0.4266),
            "chromosome":                     (0.75, 0.9369),
            "chromatin":                      (0.75, 0.8909),
            "nucleocytoplasmic transport":    (1.09, 0.3403),
            "nuclear pore":                   (1.49, 0.0446),
            "Vpr-mediated nuclear import":    (1.44, 0.0543),
        },
    }
    ALPHA = 0.05 / 8  # 0.00625
    CONTRASTS = ["C9 vs Sporadic", "C9 vs Healthy", "Sporadic vs Healthy"]
    CLUSTER_TAG = {"Splicing": "SPLI", "Chromatin": "CHR", "Transport": "TRA"}

    # ---- Color scale (matches bonferroni_matrix.py palette) ------------
    SCALE = [
        (0.00, (244, 244, 244)),
        (0.15, (220, 234, 245)),
        (0.30, (148, 194, 224)),
        (0.50, (61, 143, 192)),
        (0.75, (31, 93, 160)),
        (1.00, (10, 59, 128)),
    ]

    def color_for_intensity(t: float) -> str:
        t = max(0.0, min(1.0, t))
        for i in range(len(SCALE) - 1):
            s0, c0 = SCALE[i]
            s1, c1 = SCALE[i + 1]
            if s0 <= t <= s1:
                if s1 == s0:
                    return f"rgb({c0[0]},{c0[1]},{c0[2]})"
                f = (t - s0) / (s1 - s0)
                r = int(c0[0] + (c1[0] - c0[0]) * f)
                g = int(c0[1] + (c1[1] - c0[1]) * f)
                b = int(c0[2] + (c1[2] - c0[2]) * f)
                return f"rgb({r},{g},{b})"
        return f"rgb({SCALE[-1][1][0]},{SCALE[-1][1][1]},{SCALE[-1][1][2]})"

    # ---- Per-contrast pass count for column headers --------------------
    pass_counts = {
        c: sum(
            1 for term in TERMS
            if DATA[c][term[1]][1] < ALPHA and DATA[c][term[1]][0] > 0
        )
        for c in CONTRASTS
    }

    # ---- Assemble HTML --------------------------------------------------
    parts: list[str] = ['<div class="fig4-matrix">']
    parts.append('<div class="m-corner"></div>')
    for c in CONTRASTS:
        parts.append(
            f'<div class="m-colhead">'
            f'<span class="m-name">{c}</span>'
            f'<span class="m-count">{pass_counts[c]}/8 pass</span>'
            f'</div>'
        )
    prev_cluster: str | None = None
    for cluster, short, full in TERMS:
        is_new = (cluster != prev_cluster and prev_cluster is not None)
        prev_cluster = cluster
        cluster_lc = cluster.lower()
        cluster_tag = CLUSTER_TAG[cluster]
        new_class = ' new-cluster' if is_new else ''
        parts.append(
            f'<div class="m-rowhead {cluster_lc}{new_class}">'
            f'<span class="m-cluster">{cluster_tag}</span>'
            f'<span class="m-term">{short}</span>'
            f'</div>'
        )
        for c in CONTRASTS:
            nes, raw_p = DATA[c][short]
            intensity = -math.log10(raw_p) / 3.0
            bg = color_for_intensity(intensity)
            text_color = "#ffffff" if intensity > 0.45 else "#1a1a1a"
            passes = raw_p < ALPHA and nes > 0
            cell_classes = f'm-cell{new_class}'
            if passes:
                cell_classes += ' pass'
            parts.append(
                f'<div class="{cell_classes}" '
                f'style="background:{bg};color:{text_color}" '
                f'data-term="{full}" '
                f'data-contrast="{c}" '
                f'data-nes="{nes:.3f}" '
                f'data-p="{raw_p:.4f}" '
                f'data-passes="{str(passes).lower()}">'
                f'{nes:.2f}'
                f'</div>'
            )
    parts.append('</div>')

    # ---- Legend below the grid -----------------------------------------
    threshold_pct = 100 * (-math.log10(ALPHA) / 3.0)  # ~73.3
    legend = f'''
<div class="fig4-legend">
  <div class="m-legend-block">
    <div class="m-legend-bar"><span class="m-legend-threshold-mark" style="left:{threshold_pct:.1f}%"></span></div>
    <div class="m-legend-ticks">
      <span class="tick-left">p = 1</span>
      <span class="tick-threshold" style="left:{threshold_pct:.1f}%">p = {ALPHA:.5f}<br>(threshold)</span>
      <span class="tick-right">p &le; 0.001</span>
    </div>
    <div class="m-legend-label">color · −log<sub>10</sub>(raw p)</div>
  </div>
  <div class="m-legend-note">
    <span class="m-legend-swatch"></span>
    <span>thick black border = passes family-wise threshold (raw p &lt; {ALPHA:.5f} <em>and</em> NES &gt; 0)</span>
  </div>
</div>'''

    html = "\n".join(parts) + legend

    # ---- CSS (uses report's existing tokens via var(--...)) ------------
    css = '''
.fig4-matrix {
  display: grid;
  grid-template-columns: 220px 1fr 1fr 1fr;
  gap: 5px;
  font-family: var(--font-sans);
  min-width: 580px;
}
.m-corner { /* spacer, no content */ }
.m-colhead {
  padding: 10px 12px 12px;
  text-align: center;
  border-bottom: 2px solid var(--ink);
}
.m-colhead .m-name {
  display: block;
  font-size: 13px;
  font-weight: 700;
  color: var(--ink);
  letter-spacing: -0.1px;
}
.m-colhead .m-count {
  display: block;
  margin-top: 4px;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--muted);
}
.m-rowhead {
  padding: 14px 12px 14px 8px;
  text-align: right;
  display: flex;
  align-items: center;
  gap: 10px;
  justify-content: flex-end;
  font-size: 13px;
}
.m-rowhead .m-cluster {
  font-family: var(--font-mono);
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 1px;
  flex-shrink: 0;
}
.m-rowhead.splicing  .m-cluster { color: var(--splicing); }
.m-rowhead.chromatin .m-cluster { color: var(--chromatin); }
.m-rowhead.transport .m-cluster { color: var(--transport); }
.m-rowhead .m-term {
  color: var(--ink);
  font-weight: 500;
  line-height: 1.3;
}
.m-rowhead.new-cluster,
.m-cell.new-cluster {
  margin-top: 6px;
  border-top: 1px solid var(--rule);
  padding-top: 19px;
}
.m-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 15px;
  padding: 16px 8px;
  border-radius: 3px;
  cursor: help;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
  border: 2px solid transparent;
  position: relative;
}
.m-cell.pass {
  border: 2.5px solid var(--ink);
}
.m-cell:hover {
  transform: scale(1.06);
  box-shadow: 0 5px 18px rgba(0, 0, 30, 0.22);
  z-index: 5;
}

.fig4-legend {
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid var(--rule);
}
.m-legend-block {
  margin-bottom: 16px;
}
.m-legend-bar {
  height: 14px;
  border-radius: 2px;
  background: linear-gradient(
    to right,
    #f4f4f4 0%, #dceaf5 15%, #94c2e0 30%,
    #3d8fc0 50%, #1f5da0 75%, #0a3b80 100%
  );
  border: 1px solid var(--rule);
  position: relative;
}
.m-legend-threshold-mark {
  position: absolute;
  top: -4px;
  bottom: -4px;
  width: 2px;
  background: var(--ink);
  transform: translateX(-1px);
}
.m-legend-ticks {
  position: relative;
  height: 30px;
  margin-top: 4px;
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--muted);
}
.m-legend-ticks .tick-left {
  position: absolute; left: 0; top: 2px;
}
.m-legend-ticks .tick-right {
  position: absolute; right: 0; top: 2px;
}
.m-legend-ticks .tick-threshold {
  position: absolute;
  top: 2px;
  transform: translateX(-50%);
  text-align: center;
  color: var(--ink);
  font-weight: 700;
  line-height: 1.25;
}
.m-legend-label {
  margin-top: 14px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink);
  font-weight: 600;
}
.m-legend-note {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: var(--muted);
}
.m-legend-swatch {
  display: inline-block;
  width: 22px;
  height: 14px;
  background: #94c2e0;
  border: 2.5px solid var(--ink);
  border-radius: 2px;
  flex-shrink: 0;
}

.m-tooltip {
  display: none;
  position: absolute;
  z-index: 100;
  background: #1a1a1a;
  color: white;
  padding: 12px 15px;
  border-radius: 5px;
  box-shadow: 0 6px 24px rgba(0, 0, 30, 0.25);
  font-size: 12px;
  line-height: 1.5;
  max-width: 300px;
  pointer-events: none;
  font-family: var(--font-sans);
}
.m-tooltip .tt-term {
  font-weight: 700;
  font-size: 13px;
  margin-bottom: 8px;
  color: white;
  letter-spacing: 0.2px;
}
.m-tooltip .tt-meta {
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: #c0c0c0;
  line-height: 1.65;
}
.m-tooltip .tt-meta b {
  color: white;
  font-weight: 700;
}
.m-tooltip .tt-status {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid #444;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  font-size: 11px;
}
.m-tooltip .tt-status.pass { color: #7ab0e8; }
.m-tooltip .tt-status.fail { color: #999; }

@media (max-width: 640px) {
  figure#fig-matrix .frame { overflow-x: auto; }
  .fig4-matrix { grid-template-columns: 180px 1fr 1fr 1fr; }
}
'''

    # ---- JS (vanilla, no deps) -----------------------------------------
    js = '''(function() {
  var tooltip = document.getElementById('fig4-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'fig4-tooltip';
    tooltip.className = 'm-tooltip';
    document.body.appendChild(tooltip);
  }
  var ALPHA = 0.00625;
  var cells = document.querySelectorAll('.fig4-matrix .m-cell');
  cells.forEach(function(cell) {
    cell.addEventListener('mouseenter', function() {
      var term     = cell.dataset.term;
      var contrast = cell.dataset.contrast;
      var nes      = cell.dataset.nes;
      var p        = cell.dataset.p;
      var passes   = cell.dataset.passes === 'true';
      var status   = passes ? 'PASS' : 'fail';
      var sclass   = passes ? 'pass' : 'fail';
      tooltip.innerHTML =
        '<div class="tt-term">' + term + '</div>' +
        '<div class="tt-meta">' +
          'comparison: <b>' + contrast + '</b><br>' +
          'NES:&nbsp;&nbsp;&nbsp;<b>' + nes + '</b><br>' +
          'raw p:&nbsp;<b>' + p + '</b><br>' +
          'threshold: <b>' + ALPHA.toFixed(5) + '</b>' +
        '</div>' +
        '<div class="tt-status ' + sclass + '">' + status + '</div>';
      tooltip.style.display = 'block';
      var rect = cell.getBoundingClientRect();
      var ttRect = tooltip.getBoundingClientRect();
      var left = rect.right + window.scrollX + 12;
      if (left + ttRect.width > window.innerWidth - 20) {
        left = rect.left + window.scrollX - ttRect.width - 12;
      }
      var top = rect.top + window.scrollY + (rect.height - ttRect.height) / 2;
      if (top < window.scrollY + 10) top = window.scrollY + 10;
      tooltip.style.left = left + 'px';
      tooltip.style.top  = top  + 'px';
    });
    cell.addEventListener('mouseleave', function() {
      tooltip.style.display = 'none';
    });
  });
})();'''

    return {"html": html, "css": css, "js": js}


# =====================================================================
# Figure 5 — Per-cluster anatomy (top proteins, signed t × 3 contrasts)
#   Data computed inline from proteomics + INDRA at build time;
#   embedded as inline HTML/CSS/JS with no external files at runtime.
# =====================================================================

def _compute_fig5_data(top_n: int = 10) -> list[dict]:
    """Run the proteomics + INDRA pipeline and return top-N proteins
    per cluster with per-contrast t-statistics and pathway memberships.

    Returns: [
        {"cluster": "Splicing",
         "total_measured": int,
         "proteins": [
             {"symbol": str, "uniprot": str,
              "t_c9spor": float, "t_c9hlthy": float, "t_spctrl": float,
              "mean_abs_c9": float,
              "memberships": [short_term, ...]},
             ...
         ]},
        ...
    ]"""
    import sys
    import pandas as pd

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    # local imports — common.py lives next to this file
    from common import (
        TERMS, CONTRAST_ORDER, CONTRAST_GROUPS,
        resolve_groups, fit_per_protein_t,
        fetch_term_members_via_indra, hgnc_ids_to_uniprots,
        uniprot_to_hgnc_symbol,
    )

    root = here.parents[1]
    df_data = pd.read_csv(root / "output/proteomics/all_als.data.csv",
                          index_col=0)
    md = pd.read_csv(root / "output/proteomics/all_als.metadata.csv",
                     index_col=0)
    groups = resolve_groups(md)

    t_stats = {}
    for c in CONTRAST_ORDER:
        contrast = CONTRAST_GROUPS[c]
        t_stats[c] = fit_per_protein_t(df_data, md, groups, contrast)

    term_ids = [t[3] for t in TERMS]
    hgnc_members = fetch_term_members_via_indra(term_ids)
    cluster_members_by_term = {
        tid: hgnc_ids_to_uniprots(hgncs)
        for tid, hgncs in hgnc_members.items()
    }

    measured_proteins = list(df_data.index)
    sym_lookup = uniprot_to_hgnc_symbol(measured_proteins)
    measured_set = set(measured_proteins)

    CLUSTERS = ["Splicing", "Chromatin", "Transport"]
    result = []
    for cluster_name in CLUSTERS:
        cluster_term_ids = [t[3] for t in TERMS if t[0] == cluster_name]
        all_members: set[str] = set()
        member_to_terms: dict[str, list[str]] = {}
        for tid in cluster_term_ids:
            short = next(t[1] for t in TERMS if t[3] == tid)
            members = cluster_members_by_term.get(tid, set())
            for u in members:
                all_members.add(u)
                member_to_terms.setdefault(u, []).append(short)

        measured_members = sorted(all_members & measured_set)
        records = []
        for u in measured_members:
            t1 = float(t_stats["C9 vs Sporadic"].get(u, 0.0))
            t2 = float(t_stats["C9 vs Healthy"].get(u, 0.0))
            t3 = float(t_stats["Sporadic vs Healthy"].get(u, 0.0))
            mean_abs = (abs(t1) + abs(t2)) / 2.0
            records.append({
                "uniprot": u,
                "symbol": sym_lookup.get(u, u),
                "t_c9spor":  t1,
                "t_c9hlthy": t2,
                "t_spctrl":  t3,
                "mean_abs_c9": mean_abs,
                "memberships": member_to_terms.get(u, []),
            })
        records.sort(key=lambda r: r["mean_abs_c9"], reverse=True)
        result.append({
            "cluster": cluster_name,
            "total_measured": len(measured_members),
            "proteins": records[:top_n],
        })
    return result


def build_fig5_anatomy(data: list[dict]) -> dict[str, str]:
    """Pure builder: render per-cluster grouped-bar HTML + CSS + JS.

    Each cluster section is a vertical stack of protein rows.  Each
    row has 3 stacked horizontal bars (one per contrast), with a
    central zero line.  Hover row → tooltip with values + memberships."""

    # Axis range: round up max |t| across all displayed proteins to nearest 0.5
    max_t = 0.0
    for cluster_data in data:
        for p in cluster_data["proteins"]:
            for k in ("t_c9spor", "t_c9hlthy", "t_spctrl"):
                if abs(p[k]) > max_t:
                    max_t = abs(p[k])
    axis_max = math.ceil(max_t * 2) / 2
    if axis_max < 2.5:
        axis_max = 2.5

    parts: list[str] = ['<div class="fig5-anatomy">']

    for cluster_data in data:
        cluster = cluster_data["cluster"]
        cluster_lc = cluster.lower()
        n_shown = len(cluster_data["proteins"])
        n_total = cluster_data["total_measured"]

        parts.append(f'<div class="a-cluster {cluster_lc}">')

        # Header: cluster name + count
        parts.append(
            f'<div class="a-cluster-header">'
            f'<span class="a-cluster-name">{cluster}</span>'
            f'<span class="a-cluster-count">top {n_shown} of {n_total} measured members</span>'
            f'</div>'
        )

        # Inline legend — 3 swatches sized like the bars
        parts.append(
            f'<div class="a-legend">'
            f'<span class="a-swatch primary"></span>'
            f'<span class="a-leg-label">C9 vs Sporadic</span>'
            f'<span class="a-swatch secondary"></span>'
            f'<span class="a-leg-label">C9 vs Healthy</span>'
            f'<span class="a-swatch null"></span>'
            f'<span class="a-leg-label">Sporadic vs Healthy</span>'
            f'</div>'
        )

        # Bar rows
        parts.append('<div class="a-rows">')
        for p in cluster_data["proteins"]:
            sym = p["symbol"]
            t1 = p["t_c9spor"]
            t2 = p["t_c9hlthy"]
            t3 = p["t_spctrl"]
            mems = ", ".join(p["memberships"])

            def bar(t: float, cls: str) -> str:
                pct = min(abs(t) / axis_max * 50, 50.0)
                if t >= 0:
                    return (f'<div class="a-bar {cls}" '
                            f'style="left:50%;width:{pct:.2f}%"></div>')
                else:
                    return (f'<div class="a-bar {cls}" '
                            f'style="right:50%;width:{pct:.2f}%"></div>')

            parts.append(
                f'<div class="a-row" '
                f'data-symbol="{sym}" '
                f'data-t1="{t1:.2f}" '
                f'data-t2="{t2:.2f}" '
                f'data-t3="{t3:.2f}" '
                f'data-memberships="{mems}">'
                f'<div class="a-label">{sym}</div>'
                f'<div class="a-bars">'
                f'<div class="a-zero"></div>'
                f'{bar(t1, "primary")}'
                f'{bar(t2, "secondary")}'
                f'{bar(t3, "null")}'
                f'</div>'
                f'</div>'
            )
        parts.append('</div>')  # /a-rows

        # Axis ticks below the bars
        parts.append(
            f'<div class="a-axis">'
            f'<div class="a-axis-spacer"></div>'
            f'<div class="a-axis-track">'
            f'<span class="left">−{axis_max:g}</span>'
            f'<span class="center">0</span>'
            f'<span class="right">+{axis_max:g}</span>'
            f'</div>'
            f'</div>'
        )

        parts.append('</div>')  # /a-cluster

    parts.append('</div>')  # /fig5-anatomy

    html = "\n".join(parts)

    css = '''
.fig5-anatomy {
  font-family: var(--font-sans);
}
.a-cluster {
  margin-bottom: 24px;
}
.a-cluster + .a-cluster {
  border-top: 1px solid var(--rule);
  padding-top: 22px;
  margin-top: 22px;
}
.a-cluster-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.a-cluster-name {
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.6px;
  text-transform: uppercase;
}
.a-cluster.splicing  .a-cluster-name { color: var(--splicing); }
.a-cluster.chromatin .a-cluster-name { color: var(--chromatin); }
.a-cluster.transport .a-cluster-name { color: var(--transport); }
.a-cluster-count {
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--muted);
}
.a-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 5px 14px;
  margin-bottom: 14px;
  font-size: 11px;
  color: var(--muted);
}
.a-swatch {
  display: inline-block;
  width: 18px;
  height: 5px;
  border-radius: 1px;
  vertical-align: middle;
  margin-right: 4px;
}
.a-cluster.splicing  .a-swatch.primary   { background: var(--splicing); }
.a-cluster.splicing  .a-swatch.secondary { background: #6ea3c8; }
.a-cluster.chromatin .a-swatch.primary   { background: var(--chromatin); }
.a-cluster.chromatin .a-swatch.secondary { background: #9b7bbe; }
.a-cluster.transport .a-swatch.primary   { background: var(--transport); }
.a-cluster.transport .a-swatch.secondary { background: #ea9460; }
.a-swatch.null { background: #b0b0b0; }
.a-leg-label {
  font-family: var(--font-mono);
}

.a-rows {
  display: flex;
  flex-direction: column;
  gap: 1px;
}
.a-row {
  display: grid;
  grid-template-columns: 84px 1fr;
  gap: 12px;
  padding: 5px 6px 5px 0;
  border-radius: 3px;
  transition: background 0.1s;
  cursor: help;
}
.a-row:hover {
  background: rgba(10, 59, 128, 0.045);
}
.a-label {
  font-family: var(--font-mono);
  font-size: 12.5px;
  font-weight: 700;
  color: var(--ink);
  text-align: right;
  align-self: center;
}
.a-bars {
  position: relative;
  height: 21px;
}
.a-zero {
  position: absolute;
  top: 0; bottom: 0;
  left: 50%;
  width: 1px;
  background: var(--rule-strong);
}
.a-bar {
  position: absolute;
  height: 5px;
  border-radius: 1px;
}
.a-bar.primary   { top: 0px; }
.a-bar.secondary { top: 8px; }
.a-bar.null      { top: 16px; }
.a-cluster.splicing  .a-bar.primary   { background: var(--splicing); }
.a-cluster.splicing  .a-bar.secondary { background: #6ea3c8; }
.a-cluster.chromatin .a-bar.primary   { background: var(--chromatin); }
.a-cluster.chromatin .a-bar.secondary { background: #9b7bbe; }
.a-cluster.transport .a-bar.primary   { background: var(--transport); }
.a-cluster.transport .a-bar.secondary { background: #ea9460; }
.a-bar.null { background: #b0b0b0; }

.a-axis {
  display: grid;
  grid-template-columns: 84px 1fr;
  gap: 12px;
  margin-top: 8px;
}
.a-axis-spacer { /* aligns with .a-label column */ }
.a-axis-track {
  position: relative;
  height: 16px;
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--muted);
}
.a-axis-track .left   { position: absolute; left: 0;   top: 2px; }
.a-axis-track .center { position: absolute; left: 50%; top: 2px; transform: translateX(-50%); }
.a-axis-track .right  { position: absolute; right: 0;  top: 2px; }

.m-tooltip .tt-mems {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid #444;
  font-size: 11px;
  color: #cccccc;
  line-height: 1.5;
}
.m-tooltip .tt-mems b {
  color: white;
  font-weight: 700;
}
'''

    js = '''(function() {
  var tooltip = document.getElementById('fig4-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'fig4-tooltip';
    tooltip.className = 'm-tooltip';
    document.body.appendChild(tooltip);
  }
  function sign(v) { return v >= 0 ? '+' + v : v; }
  var rows = document.querySelectorAll('.fig5-anatomy .a-row');
  rows.forEach(function(row) {
    row.addEventListener('mouseenter', function() {
      var sym  = row.dataset.symbol;
      var t1   = parseFloat(row.dataset.t1);
      var t2   = parseFloat(row.dataset.t2);
      var t3   = parseFloat(row.dataset.t3);
      var mems = row.dataset.memberships || '';
      var memHtml = mems ? '<div class="tt-mems"><b>in:</b> ' + mems + '</div>' : '';
      tooltip.innerHTML =
        '<div class="tt-term">' + sym + '</div>' +
        '<div class="tt-meta">' +
          'C9 vs Sporadic:&nbsp;&nbsp;<b>' + sign(t1.toFixed(2)) + '</b><br>' +
          'C9 vs Healthy:&nbsp;&nbsp;&nbsp;<b>' + sign(t2.toFixed(2)) + '</b><br>' +
          'Sporadic vs Healthy: <b>' + sign(t3.toFixed(2)) + '</b>' +
        '</div>' +
        memHtml;
      tooltip.style.display = 'block';
      var rect = row.getBoundingClientRect();
      var ttRect = tooltip.getBoundingClientRect();
      var left = rect.right + window.scrollX + 12;
      if (left + ttRect.width > window.innerWidth - 20) {
        left = rect.left + window.scrollX - ttRect.width - 12;
      }
      var top = rect.top + window.scrollY + (rect.height - ttRect.height) / 2;
      if (top < window.scrollY + 10) top = window.scrollY + 10;
      tooltip.style.left = left + 'px';
      tooltip.style.top  = top  + 'px';
    });
    row.addEventListener('mouseleave', function() {
      tooltip.style.display = 'none';
    });
  });
})();'''

    return {"html": html, "css": css, "js": js}


# =====================================================================
# Figure 6 — INDRA vs STRING (diverging mini-matrices, sign flip)
# =====================================================================

def build_fig6_string() -> dict[str, str]:
    """Two side-by-side mini-matrices showing the sign flip at every
    cluster term when the regulatory graph is replaced with physical PPI.

    Same 8 cluster terms × 2 C9 contrasts on both sides; left panel
    (INDRA / regulatory) saturated blue, right panel (STRING /
    physical) saturated red. Mirror images. Diverging colormap.

    Data: T43 wave_24k alternative-network test (discovery-era values;
    internally consistent for the INDRA-vs-STRING comparison). Source:
    output/string_alternative_network.md."""

    # Row order matches Fig 4 for visual consistency across the report
    TERMS = [
        ("Splicing",  "SPLI", "mRNA Splicing"),
        ("Splicing",  "SPLI", "Processing Capped Pre-mRNA"),
        ("Splicing",  "SPLI", "mRNA splicing, via spliceosome"),
        ("Chromatin", "CHR",  "chromosome"),
        ("Chromatin", "CHR",  "chromatin"),
        ("Transport", "TRA",  "nucleocytoplasmic transport"),
        ("Transport", "TRA",  "nuclear pore"),
        ("Transport", "TRA",  "Vpr-mediated nuclear import"),
    ]
    # (indra_c9spor, indra_c9hlthy, string_c9spor, string_c9hlthy,
    #  string_q_c9spor, string_q_c9hlthy)
    DATA = {
        "mRNA Splicing":                  (+2.30, +1.94, -2.42, -2.18, "< 0.001", "< 0.001"),
        "Processing Capped Pre-mRNA":     (+2.37, +2.14, -2.79, -2.57, "< 0.001", "< 0.001"),
        "mRNA splicing, via spliceosome": (+2.21, +2.06, -1.85, -1.80, "= 0.001", "= 0.001"),
        "chromosome":                     (+2.50, +3.07, -2.02, -2.50, "< 0.001", "< 0.001"),
        "chromatin":                      (+2.35, +2.70, -1.41, -2.20, "= 0.032", "< 0.001"),
        "nucleocytoplasmic transport":    (+2.15, +2.29, -2.05, -2.72, "< 0.001", "< 0.001"),
        "nuclear pore":                   (+2.42, +2.55, -2.57, -2.95, "< 0.001", "< 0.001"),
        "Vpr-mediated nuclear import":    (+2.29, +2.56, -2.69, -2.98, "< 0.001", "< 0.001"),
    }
    CONTRASTS = ["C9 vs Sporadic", "C9 vs Healthy"]

    BLUE_STOPS = [
        (0.00, (244, 244, 244)),
        (0.15, (220, 234, 245)),
        (0.30, (148, 194, 224)),
        (0.50, (61, 143, 192)),
        (0.75, (31, 93, 160)),
        (1.00, (10, 59, 128)),
    ]
    RED_STOPS = [
        (0.00, (244, 244, 244)),
        (0.15, (250, 218, 207)),
        (0.30, (235, 168, 150)),
        (0.50, (215, 100, 75)),
        (0.75, (180, 50, 35)),
        (1.00, (135, 25, 18)),
    ]

    def color_for(value: float, scale_max: float = 3.0) -> str:
        intensity = min(1.0, abs(value) / scale_max)
        stops = BLUE_STOPS if value >= 0 else RED_STOPS
        for i in range(len(stops) - 1):
            s0, c0 = stops[i]
            s1, c1 = stops[i + 1]
            if s0 <= intensity <= s1:
                if s1 == s0:
                    return f"rgb({c0[0]},{c0[1]},{c0[2]})"
                f = (intensity - s0) / (s1 - s0)
                r = int(c0[0] + (c1[0] - c0[0]) * f)
                g = int(c0[1] + (c1[1] - c0[1]) * f)
                b = int(c0[2] + (c1[2] - c0[2]) * f)
                return f"rgb({r},{g},{b})"
        last = stops[-1][1]
        return f"rgb({last[0]},{last[1]},{last[2]})"

    parts: list[str] = ['<div class="fig6-string">']

    # ---- Header row: panel titles (each spans 2 cells) -----------------
    parts.append('<div class="g-corner"></div>')
    parts.append('<div class="g-panel-header indra">INDRA · regulatory edges</div>')
    parts.append('<div class="g-spacer"></div>')
    parts.append('<div class="g-panel-header string">STRING · physical edges</div>')

    # ---- Sub-header row: contrast names --------------------------------
    parts.append('<div class="g-corner"></div>')
    for c in CONTRASTS:
        parts.append(f'<div class="g-contrast">{c}</div>')
    parts.append('<div class="g-spacer"></div>')
    for c in CONTRASTS:
        parts.append(f'<div class="g-contrast">{c}</div>')

    # ---- Term rows -----------------------------------------------------
    prev_cluster: str | None = None
    for cluster, tag, short in TERMS:
        is_new = (cluster != prev_cluster and prev_cluster is not None)
        prev_cluster = cluster
        cluster_lc = cluster.lower()
        new_class = ' new-cluster' if is_new else ''

        parts.append(
            f'<div class="g-rowhead {cluster_lc}{new_class}">'
            f'<span class="g-cluster-tag">{tag}</span>'
            f'<span class="g-term">{short}</span>'
            f'</div>'
        )

        indra_c9s, indra_c9h, string_c9s, string_c9h, sq_c9s, sq_c9h = DATA[short]

        # INDRA cells (positive)
        for nes, contrast in ((indra_c9s, "C9 vs Sporadic"),
                              (indra_c9h, "C9 vs Healthy")):
            bg = color_for(nes)
            text_color = "#ffffff" if abs(nes) > 1.5 else "#1a1a1a"
            parts.append(
                f'<div class="g-cell{new_class}" '
                f'style="background:{bg};color:{text_color}" '
                f'data-term="{short}" '
                f'data-contrast="{contrast}" '
                f'data-graph="INDRA · regulatory" '
                f'data-nes="{nes:+.2f}" '
                f'data-q="">'
                f'{nes:+.2f}'
                f'</div>'
            )

        # Spacer (column 4)
        parts.append(f'<div class="g-spacer{new_class}"></div>')

        # STRING cells (negative)
        for nes, q, contrast in ((string_c9s, sq_c9s, "C9 vs Sporadic"),
                                 (string_c9h, sq_c9h, "C9 vs Healthy")):
            bg = color_for(nes)
            text_color = "#ffffff" if abs(nes) > 1.5 else "#1a1a1a"
            parts.append(
                f'<div class="g-cell{new_class}" '
                f'style="background:{bg};color:{text_color}" '
                f'data-term="{short}" '
                f'data-contrast="{contrast}" '
                f'data-graph="STRING · physical" '
                f'data-nes="{nes:+.2f}" '
                f'data-q="q {q}">'
                f'{nes:+.2f}'
                f'</div>'
            )

    parts.append('</div>')

    # ---- Legend: diverging scale ---------------------------------------
    legend = '''
<div class="fig6-legend">
  <div class="g-legend-row">
    <div class="g-legend-block">
      <div class="g-legend-bar blue"></div>
      <div class="g-legend-ticks"><span>0</span><span>+3</span></div>
      <div class="g-legend-cap">positive NES — perturbation concentrated near anchors</div>
    </div>
    <div class="g-legend-block">
      <div class="g-legend-bar red"></div>
      <div class="g-legend-ticks"><span>0</span><span>−3</span></div>
      <div class="g-legend-cap">negative NES — perturbation anti-concentrated</div>
    </div>
  </div>
</div>'''

    html = "\n".join(parts) + legend

    css = '''
.fig6-string {
  display: grid;
  grid-template-columns:
    190px
    minmax(58px, 1fr) minmax(58px, 1fr)
    20px
    minmax(58px, 1fr) minmax(58px, 1fr);
  gap: 4px;
  font-family: var(--font-sans);
  min-width: 540px;
}
.g-corner, .g-spacer { /* layout only */ }
.g-panel-header {
  grid-column: span 2;
  padding: 8px 10px 11px;
  text-align: center;
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: 0.5px;
  border-bottom: 2px solid var(--ink);
}
.g-panel-header.indra  { color: #0a3b80; }
.g-panel-header.string { color: #871912; }
.g-contrast {
  text-align: center;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--muted);
  padding: 8px 4px 4px;
}
.g-rowhead {
  padding: 12px 10px 12px 4px;
  text-align: right;
  display: flex;
  align-items: center;
  gap: 8px;
  justify-content: flex-end;
  font-size: 12.5px;
}
.g-cluster-tag {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  flex-shrink: 0;
}
.g-rowhead.splicing  .g-cluster-tag { color: var(--splicing); }
.g-rowhead.chromatin .g-cluster-tag { color: var(--chromatin); }
.g-rowhead.transport .g-cluster-tag { color: var(--transport); }
.g-term { color: var(--ink); font-weight: 500; line-height: 1.3; }
.g-rowhead.new-cluster,
.g-cell.new-cluster,
.g-spacer.new-cluster {
  margin-top: 4px;
  border-top: 1px solid var(--rule);
  padding-top: 15px;
}
.g-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 14px;
  padding: 13px 4px;
  border-radius: 3px;
  cursor: help;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
  border: 2px solid transparent;
}
.g-cell:hover {
  transform: scale(1.06);
  box-shadow: 0 5px 18px rgba(0, 0, 30, 0.18);
  z-index: 5;
}

.fig6-legend {
  margin-top: 22px;
  padding-top: 18px;
  border-top: 1px solid var(--rule);
}
.g-legend-row {
  display: flex;
  gap: 28px;
  flex-wrap: wrap;
}
.g-legend-block {
  flex: 1;
  min-width: 200px;
}
.g-legend-bar {
  height: 14px;
  border-radius: 2px;
  border: 1px solid var(--rule);
}
.g-legend-bar.blue {
  background: linear-gradient(to right,
    #f4f4f4 0%, #dceaf5 15%, #94c2e0 30%,
    #3d8fc0 50%, #1f5da0 75%, #0a3b80 100%);
}
.g-legend-bar.red {
  background: linear-gradient(to right,
    #f4f4f4 0%, #fadacf 15%, #eba896 30%,
    #d7644b 50%, #b43223 75%, #871912 100%);
}
.g-legend-ticks {
  display: flex;
  justify-content: space-between;
  margin-top: 4px;
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--muted);
}
.g-legend-cap {
  margin-top: 6px;
  font-size: 11.5px;
  color: var(--muted);
  font-style: italic;
  line-height: 1.4;
}

@media (max-width: 640px) {
  figure#fig-string .frame { overflow-x: auto; }
}
'''

    js = '''(function() {
  var tooltip = document.getElementById('fig4-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'fig4-tooltip';
    tooltip.className = 'm-tooltip';
    document.body.appendChild(tooltip);
  }
  document.querySelectorAll('.fig6-string .g-cell').forEach(function(cell) {
    cell.addEventListener('mouseenter', function() {
      var term     = cell.dataset.term;
      var contrast = cell.dataset.contrast;
      var graph    = cell.dataset.graph;
      var nes      = cell.dataset.nes;
      var q        = cell.dataset.q;
      var qLine    = q ? '<br>' + q : '';
      tooltip.innerHTML =
        '<div class="tt-term">' + term + '</div>' +
        '<div class="tt-meta">' +
          'graph:&nbsp;&nbsp;&nbsp;&nbsp;<b>' + graph + '</b><br>' +
          'comparison: <b>' + contrast + '</b><br>' +
          'NES:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>' + nes + '</b>' + qLine +
        '</div>';
      tooltip.style.display = 'block';
      var rect = cell.getBoundingClientRect();
      var ttRect = tooltip.getBoundingClientRect();
      var left = rect.right + window.scrollX + 12;
      if (left + ttRect.width > window.innerWidth - 20) {
        left = rect.left + window.scrollX - ttRect.width - 12;
      }
      var top = rect.top + window.scrollY + (rect.height - ttRect.height) / 2;
      if (top < window.scrollY + 10) top = window.scrollY + 10;
      tooltip.style.left = left + 'px';
      tooltip.style.top  = top  + 'px';
    });
    cell.addEventListener('mouseleave', function() {
      tooltip.style.display = 'none';
    });
  });
})();'''

    return {"html": html, "css": css, "js": js}


# =====================================================================
# Build all figures (run as module)
# =====================================================================

def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "output" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "fig1_pipeline.svg": build_fig1_pipeline,
        "fig2_cohort.svg":   build_fig2_cohort,
        "fig3_shell.svg":    build_fig3_shell,
    }
    for name, builder in figures.items():
        svg = builder()
        out_path = out_dir / name
        out_path.write_text(svg)
        print(f"wrote {out_path} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
