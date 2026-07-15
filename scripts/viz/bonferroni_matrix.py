"""Bonferroni-8 same-cohort consistency matrix — the headline image.

8 discovery-derived frozen cluster terms × 3 comparisons. Colored by raw p
on a log scale, annotated with NES, with a pass/fail border marking
the eightfold reporting threshold (raw_p < 0.00625).

The visual story: the C9 columns are saturated with passing cells in
warm colors; the sporadic-vs-healthy column is uniformly cool with
no passes.  Triangulation in one image.

Output: output/viz/bonferroni_matrix.html  (open in any browser)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

if __package__:
    from . import common as _common  # type: ignore[attr-defined]  # noqa: E402
else:
    import common as _common  # type: ignore[no-redef]  # noqa: E402

ALPHA_FAMILY = _common.ALPHA_FAMILY
ALPHA_PER_TEST = _common.ALPHA_PER_TEST
BONFERRONI_8 = _common.BONFERRONI_8
CONTRAST_ORDER = _common.CONTRAST_ORDER
N_TERMS = _common.N_TERMS
CANONICAL_TERMS = _common.TERMS


# Discovery-derived frozen terms, ordered by biological cluster. Numeric results
# come through common.py from the tracked production publication manifest.
TERMS = [(cluster, short, full) for cluster, short, full, _ in CANONICAL_TERMS]

DATA = BONFERRONI_8
CONTRASTS = CONTRAST_ORDER

# Cluster band colors (for the row-label background)
CLUSTER_BAND = {
    "Splicing":  "rgba(125, 175, 220, 0.18)",   # cool blue
    "Chromatin": "rgba(200, 145, 220, 0.18)",   # cool purple
    "Transport": "rgba(220, 165, 110, 0.18)",   # warm tan
}


def build_arrays():
    """Build (n_terms, n_contrasts) arrays of NES, raw_p, and pass."""
    nes = np.zeros((len(TERMS), len(CONTRASTS)))
    raw_p = np.zeros_like(nes)
    for i, (_, short, _) in enumerate(TERMS):
        for j, c in enumerate(CONTRASTS):
            n, p = DATA[c][short]
            nes[i, j] = n
            raw_p[i, j] = p
    passing = (raw_p < ALPHA_PER_TEST) & (nes > 0)
    return nes, raw_p, passing


def main() -> None:
    nes, raw_p, passing = build_arrays()
    short_labels = [t[1] for t in TERMS]
    clusters = [t[0] for t in TERMS]

    # Color heatmap = -log10(raw_p).  Caps at 3 (p=0.001 floor of the
    # permutation null) to keep all "p=0.001" cells visually distinct
    # from anything weaker.
    color_values = -np.log10(raw_p)

    # Annotations: NES on each cell.
    annotations = []
    for i, term_short in enumerate(short_labels):
        for j, contrast in enumerate(CONTRASTS):
            text = f"<b>{nes[i, j]:.2f}</b>"
            color = "white" if color_values[i, j] > 1.5 else "#333"
            annotations.append(dict(
                x=contrast, y=term_short,
                text=text, showarrow=False,
                font=dict(color=color, size=14, family="Arial, sans-serif"),
            ))

    # Cluster band shapes on the y-axis (one rectangle per cluster).
    shapes = []
    # Find consecutive runs of clusters.
    i = 0
    while i < len(clusters):
        j = i
        while j + 1 < len(clusters) and clusters[j + 1] == clusters[i]:
            j += 1
        cluster = clusters[i]
        shapes.append(dict(
            type="rect",
            xref="paper", yref="y",
            x0=-0.30, x1=-0.005,
            y0=i - 0.5, y1=j + 0.5,
            fillcolor=CLUSTER_BAND[cluster],
            line=dict(width=0),
            layer="below",
        ))
        # Cluster label as an annotation in the band
        annotations.append(dict(
            xref="paper", yref="y",
            x=-0.155, y=(i + j) / 2,
            text=f"<b>{cluster}</b>",
            showarrow=False,
            font=dict(size=12, color="#555", family="Arial, sans-serif"),
            xanchor="center", yanchor="middle",
        ))
        i = j + 1

    # Border highlight on passing cells.  Plotly heatmap doesn't have
    # per-cell borders, so we use rect shapes layered on top.
    for i in range(len(short_labels)):
        for j in range(len(CONTRASTS)):
            if passing[i, j]:
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=j - 0.48, x1=j + 0.48,
                    y0=i - 0.48, y1=i + 0.48,
                    line=dict(color="#000", width=3),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                ))

    # Custom hover text per cell
    hover_text = [[None] * len(CONTRASTS) for _ in range(len(short_labels))]
    for i, t in enumerate(short_labels):
        for j, c in enumerate(CONTRASTS):
            pass_str = "PASS (Bonferroni-8)" if passing[i, j] else "fail"
            hover_text[i][j] = (
                f"<b>{t}</b><br>{c}<br>"
                f"NES: {nes[i, j]:.3f}<br>"
                f"raw p: {raw_p[i, j]:.4f}<br>"
                f"eightfold reporting threshold: {ALPHA_PER_TEST:.5f}<br>"
                f"<b>{pass_str}</b>"
            )

    # Per-contrast pass counts for the column subtitle
    pass_counts = passing.sum(axis=0)
    contrast_labels = [
        f"<b>{c}</b><br><span style='font-size:11px;color:#666'>"
        f"{pass_counts[j]}/8 pass</span>"
        for j, c in enumerate(CONTRASTS)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=color_values,
        x=contrast_labels,
        y=short_labels,
        colorscale=[
            [0.0,  "#f4f4f4"],   # very weak signal
            [0.15, "#dceaf5"],
            [0.30, "#94c2e0"],
            [0.50, "#3d8fc0"],
            [0.75, "#1f5da0"],
            [1.0,  "#0a3b80"],   # p ≤ 1e-3
        ],
        zmin=0, zmax=3,
        colorbar=dict(
            title=dict(text="-log₁₀(p)", side="right"),
            tickvals=[0, 0.5, 1, np.log10(1/ALPHA_PER_TEST), 2, 3],
            ticktext=["1", "0.32", "0.10", f"<b>{ALPHA_PER_TEST:.5f}</b><br>(threshold)", "0.01", "≤ 0.001"],
            len=0.6,
            thickness=14,
        ),
        text=hover_text,
        hoverinfo="text",
        xgap=2, ygap=2,
    ))

    fig.update_layout(
        title=dict(
            text=(
                "<b>Fixed-panel same-cohort consistency</b><br>"
                "<span style='font-size:13px;color:#666'>"
                "8 discovery-derived frozen terms × 3 group comparisons. "
                "Cells in cells = NES; color = -log₁₀(raw p); thick black "
                "border = passes the eightfold reporting threshold (raw p < 0.00625). "
                "No post-selection FWER guarantee; internal consistency, not independent confirmation."
                "</span>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=18),
        ),
        xaxis=dict(
            side="top",
            tickfont=dict(size=12),
            showgrid=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12),
            showgrid=False,
        ),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=240, r=180, t=140, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1080, height=540,
    )

    out = ROOT / "output" / "viz" / "bonferroni_matrix.html"
    fig.write_html(
        str(out),
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(displaylogo=False, modeBarButtonsToRemove=["lasso2d", "select2d"]),
    )
    print(f"Wrote {out}")
    print(f"Pass counts: c9spor={pass_counts[0]}/8  "
          f"c9ctrl={pass_counts[1]}/8  spctrl={pass_counts[2]}/8")


if __name__ == "__main__":
    main()
