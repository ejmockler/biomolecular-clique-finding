"""C9 cluster explorer — single rich HTML with three sections.

Section 1: Bonferroni-8 confirmatory matrix (the wave_24l headline).
Section 2: Where the cluster terms rank in the universe of tested pathways.
Section 3: Inside each cluster — member proteins ranked by |t|.

Output: output/viz/cluster_explorer.html (open in any browser).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "viz"))

from common import (
    TERMS, CLUSTER_COLOR, CLUSTER_TINT,
    BONFERRONI_8, ALPHA_PER_TEST,
    CONTRAST_ORDER, CONTRAST_CODE, CONTRAST_GROUPS,
    resolve_groups, fit_per_protein_t,
    fetch_term_members_via_indra, hgnc_ids_to_uniprots,
    uniprot_to_hgnc_symbol,
)
from architecture import build_pipeline_fig

OUT_HTML = ROOT / "output" / "viz" / "cluster_explorer.html"


# ----------------------------------------------------------------------
# Section 1 — Bonferroni-8 matrix (clean layout, more room to breathe)
# ----------------------------------------------------------------------

def build_matrix_fig() -> go.Figure:
    n_terms = len(TERMS)
    nes = np.zeros((n_terms, len(CONTRAST_ORDER)))
    raw_p = np.zeros_like(nes)
    for i, (_, short, _, _) in enumerate(TERMS):
        for j, c in enumerate(CONTRAST_ORDER):
            n, p = BONFERRONI_8[c][short]
            nes[i, j] = n
            raw_p[i, j] = p
    passing = (raw_p < ALPHA_PER_TEST) & (nes > 0)

    short_labels = [t[1] for t in TERMS]
    clusters = [t[0] for t in TERMS]

    # Prepend cluster name as a discreet prefix so we don't need an
    # overlapping band-label annotation.
    y_labels = [
        f"<span style='color:{CLUSTER_COLOR[c]};font-weight:700'>{c[:4].upper()}</span>"
        f"  <span style='color:#222'>{s}</span>"
        for c, s in zip(clusters, short_labels)
    ]

    color_values = -np.log10(raw_p)

    pass_counts = passing.sum(axis=0)
    contrast_labels = [
        f"<b style='font-size:14px'>{c}</b><br>"
        f"<span style='font-size:11px;color:#555'>{pass_counts[j]}/8 pass</span>"
        for j, c in enumerate(CONTRAST_ORDER)
    ]

    annotations: list[dict] = []
    for i in range(n_terms):
        for j in range(len(CONTRAST_ORDER)):
            text = f"<b>{nes[i, j]:.2f}</b>"
            color = "white" if color_values[i, j] > 1.5 else "#333"
            annotations.append(dict(
                x=j, y=i,
                xref="x", yref="y",
                text=text, showarrow=False,
                font=dict(color=color, size=14, family="Arial, sans-serif"),
            ))

    shapes: list[dict] = []
    for i in range(n_terms):
        for j in range(len(CONTRAST_ORDER)):
            if passing[i, j]:
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=j - 0.49, x1=j + 0.49,
                    y0=i - 0.49, y1=i + 0.49,
                    line=dict(color="#000", width=3),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                ))

    hover_text = [[None] * len(CONTRAST_ORDER) for _ in range(n_terms)]
    for i, t in enumerate(short_labels):
        for j, c in enumerate(CONTRAST_ORDER):
            pass_str = "PASS (Bonferroni-8)" if passing[i, j] else "fail"
            hover_text[i][j] = (
                f"<b>{t}</b> ({clusters[i]})<br>{c}<br>"
                f"NES: {nes[i, j]:.3f}<br>"
                f"raw p: {raw_p[i, j]:.4f}<br>"
                f"family-wise threshold: {ALPHA_PER_TEST:.5f}<br>"
                f"<b>{pass_str}</b>"
            )

    fig = go.Figure(data=go.Heatmap(
        z=color_values,
        x=list(range(len(CONTRAST_ORDER))),
        y=list(range(n_terms)),
        colorscale=[
            [0.0,  "#f6f7f9"],
            [0.15, "#dceaf5"],
            [0.30, "#9ec6e3"],
            [0.50, "#4793c4"],
            [0.75, "#1f5da0"],
            [1.0,  "#0a3b80"],
        ],
        zmin=0, zmax=3,
        colorbar=dict(
            title=dict(text="-log₁₀(p)", side="right"),
            tickvals=[0, 0.5, 1, np.log10(1 / ALPHA_PER_TEST), 2, 3],
            ticktext=["1", "0.32", "0.10",
                      f"<b>{ALPHA_PER_TEST:.5f}</b><br>(threshold)",
                      "0.01", "≤ 0.001"],
            len=0.65,
            thickness=14,
            xpad=10,
        ),
        text=hover_text,
        hoverinfo="text",
        xgap=4, ygap=4,
    ))

    fig.update_layout(
        title=dict(
            text=(
                "<b>Pre-registered cluster pathway terms × group comparisons</b><br>"
                "<span style='font-size:12px;color:#777'>"
                "Cell number = enrichment score (NES).  Color = -log₁₀(raw p), saturated at the 1/1000 permutation floor.  "
                "Thick black border = passes the family-wise threshold (raw p < 0.00625 AND NES > 0).</span>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=17),
        ),
        xaxis=dict(
            side="top",
            tickmode="array",
            tickvals=list(range(len(CONTRAST_ORDER))),
            ticktext=contrast_labels,
            tickfont=dict(size=12),
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickmode="array",
            tickvals=list(range(n_terms)),
            ticktext=y_labels,
            tickfont=dict(size=13),
            showgrid=False, zeroline=False,
            automargin=True,
        ),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=320, r=180, t=130, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1200, height=560,
    )
    return fig


# ----------------------------------------------------------------------
# Section 2 — Where the cluster terms rank in the universe of tested
#             pathways (one panel per contrast).
# ----------------------------------------------------------------------

def load_universe_gsea(contrast_code: str) -> pd.DataFrame:
    """Load + concatenate the four pathway-database GSEA outputs for
    the given contrast, robust scope.  These are wave_24i (with-
    intermediates) discovery-era results — used for the term ranking
    universe view only.  The wave_24l confirmatory headline is on the
    matrix above."""
    # The local GSEA outputs split by contrast as separate dirs.
    contrast_dir_map = {
        "c9spor": "landscape_gsea",                  # default / original
        "c9ctrl": "landscape_gsea_c9_vs_control",
        "spctrl": "landscape_gsea_sporadic_vs_control",
    }
    base = ROOT / "output" / contrast_dir_map[contrast_code]
    parts: list[pd.DataFrame] = []
    for db in ("go", "reactome", "wikipathways", "phenotype"):
        path = base / f"robust_{db}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["db"] = db
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_universe_fig() -> go.Figure:
    """3x3 grid: cluster rows × group-comparison columns.

    Each cell shows that cluster's 2–3 members at their NES rank
    in that comparison, on a shared log x-axis.  Color reinforces the
    row (cluster).  No inline labels — within-cell hover names the
    member.

    Two dimensions readable:
    - Grouping (cluster)            → row + color
    - Empirical (rank-within-group) → horizontal position
    """
    cluster_term_ids = [t[3] for t in TERMS]
    cluster_of = {t[3]: t[0] for t in TERMS}
    short_of = {t[3]: t[1] for t in TERMS}
    CLUSTERS = ["Splicing", "Chromatin", "Transport"]
    cluster_count = {c: sum(1 for t in TERMS if t[0] == c) for c in CLUSTERS}

    # Per-contrast rank/NES lookup
    contrast_data: dict[str, tuple[dict[str, int], dict[str, float], int]] = {}
    for contrast in CONTRAST_ORDER:
        code = CONTRAST_CODE[contrast]
        u = load_universe_gsea(code)
        if u.empty:
            continue
        u = u.sort_values("NES", ascending=False).reset_index(drop=True)
        u["rank"] = np.arange(1, len(u) + 1)
        ranks: dict[str, int] = {}
        nes_vals: dict[str, float] = {}
        for tid in cluster_term_ids:
            m = u[u["Term"] == tid]
            if len(m) > 0:
                ranks[tid] = int(m["rank"].iloc[0])
                nes_vals[tid] = float(m["NES"].iloc[0])
        contrast_data[contrast] = (ranks, nes_vals, len(u))

    n_max = max((d[2] for d in contrast_data.values()), default=10000)

    fig = make_subplots(
        rows=len(CLUSTERS), cols=len(CONTRAST_ORDER),
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.035,
        vertical_spacing=0.08,
        column_titles=[
            f"<b style='font-size:13px;color:#222'>{c}</b>"
            for c in CONTRAST_ORDER
        ],
    )

    for row_idx, cluster_name in enumerate(CLUSTERS, start=1):
        for col_idx, contrast in enumerate(CONTRAST_ORDER, start=1):
            if contrast not in contrast_data:
                continue
            ranks, nes_vals, n_total = contrast_data[contrast]

            # Reference lines — per-cell top 1% / 5%
            for ref_pct, ref_color in ((0.05, "#e2e2e2"), (0.01, "#b8b8b8")):
                fig.add_vline(
                    x=n_total * ref_pct,
                    line_dash="dash", line_color=ref_color, line_width=1,
                    row=row_idx, col=col_idx,
                )

            # This cluster's members in this contrast
            xs: list[float] = []
            ys: list[float] = []
            hover: list[str] = []
            rng = np.random.default_rng(
                hash((cluster_name, contrast)) % (2**31 - 1)
            )
            for tid in cluster_term_ids:
                if cluster_of[tid] != cluster_name or tid not in ranks:
                    continue
                xs.append(float(ranks[tid]))
                ys.append(0.5 + float(rng.uniform(-0.20, 0.20)))
                hover.append(
                    f"<b>{short_of[tid]}</b><br>"
                    f"{contrast}<br>"
                    f"rank: {ranks[tid]:,} of {n_total:,}<br>"
                    f"NES: {nes_vals[tid]:.3f}"
                )

            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(
                        size=15,
                        color=CLUSTER_COLOR[cluster_name],
                        line=dict(width=1.2, color="#222"),
                        symbol="circle",
                        opacity=0.92,
                    ),
                    hovertext=hover, hoverinfo="text",
                    showlegend=False,
                ), row=row_idx, col=col_idx)

    # All subplots share the same log x-range; only bottom row shows ticks
    for row_idx in range(1, len(CLUSTERS) + 1):
        for col_idx in range(1, len(CONTRAST_ORDER) + 1):
            fig.update_xaxes(
                type="log",
                range=[np.log10(0.8), np.log10(n_max * 1.10)],
                showgrid=True, gridcolor="#f3f3f3",
                zeroline=False,
                row=row_idx, col=col_idx,
            )
            fig.update_yaxes(
                visible=False, range=[0, 1], fixedrange=True,
                row=row_idx, col=col_idx,
            )
    fig.update_xaxes(
        title=dict(
            text="rank in the universe of tested pathway terms "
                 "(log · 1 = top of NES ranking)",
            font=dict(size=11),
        ),
        row=len(CLUSTERS), col=2,
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>The 8 cluster terms by biological group × group comparison</b><br>"
                "<span style='font-size:12px;color:#777'>"
                "Rows = biological clusters; columns = group comparisons.  "
                "Each mark is one cluster member at its NES rank.  "
                "Within-cell spread = how tightly the cluster's members rank "
                "together; cross-column shift = triangulation."
                "</span>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=16),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=200, r=50, t=130, b=70),
        width=1400, height=460,
        showlegend=False,
    )

    # Row labels — single-line cluster name only, centered vertically
    for row_idx, cluster_name in enumerate(CLUSTERS, start=1):
        yaxis_idx = (row_idx - 1) * len(CONTRAST_ORDER) + 1
        yaxis_attr = "yaxis" if yaxis_idx == 1 else f"yaxis{yaxis_idx}"
        try:
            domain = getattr(fig.layout, yaxis_attr).domain
            y_center = (domain[0] + domain[1]) / 2
        except Exception:
            y_center = 1.0 - (row_idx - 0.5) / len(CLUSTERS)

        fig.add_annotation(
            xref="paper", yref="paper",
            x=-0.005, y=y_center,
            text=(
                f"<b style='font-size:15px;"
                f"color:{CLUSTER_COLOR[cluster_name]};"
                f"letter-spacing:0.4px'>{cluster_name}</b><br>"
                f"<span style='font-size:9.5px;color:#999'>"
                f"{cluster_count[cluster_name]} terms</span>"
            ),
            showarrow=False, xanchor="right", yanchor="middle",
            align="right",
        )

    # Per-cell summary count in top-right corner — "N/M ≤ 1%"
    for row_idx, cluster_name in enumerate(CLUSTERS, start=1):
        for col_idx, contrast in enumerate(CONTRAST_ORDER, start=1):
            if contrast not in contrast_data:
                continue
            ranks, _, n_total = contrast_data[contrast]
            in_top1 = 0
            cluster_n = 0
            for tid in cluster_term_ids:
                if cluster_of[tid] != cluster_name or tid not in ranks:
                    continue
                cluster_n += 1
                if ranks[tid] <= n_total * 0.01:
                    in_top1 += 1
            if cluster_n == 0:
                continue
            # Bold the count when all members pass; mute otherwise
            all_pass = (in_top1 == cluster_n)
            count_color = CLUSTER_COLOR[cluster_name] if all_pass else "#999"
            count_weight = "700" if all_pass else "500"
            fig.add_annotation(
                x=n_max * 1.0, y=0.92,
                xanchor="right", yanchor="top",
                text=(
                    f"<span style='font-family:ui-monospace,Menlo,monospace;"
                    f"font-size:11px;color:{count_color};"
                    f"font-weight:{count_weight}'>"
                    f"{in_top1}/{cluster_n}</span>"
                    f"<span style='font-size:8.5px;color:#aaa'>"
                    f" ≤ 1%</span>"
                ),
                showarrow=False,
                row=row_idx, col=col_idx,
            )

    return fig


# ----------------------------------------------------------------------
# Section 3 — Inside each cluster: member proteins by |t|.
# ----------------------------------------------------------------------

def build_anatomy_figs(
    t_stats: dict[str, pd.Series],
    cluster_members: dict[str, dict[str, set[str]]],
    sym_lookup: dict[str, str],
    top_n: int = 30,
) -> dict[str, go.Figure]:
    """Per-cluster horizontal bar chart of member proteins ranked by
    mean(|t|) in C9 comparisons.  Bars colored by which cluster term
    the protein is in (membership tag)."""
    figs: dict[str, go.Figure] = {}

    for cluster_name in ("Splicing", "Chromatin", "Transport"):
        # Union of UniProts across cluster's terms
        cluster_term_ids = [t[3] for t in TERMS if t[0] == cluster_name]
        all_members: set[str] = set()
        member_to_terms: dict[str, list[str]] = {}
        for tid in cluster_term_ids:
            members = cluster_members.get(tid, set())
            for u in members:
                all_members.add(u)
                short = next((t[1] for t in TERMS if t[3] == tid), tid)
                member_to_terms.setdefault(u, []).append(short)

        # Restrict to measured proteins (intersect with our t-stat index)
        measured_set = set(t_stats["C9 vs Sporadic"].index)
        measured_members = sorted(all_members & measured_set)
        if not measured_members:
            continue

        # Build a frame: protein × {t_c9spor, t_c9ctrl, t_spctrl, abs_mean_c9}
        df = pd.DataFrame(index=measured_members)
        for c in CONTRAST_ORDER:
            df[c] = t_stats[c].reindex(measured_members)
        df["abs_c9spor"] = df["C9 vs Sporadic"].abs()
        df["abs_c9ctrl"] = df["C9 vs Healthy"].abs()
        df["abs_spctrl"] = df["Sporadic vs Healthy"].abs()
        df["abs_mean_c9"] = (df["abs_c9spor"] + df["abs_c9ctrl"]) / 2.0
        df["symbol"] = [sym_lookup.get(u, u) for u in measured_members]
        df["membership"] = [
            ", ".join(member_to_terms.get(u, ["?"])) for u in measured_members
        ]
        df = df.sort_values("abs_mean_c9", ascending=False).head(top_n)

        # Bar chart: one row per protein, x = signed t per contrast
        y_labels = [
            f"<b>{s}</b>" for s in df["symbol"]
        ]
        # Reverse for top-of-axis appearance
        y_labels = y_labels[::-1]
        df_rev = df.iloc[::-1]

        fig = go.Figure()
        contrast_color = {
            "C9 vs Sporadic": CLUSTER_COLOR[cluster_name],
            "C9 vs Healthy":  "#777",
            "Sporadic vs Healthy": "#bbb",
        }
        for c in CONTRAST_ORDER:
            fig.add_trace(go.Bar(
                x=df_rev[c],
                y=y_labels,
                orientation="h",
                name=c,
                marker=dict(
                    color=contrast_color[c],
                    line=dict(width=0.5, color="#222"),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{c}<br>"
                    "t-stat: %{x:.2f}<br>"
                    "in terms: %{customdata[1]}<extra></extra>"
                ),
                customdata=df_rev[["symbol", "membership"]].values,
            ))

        n_measured = len(measured_members)
        n_total_cluster = len(all_members)
        height = max(420, 22 * len(df_rev) + 220)

        fig.update_layout(
            title=dict(
                text=(
                    f"<b style='color:{CLUSTER_COLOR[cluster_name]}'>{cluster_name}</b>"
                    f" — top {len(df_rev)} of {n_measured} measured cluster members (out of {n_total_cluster} total)<br>"
                    "<span style='font-size:11px;color:#777'>"
                    "Per-protein t-statistic across the three comparisons.  Sorted by mean |t| across the two C9 comparisons.  "
                    "Sign indicates direction (positive = higher in case group)."
                    "</span>"
                ),
                x=0.5, xanchor="center", font=dict(size=15),
            ),
            barmode="group",
            xaxis=dict(
                title=dict(text="t-statistic (signed)", font=dict(size=11)),
                zeroline=True, zerolinecolor="#444", zerolinewidth=1,
            ),
            yaxis=dict(
                tickfont=dict(size=11),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
                font=dict(size=10),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=110, r=20, t=130, b=50),
            width=1100, height=height,
        )
        figs[cluster_name] = fig
    return figs


# ----------------------------------------------------------------------
# Stitch into one HTML page
# ----------------------------------------------------------------------

HEADER_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>C9-ALS cluster claim — explorer</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         color: #1a1a1a; background: #fafafa; margin: 0; padding: 0; }
  .wrap { max-width: 1450px; margin: 0 auto; padding: 40px 20px 80px 20px; }
  h1 { font-size: 26px; font-weight: 700; margin: 0 0 6px 0; }
  .sub { color: #777; font-size: 13px; margin-bottom: 36px; }
  .section { margin-top: 56px; }
  .section h2 { font-size: 19px; font-weight: 700; border-bottom: 1px solid #ddd;
                padding-bottom: 8px; margin-bottom: 12px; }
  .section .lede { color: #555; font-size: 13px; margin-bottom: 24px; max-width: 880px;
                   line-height: 1.55; }
  .card { background: white; border: 1px solid #e6e6e6; border-radius: 6px;
          padding: 18px; margin-bottom: 28px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
</style>
</head>
<body>
<div class="wrap">
  <h1>C9-ALS regulatory-neighborhood cluster claim — explorer</h1>
  <div class="sub">AnswerALS PBMC proteomics, 3,264 proteins × 436 samples.  Three pairwise comparisons.  Eight pre-registered cluster pathway terms.</div>
"""

SECTION_TEMPLATES = {
    "architecture": {
        "title": "0. Architecture",
        "lede": (
            "The analysis as a five-station pipeline.  Each glyph is the "
            "shape of the data that station emits — a matrix collapses to a "
            "vector, joins a regulatory graph, projects onto a per-anchor "
            "concentration statistic, and lands as a ranked list of pathway "
            "outcomes."
        ),
    },
    "matrix": {
        "title": "1. The confirmation in one image",
        "lede": (
            "Eight pre-registered pathway terms tested under the wave_24l "
            "measured-only-paths regulatory-network pipeline.  The triangulation "
            "comes through cleanly: the C9 columns are saturated with bordered "
            "(passing) cells; the sporadic-vs-healthy column has none.  Six terms "
            "pass the family-wise threshold in <em>both</em> C9 comparisons — that "
            "set is the graph-invariant core of the cluster claim."
        ),
    },
    "universe": {
        "title": "2. Where the cluster terms land in the universe of tested pathways",
        "lede": (
            "Among all pathway terms tested by the discovery pipeline, where do the "
            "eight cluster terms rank?  In both C9 comparisons, cluster terms "
            "concentrate near the top of the NES ranking (positive enrichment, "
            "rank near 1); in sporadic-vs-healthy they scatter through the middle.  "
            "This is the empirical version of triangulation: the cluster signal isn't "
            "just present in C9 — it dominates the top of the ranking, in a way it "
            "doesn't in sporadic."
        ),
    },
    "anatomy": {
        "title": "3. Inside each cluster: which proteins drive the signal",
        "lede": (
            "Each biological cluster has its constituent pathway terms and, in turn, "
            "their gene-set members.  For the members that are measured in our cohort, "
            "we show the per-protein t-statistic across all three comparisons.  Sort is "
            "by mean magnitude in the two C9 comparisons (most differential at top).  "
            "The pattern across rows tells you whether the cluster's signal is dominated "
            "by a few strongly-perturbed proteins or spread broadly across members."
        ),
    },
}

FOOTER_HTML = """
</div>
</body>
</html>
"""


def main() -> None:
    print("Building Section 0 — pipeline diagram…")
    fig_pipeline = build_pipeline_fig()

    print("Building Section 1 — Bonferroni-8 matrix…")
    fig_matrix = build_matrix_fig()

    print("Building Section 2 — universe ranking…")
    fig_universe = build_universe_fig()

    print("Loading proteomics + computing per-protein t-statistics…")
    data = pd.read_csv(ROOT / "output/proteomics/all_als.data.csv", index_col=0)
    md = pd.read_csv(ROOT / "output/proteomics/all_als.metadata.csv", index_col=0)
    groups = resolve_groups(md)
    t_stats: dict[str, pd.Series] = {}
    for c in CONTRAST_ORDER:
        contrast = CONTRAST_GROUPS[c]
        print(f"  fit {c}")
        t_stats[c] = fit_per_protein_t(data, md, groups, contrast)

    print("Fetching cluster term members from INDRA…")
    term_ids = [t[3] for t in TERMS]
    hgnc_members = fetch_term_members_via_indra(term_ids)
    cluster_members: dict[str, set[str]] = {}
    for tid, hgncs in hgnc_members.items():
        cluster_members[tid] = hgnc_ids_to_uniprots(hgncs)

    # Symbol lookup for measured proteins only
    measured_proteins = list(data.index)
    sym_lookup = uniprot_to_hgnc_symbol(measured_proteins)

    print("Building Section 3 — per-cluster anatomy…")
    figs_anatomy = build_anatomy_figs(t_stats, cluster_members, sym_lookup, top_n=30)

    # Render HTML
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    parts = [HEADER_HTML]

    # Section 0 — Architecture (pipeline only)
    parts.append('<div class="section">')
    parts.append(f'<h2>{SECTION_TEMPLATES["architecture"]["title"]}</h2>')
    parts.append(f'<div class="lede">{SECTION_TEMPLATES["architecture"]["lede"]}</div>')
    parts.append('<div class="card">')
    parts.append(fig_pipeline.to_html(
        include_plotlyjs="cdn", full_html=False,
        config=dict(displaylogo=False),
    ))
    parts.append('</div></div>')

    # Section 1
    parts.append('<div class="section">')
    parts.append(f'<h2>{SECTION_TEMPLATES["matrix"]["title"]}</h2>')
    parts.append(f'<div class="lede">{SECTION_TEMPLATES["matrix"]["lede"]}</div>')
    parts.append('<div class="card">')
    parts.append(fig_matrix.to_html(
        include_plotlyjs=False, full_html=False,
        config=dict(displaylogo=False),
    ))
    parts.append('</div></div>')

    # Section 2
    parts.append('<div class="section">')
    parts.append(f'<h2>{SECTION_TEMPLATES["universe"]["title"]}</h2>')
    parts.append(f'<div class="lede">{SECTION_TEMPLATES["universe"]["lede"]}</div>')
    parts.append('<div class="card">')
    parts.append(fig_universe.to_html(
        include_plotlyjs=False, full_html=False,
        config=dict(displaylogo=False),
    ))
    parts.append('</div></div>')

    # Section 3 — one card per cluster
    parts.append('<div class="section">')
    parts.append(f'<h2>{SECTION_TEMPLATES["anatomy"]["title"]}</h2>')
    parts.append(f'<div class="lede">{SECTION_TEMPLATES["anatomy"]["lede"]}</div>')
    for cluster in ("Splicing", "Chromatin", "Transport"):
        fig = figs_anatomy.get(cluster)
        if fig is None:
            continue
        parts.append('<div class="card">')
        parts.append(fig.to_html(
            include_plotlyjs=False, full_html=False,
            config=dict(displaylogo=False),
        ))
        parts.append('</div>')
    parts.append('</div>')

    parts.append(FOOTER_HTML)

    OUT_HTML.write_text("".join(parts))
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
