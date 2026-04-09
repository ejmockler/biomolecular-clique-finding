"""
Validation result visualizations.

This module provides visualizations for the validation pipeline output,
telling an **epistemic confidence story**: "why should you believe this?"

The narrative structure is a legal argument:
- Act I:  Verdict (Figure 01) — "did it validate?"
- Act II: Statistical backbone (Figures 02-06) — evidence for/against
- Act III: Mechanistic corroboration (Figures 07-10) — network biology support

The recurring visual metaphor is "observed vs null" — a vertical line against
a histogram answering "how extreme is our signal?"

Visualization Hierarchy:
========================

Act I - The Verdict (1 figure):
    01. Verdict Scorecard — phase pass/fail summary + verdict badge

Act II - The Statistical Backbone (5 figures):
    02. Covariate-Adjusted Enrichment — density separation + forest point
    03. Label Permutation Null — stratified/free null histograms
    04. Specificity Contrasts — forest plot of subtype contrasts
    05. Negative Control Distribution — random gene set null
    06. Graph Permutation Distribution — topology null

Act III - Mechanistic Corroboration (4 figures):
    07. Proximity Decay — DE signal vs graph distance
    08. Reverse Causal Ranking — upstream regulator lollipop
    09. Differential Landscape — volcano plot with targets highlighted
    10. Evidence Summary Table — all phases in one table

Design Philosophy:
==================
- Null distributions: cool grays (recede perceptually)
- Observed statistic: CASE blue (#2563eb, advances)
- Pass/fail: green/red (universal)
- Null-vs-observed contrast is pre-attentive
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats

from cliquefinder.viz.core import Figure, FigureCollection
from cliquefinder.viz.styles import (
    Palette,
    PALETTES,
    configure_style,
    format_pvalue,
    italicize_gene,
)


# =============================================================================
# Color Semantics for Validation
# =============================================================================

VALIDATION_COLORS = {
    # Verdict states
    "validated": "#059669",  # Emerald-600
    "inconclusive": "#d97706",  # Amber-600
    "refuted": "#dc2626",  # Red-600
    # Phase status
    "pass": "#059669",  # Emerald-600
    "fail": "#dc2626",  # Red-600
    "skipped": "#9ca3af",  # Gray-400
    # Null distribution — the backbone visual
    "null_fill": "#e2e8f0",  # Slate-200 (muted, recedes)
    "null_edge": "#94a3b8",  # Slate-400
    "observed": "#2563eb",  # Blue-600 (CASE color — signal under test)
    "tail_shade": "#dbeafe",  # Blue-100
    "quantile_line": "#cbd5e1",  # Slate-300
    # Supplementary markers
    "gate_badge": "#1e40af",  # Blue-800
    "supplementary": "#6366f1",  # Indigo-500
}


# =============================================================================
# Phase metadata for scorecard rendering
# =============================================================================

_PHASE_META = [
    ("covariate_adjusted", "Covariate-Adjusted Enrichment", True),
    ("label_permutation", "Label Permutation Null", True),
    ("specificity", "Specificity Contrasts", False),
    ("matched_reanalysis", "Matched Reanalysis", False),
    ("negative_controls", "Negative Controls", False),
    ("graph_permutation", "Graph Permutation", False),
    ("network_proximity", "Network Proximity", False),
]


# =============================================================================
# Data Loading
# =============================================================================


def load_validation_data(validation_dir: Path) -> dict[str, Any]:
    """
    Load all available validation output files from a directory.

    Supports both unprefixed (new) and ``phase{n}_*`` (old) naming conventions.
    Missing files are returned as ``None`` for graceful degradation.

    Parameters
    ----------
    validation_dir : Path
        Directory containing validation pipeline outputs.

    Returns
    -------
    dict[str, Any]
        Keys are data identifiers; values are parsed JSON dicts,
        DataFrames, or None if the file is absent.
    """
    d = Path(validation_dir)

    def _load_json(*candidates: str) -> dict | None:
        for name in candidates:
            p = d / name
            if p.exists():
                with open(p) as f:
                    return json.load(f)
        return None

    def _load_csv(name: str) -> pd.DataFrame | None:
        p = d / name
        if p.exists():
            return pd.read_csv(p)
        return None

    return {
        "report": _load_json("validation_report.json"),
        "phase1": _load_json("covariate_enrichment.json", "phase1_covariate_enrichment.json"),
        "phase2": _load_json("specificity.json", "phase2_specificity.json"),
        "phase3": _load_json("label_permutation.json", "phase3_label_permutation.json"),
        "phase4": _load_json("matched_enrichment.json", "phase4_matched_enrichment.json"),
        "phase5a": _load_json("negative_controls.json", "phase5_negative_controls.json"),
        "phase5b": _load_json("graph_permutation.json", "phase5_graph_permutation.json"),
        "phase6": _load_json("network_proximity.json"),
        "discovery": _load_json("discovery_results.json"),
        "label_perm_dist": _load_csv("label_permutation_distributions.csv"),
        "neg_ctrl_dist": _load_csv("negative_control_distributions.csv"),
        "graph_perm_dist": _load_csv("graph_permutation_distributions.csv"),
        "protein_df": _load_csv("protein_differential_results.csv"),
        "decay_curve": _load_csv("proximity_decay_curve.csv"),
        "regulators": _load_csv("reverse_causal_top_regulators.csv"),
    }


# =============================================================================
# Visualizer
# =============================================================================


class ValidationVisualizer:
    """
    Publication-quality visualizations for validation pipeline results.

    Follows the same constructor pattern as ``DifferentialCliqueVisualizer``:
    resolves palette, configures style, sets font-size dict.

    Parameters
    ----------
    palette : str or Palette
        Color palette name or instance.
    style : {"paper", "presentation", "notebook"}
        Target medium.
    """

    def __init__(
        self,
        palette: str | Palette = "default",
        style: Literal["paper", "presentation", "notebook"] = "paper",
    ):
        if isinstance(palette, str):
            self.palette = PALETTES.get(palette, PALETTES["default"])
        else:
            self.palette = palette
        self.style = style
        configure_style(style=style, palette=self.palette)

        self.font_sizes = {
            "paper": {"title": 14, "label": 11, "tick": 9, "annotation": 8},
            "presentation": {"title": 18, "label": 14, "tick": 12, "annotation": 10},
            "notebook": {"title": 12, "label": 10, "tick": 8, "annotation": 7},
        }[style]

    # =========================================================================
    # Private: Null distribution template
    # =========================================================================

    def _plot_null_distribution(
        self,
        ax: plt.Axes,
        null_values: np.ndarray,
        observed: float,
        *,
        x_label: str,
        p_value: float | None = None,
        tail: Literal["right", "left"] = "right",
        label_null: str = "Null",
        label_observed: str = "Observed",
    ) -> None:
        """
        Shared template for observed-vs-null histogram figures.

        Visual stack (z-order):
        1. Histogram (density, muted fill)
        2. KDE overlay
        3. 5th/95th percentile dashed lines
        4. Tail shading beyond observed
        5. Observed vertical line (bold blue)
        6. P-value annotation box
        """
        null_values = np.asarray(null_values, dtype=float)
        null_values = null_values[np.isfinite(null_values)]
        if len(null_values) == 0:
            ax.text(0.5, 0.5, "No valid null values", ha="center", va="center",
                    transform=ax.transAxes, fontsize=self.font_sizes["label"])
            return

        n_bins = min(40, max(10, int(np.sqrt(len(null_values)))))

        # 1. Histogram
        ax.hist(
            null_values,
            bins=n_bins,
            density=True,
            color=VALIDATION_COLORS["null_fill"],
            edgecolor=VALIDATION_COLORS["null_edge"],
            linewidth=0.5,
            label=label_null,
            zorder=2,
        )

        # 2. KDE overlay
        if len(null_values) > 5:
            try:
                kde = sp_stats.gaussian_kde(null_values)
                x_grid = np.linspace(null_values.min(), null_values.max(), 200)
                ax.plot(
                    x_grid, kde(x_grid),
                    color=VALIDATION_COLORS["null_edge"],
                    alpha=0.6,
                    linewidth=1.0,
                    zorder=3,
                )
            except np.linalg.LinAlgError:
                pass  # Degenerate distribution

        # 3. Percentile lines
        q05, q95 = np.percentile(null_values, [5, 95])
        for q_val, q_label in [(q05, "5th"), (q95, "95th")]:
            ax.axvline(
                q_val,
                color=VALIDATION_COLORS["quantile_line"],
                linestyle="--",
                linewidth=0.8,
                zorder=4,
            )
            ax.text(
                q_val,
                ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1.0,
                q_label,
                ha="center",
                va="top",
                fontsize=self.font_sizes["annotation"] - 1,
                color=VALIDATION_COLORS["null_edge"],
            )

        # 4. Tail shading
        if tail == "right":
            ax.axvspan(
                observed,
                ax.get_xlim()[1],
                color=VALIDATION_COLORS["tail_shade"],
                alpha=0.3,
                zorder=1,
            )
        else:
            ax.axvspan(
                ax.get_xlim()[0],
                observed,
                color=VALIDATION_COLORS["tail_shade"],
                alpha=0.3,
                zorder=1,
            )

        # 5. Observed line
        ax.axvline(
            observed,
            color=VALIDATION_COLORS["observed"],
            linewidth=2.5,
            zorder=5,
            label=label_observed,
        )

        # 6. P-value annotation
        if p_value is not None:
            p_text = format_pvalue(p_value)
            ax.annotate(
                p_text,
                xy=(observed, ax.get_ylim()[1] * 0.85),
                fontsize=self.font_sizes["annotation"],
                fontweight="bold",
                color=VALIDATION_COLORS["observed"],
                ha="left" if tail == "right" else "right",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=VALIDATION_COLORS["observed"],
                    linewidth=1.0,
                    alpha=0.9,
                ),
                zorder=6,
            )

        ax.set_xlabel(x_label, fontsize=self.font_sizes["label"])
        ax.set_ylabel("Density", fontsize=self.font_sizes["label"])
        ax.legend(fontsize=self.font_sizes["annotation"], loc="upper left")

    # =========================================================================
    # Figure 01: Verdict Scorecard
    # =========================================================================

    def plot_verdict_scorecard(
        self,
        report: dict,
        figsize: tuple[float, float] = (14, 7),
    ) -> Figure:
        """
        Scorecard answering "did it validate?" in <2 seconds.

        Left zone (60%): phase pass/fail rows with GATE badges.
        Right zone (40%): large verdict badge with summary text.
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, figure=fig, width_ratios=[3, 2], wspace=0.05)

        phases = report.get("phases", {})
        phase_details = report.get("phase_details", {})
        verdict = report.get("verdict", "inconclusive")
        summary = report.get("summary", "")

        # ---- Left: Phase rows ----
        ax_left = fig.add_subplot(gs[0])
        ax_left.set_xlim(0, 10)
        ax_left.set_ylim(-0.5, len(_PHASE_META) + 0.5)
        ax_left.axis("off")

        # Title
        ax_left.text(
            0.0, len(_PHASE_META) + 0.2, "Validation Phases",
            fontsize=self.font_sizes["title"],
            fontweight="bold", va="bottom",
        )

        # Supplementary count
        n_supp_pass = 0
        n_supp_total = 0

        for idx, (key, label, is_gate) in enumerate(_PHASE_META):
            y = len(_PHASE_META) - 1 - idx
            present = key in phases
            detail = phase_details.get(key, "")

            # Also check compound keys like label_permutation_stratified
            if not detail:
                for dk, dv in phase_details.items():
                    if dk.startswith(key):
                        detail = dv
                        break

            # Determine pass/fail
            passed = None
            if present and detail:
                detail_lower = detail.lower()
                if "(pass)" in detail_lower:
                    passed = True
                elif "(fail)" in detail_lower:
                    passed = False
                elif key == "label_permutation":
                    # Check the stratified p-value
                    lp = phases.get("label_permutation", {})
                    strat = lp.get("stratified", lp)
                    p = strat.get("permutation_pvalue", 1.0)
                    passed = p < 0.05
                elif key == "specificity":
                    passed = None  # Characterization, not pass/fail
                elif key == "network_proximity":
                    np_data = phases.get("network_proximity", {})
                    passed = np_data.get("any_significant", False)

            if not present:
                color = VALIDATION_COLORS["skipped"]
                status_char = "–"
            elif passed is True:
                color = VALIDATION_COLORS["pass"]
                status_char = "P"
            elif passed is False:
                color = VALIDATION_COLORS["fail"]
                status_char = "F"
            else:
                color = VALIDATION_COLORS["skipped"]
                status_char = "○"

            # Track supplementary pass/fail
            if not is_gate and present and passed is not None:
                n_supp_total += 1
                if passed:
                    n_supp_pass += 1

            # Status circle
            circle = mpatches.Circle(
                (0.4, y), 0.2,
                facecolor=color, edgecolor="white", linewidth=1.5,
                transform=ax_left.transData, zorder=3,
            )
            ax_left.add_patch(circle)
            ax_left.text(
                0.4, y, status_char,
                ha="center", va="center",
                fontsize=self.font_sizes["annotation"],
                fontweight="bold", color="white", zorder=4,
            )

            # Phase label
            ax_left.text(
                1.0, y + 0.1, label,
                fontsize=self.font_sizes["label"],
                fontweight="bold", va="center",
            )

            # Compact detail
            if detail:
                ax_left.text(
                    1.0, y - 0.2, detail,
                    fontsize=self.font_sizes["annotation"],
                    color="#6b7280", va="center",
                )

            # GATE badge
            if is_gate:
                badge = mpatches.FancyBboxPatch(
                    (8.0, y - 0.15), 1.2, 0.3,
                    boxstyle="round,pad=0.05",
                    facecolor=VALIDATION_COLORS["gate_badge"],
                    edgecolor="none",
                    zorder=3,
                )
                ax_left.add_patch(badge)
                ax_left.text(
                    8.6, y, "GATE",
                    ha="center", va="center",
                    fontsize=self.font_sizes["annotation"] - 1,
                    fontweight="bold", color="white", zorder=4,
                )

            # Separator after gates
            if idx == 1:
                ax_left.axhline(
                    y - 0.4,
                    xmin=0.0, xmax=0.95,
                    color="#e5e7eb", linewidth=1.5,
                )

        # ---- Right: Verdict badge ----
        ax_right = fig.add_subplot(gs[1])
        ax_right.set_xlim(0, 10)
        ax_right.set_ylim(0, 10)
        ax_right.axis("off")

        verdict_color = VALIDATION_COLORS.get(verdict, VALIDATION_COLORS["inconclusive"])

        # Background badge
        badge_bg = mpatches.FancyBboxPatch(
            (0.5, 3.5), 9.0, 5.0,
            boxstyle="round,pad=0.3",
            facecolor=verdict_color,
            alpha=0.10,
            edgecolor=verdict_color,
            linewidth=2,
        )
        ax_right.add_patch(badge_bg)

        # Verdict word
        ax_right.text(
            5.0, 7.0, verdict.upper(),
            ha="center", va="center",
            fontsize=28, fontweight="bold",
            color=verdict_color,
        )

        # Summary
        if summary:
            # Wrap long text
            wrapped = _wrap_text(summary, max_chars=40)
            ax_right.text(
                5.0, 5.0, wrapped,
                ha="center", va="center",
                fontsize=self.font_sizes["annotation"],
                color="#374151",
                linespacing=1.4,
            )

        # Supplementary count
        if n_supp_total > 0:
            ax_right.text(
                5.0, 3.8, f"Supplementary: {n_supp_pass}/{n_supp_total} pass",
                ha="center", va="center",
                fontsize=self.font_sizes["annotation"],
                color="#6b7280",
            )

        fig.suptitle(
            "Validation Scorecard",
            fontsize=self.font_sizes["title"] + 2,
            fontweight="bold",
            y=0.98,
        )

        return Figure(
            fig=fig,
            title="Validation Scorecard",
            description="Phase-by-phase pass/fail summary with overall verdict.",
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 02: Covariate-Adjusted Enrichment
    # =========================================================================

    def plot_covariate_enrichment(
        self,
        phase1: dict,
        protein_df: pd.DataFrame | None = None,
        figsize: tuple[float, float] = (8, 5),
    ) -> Figure:
        """
        Single-panel |t-stat| density for targets vs background with stats box.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        z = phase1.get("z_score", 0)
        p = phase1.get("empirical_pvalue", 1)
        n_targets = phase1.get("n_targets", 0)
        pct_down = phase1.get("pct_down", 0)
        vif = phase1.get("variance_inflation_factor", 1.0)
        mw_p = phase1.get("mannwhitney_pvalue")

        if protein_df is not None and "t_statistic" in protein_df.columns and "is_target" in protein_df.columns:
            targets = protein_df.loc[protein_df["is_target"] == True, "t_statistic"].dropna().abs()  # noqa: E712
            background = protein_df.loc[protein_df["is_target"] == False, "t_statistic"].dropna().abs()  # noqa: E712

            bins = np.linspace(0, max(targets.max(), background.quantile(0.99)) * 1.1, 50)

            ax.hist(
                background, bins=bins, density=True, alpha=0.4,
                color="#94a3b8", edgecolor="#94a3b8", linewidth=0.5,
                label=f"Background (n={len(background)})",
            )
            ax.hist(
                targets, bins=bins, density=True, alpha=0.6,
                color=VALIDATION_COLORS["observed"], edgecolor=VALIDATION_COLORS["observed"],
                linewidth=0.5, label=f"Targets (n={len(targets)})",
            )

            # Target mean line
            target_mean = targets.mean()
            ax.axvline(target_mean, color=VALIDATION_COLORS["observed"], linestyle="--", linewidth=1.5)
            ax.text(
                target_mean, ax.get_ylim()[1] * 0.9,
                f"target mean={target_mean:.2f}",
                fontsize=self.font_sizes["annotation"],
                color=VALIDATION_COLORS["observed"],
                ha="left",
            )

            if mw_p is not None:
                ax.text(
                    0.95, 0.95, f"Mann-Whitney {format_pvalue(mw_p)}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=self.font_sizes["annotation"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb"),
                )

            ax.set_xlabel("|t-statistic|", fontsize=self.font_sizes["label"])
            ax.set_ylabel("Density", fontsize=self.font_sizes["label"])
            ax.legend(fontsize=self.font_sizes["annotation"])
        else:
            ax.text(
                0.5, 0.5, "Protein data unavailable\n(density plot skipped)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=self.font_sizes["label"], color="#6b7280",
            )

        # Stats box — top-right, inside axes
        ann_parts = [
            f"z = {z:.2f}",
            format_pvalue(p),
            f"n = {n_targets}",
            f"{pct_down:.0f}% down",
            f"VIF = {vif:.1f}",
        ]
        ax.text(
            0.98, 0.75, "\n".join(ann_parts),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=self.font_sizes["annotation"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e5e7eb"),
        )

        ax.set_title("Covariate-Adjusted Enrichment", fontsize=self.font_sizes["title"])

        fig.tight_layout()
        return Figure(
            fig=fig,
            title="Covariate-Adjusted Enrichment",
            description="Target set enrichment after controlling for confounders.",
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 03: Label Permutation Null
    # =========================================================================

    def plot_label_permutation(
        self,
        phase3: dict,
        label_perm_dist_df: pd.DataFrame | None = None,
        figsize: tuple[float, float] = (12, 5),
    ) -> Figure:
        """
        Stratified and free permutation null distributions.

        Two panels, each showing the null histogram with the observed z-score.
        """
        strat_data = phase3.get("stratified", {})
        free_data = phase3.get("free", {})
        observed_z = strat_data.get("observed_z", free_data.get("observed_z", 0))

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        for ax, mode_data, mode_label in [
            (axes[0], strat_data, "Stratified"),
            (axes[1], free_data, "Free"),
        ]:
            p_val = mode_data.get("permutation_pvalue")
            frozen = mode_data.get("frozen_fraction", 0)

            # Get null distribution values from CSV
            null_vals = None
            if label_perm_dist_df is not None and "mode" in label_perm_dist_df.columns:
                mask = label_perm_dist_df["mode"] == mode_label.lower()
                if mask.any() and "competitive_z" in label_perm_dist_df.columns:
                    null_vals = label_perm_dist_df.loc[mask, "competitive_z"].values

            if null_vals is not None and len(null_vals) > 0:
                self._plot_null_distribution(
                    ax, null_vals, observed_z,
                    x_label="Competitive z-score",
                    p_value=p_val,
                    tail="right",
                    label_null=f"{mode_label} null",
                    label_observed="Observed z",
                )
            else:
                # Synthesize from summary stats
                null_mean = mode_data.get("null_mean", 0)
                null_std = mode_data.get("null_std", 1)
                n_perm = mode_data.get("n_permutations", 500)
                synth = np.random.default_rng(42).normal(null_mean, null_std, n_perm)
                self._plot_null_distribution(
                    ax, synth, observed_z,
                    x_label="Competitive z-score",
                    p_value=p_val,
                    tail="right",
                    label_null=f"{mode_label} null (synthetic)",
                    label_observed="Observed z",
                )

            ax.set_title(
                f"{mode_label} Permutation",
                fontsize=self.font_sizes["title"],
            )

            # Frozen fraction warning
            if frozen > 0.10:
                ax.text(
                    0.02, 0.02,
                    f"WARN: frozen={frozen:.0%}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=self.font_sizes["annotation"],
                    color="#d97706",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fffbeb", edgecolor="#d97706"),
                )

        fig.suptitle(
            "Label Permutation Null Test",
            fontsize=self.font_sizes["title"] + 2,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        return Figure(
            fig=fig,
            title="Label Permutation Null",
            description="Can random label shuffling reproduce this signal?",
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 04: Specificity Contrasts
    # =========================================================================

    def plot_specificity(
        self,
        phase2: dict,
        figsize: tuple[float, float] = (10, 5),
    ) -> Figure:
        """
        Forest plot of subtype contrasts with interaction test annotation.
        """
        contrasts = phase2.get("contrasts", {})
        interaction = phase2.get("interaction_test", {})
        spec_label = phase2.get("specificity_label", "unknown")

        fig, ax = plt.subplots(figsize=figsize)

        names = list(contrasts.keys())
        n = len(names)
        if n == 0:
            ax.text(0.5, 0.5, "No contrast data", ha="center", va="center",
                    transform=ax.transAxes)
            return Figure(fig=fig, title="Specificity Contrasts",
                          description="No data.", figure_type="matplotlib")

        y_positions = np.arange(n)

        for i, name in enumerate(names):
            c = contrasts[name]
            z = c.get("z_score", 0)
            p = c.get("empirical_pvalue", 1)
            sig = p < 0.05

            color = VALIDATION_COLORS["observed"] if sig else "#94a3b8"
            ax.plot(z, i, "o", color=color, markersize=8, zorder=3)

            # Horizontal line from 0 to z
            ax.hlines(i, 0, z, color=color, linewidth=1.5, zorder=2)

            # P-value annotation
            ax.text(
                z + 0.05 * np.sign(z) if z != 0 else 0.05, i + 0.15,
                f"z={z:.2f}, {format_pvalue(p)}",
                fontsize=self.font_sizes["annotation"],
                color=color, va="bottom",
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [n.replace("_", " ") for n in names],
            fontsize=self.font_sizes["label"],
        )
        ax.axvline(0, color="#e5e7eb", linewidth=1, zorder=1)
        ax.set_xlabel("z-score", fontsize=self.font_sizes["label"])

        # Title with specificity badge inline (avoids overlapping data rows)
        badge_color = VALIDATION_COLORS["observed"] if spec_label == "specific" else "#d97706"
        ax.set_title("Specificity Contrasts", fontsize=self.font_sizes["title"])

        # Specificity badge — placed in top-right margin, clear of data
        fig.text(
            0.98, 0.97,
            spec_label.upper(),
            ha="right", va="top",
            fontsize=self.font_sizes["label"],
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_color, edgecolor="none"),
        )

        # Interaction test annotation — placed below all axes content.
        # Reserve bottom space first, then place text in the margin.
        if interaction:
            z_diff = interaction.get("z_difference", 0)
            int_p = interaction.get("interaction_pvalue", 1)
            ci = interaction.get("z_difference_ci", [None, None])
            ci_str = ""
            if ci[0] is not None:
                ci_str = f", 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"
            fig.tight_layout(rect=[0, 0.08, 1, 1])
            fig.text(
                0.5, 0.02,
                f"Interaction: $\\Delta$z={z_diff:.2f}, {format_pvalue(int_p)}{ci_str}",
                ha="center", va="bottom",
                fontsize=self.font_sizes["annotation"],
                color="#374151",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb"),
            )
        else:
            fig.tight_layout()

        return Figure(
            fig=fig,
            title="Specificity Contrasts",
            description="Is the signal C9orf72-specific or shared across subtypes?",
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 05: Negative Control Distribution
    # =========================================================================

    def plot_negative_controls(
        self,
        phase5a: dict,
        neg_ctrl_dist_df: pd.DataFrame | None = None,
        figsize: tuple[float, float] = (10, 5),
    ) -> Figure:
        """
        Null histogram of competitive z-scores from random gene sets.
        """
        fig, ax = plt.subplots(figsize=figsize)

        competitive = phase5a.get("competitive_z", {})
        target_z = competitive.get("target_z")
        fpr = competitive.get("fpr")
        pctl = competitive.get("percentile")

        # Try competitive_z from distribution CSV
        null_vals = None
        if neg_ctrl_dist_df is not None and "competitive_z" in neg_ctrl_dist_df.columns:
            ctrl_mask = neg_ctrl_dist_df["type"] != "target"
            null_vals = neg_ctrl_dist_df.loc[ctrl_mask, "competitive_z"].dropna().values

        # Compute annotation p-value from percentile rank
        ann_p = pctl / 100.0 if pctl is not None else fpr

        if target_z is not None and null_vals is not None and len(null_vals) > 0:
            self._plot_null_distribution(
                ax, null_vals, target_z,
                x_label="Competitive z-score",
                p_value=ann_p,
                tail="right",
                label_null="Random gene sets",
                label_observed="Target set",
            )
        elif null_vals is None or len(null_vals) == 0:
            # Fallback: use roast_pvalue distribution
            if neg_ctrl_dist_df is not None and "roast_pvalue" in neg_ctrl_dist_df.columns:
                ctrl_mask = neg_ctrl_dist_df["type"] != "target"
                null_pvals = neg_ctrl_dist_df.loc[ctrl_mask, "roast_pvalue"].dropna().values
                target_pval = phase5a.get("target_pvalue", 0.05)

                self._plot_null_distribution(
                    ax, null_pvals, target_pval,
                    x_label="ROAST p-value",
                    p_value=ann_p,
                    tail="left",
                    label_null="Random gene sets",
                    label_observed="Target set",
                )
            else:
                ax.text(0.5, 0.5, "Distribution data unavailable",
                        ha="center", va="center", transform=ax.transAxes)

        # Percentile annotation
        ann_parts = []
        if pctl is not None:
            ann_parts.append(f"Percentile: {pctl:.1f}%")
        if fpr is not None:
            ann_parts.append(f"FPR: {fpr:.3f}")
        n_ctrl = phase5a.get("n_control_sets", 0)
        if n_ctrl:
            ann_parts.append(f"n controls: {n_ctrl}")
        if ann_parts:
            ax.text(
                0.98, 0.98, "\n".join(ann_parts),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=self.font_sizes["annotation"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb"),
            )

        ax.set_title("Negative Control Distribution", fontsize=self.font_sizes["title"])
        fig.tight_layout()

        return Figure(
            fig=fig,
            title="Negative Control Distribution",
            description="Is this signal exceptional vs random gene sets?",
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 06: Graph Permutation Distribution
    # =========================================================================

    def plot_graph_permutation(
        self,
        phase5b: dict,
        graph_perm_dist_df: pd.DataFrame | None = None,
        figsize: tuple[float, float] = (10, 5),
    ) -> Figure:
        """
        Null histogram of ROAST p-values from graph permutations.
        """
        fig, ax = plt.subplots(figsize=figsize)

        target_p = phase5b.get("target_pvalue")
        fpr = phase5b.get("fpr")
        pctl = phase5b.get("target_percentile")
        n_elig = phase5b.get("n_eligible_regulators")
        med_size = phase5b.get("median_control_set_size")
        graph_stats = phase5b.get("graph_stats", {})

        null_vals = None
        if graph_perm_dist_df is not None and "roast_pvalue" in graph_perm_dist_df.columns:
            ctrl_mask = graph_perm_dist_df["type"] != "target"
            null_vals = graph_perm_dist_df.loc[ctrl_mask, "roast_pvalue"].dropna().values

        if target_p is not None and null_vals is not None and len(null_vals) > 0:
            self._plot_null_distribution(
                ax, null_vals, target_p,
                x_label="ROAST p-value",
                p_value=target_p,
                tail="left",
                label_null="Graph permutations",
                label_observed="Target set",
            )
        else:
            ax.text(0.5, 0.5, "Distribution data unavailable",
                    ha="center", va="center", transform=ax.transAxes)

        # Annotation block
        ann_parts = []
        if pctl is not None:
            ann_parts.append(f"Percentile: {pctl:.1f}%")
        if fpr is not None:
            ann_parts.append(f"FPR: {fpr:.2f}")
        if n_elig is not None:
            ann_parts.append(f"Eligible regulators: {n_elig}")
        if med_size is not None:
            ann_parts.append(f"Median ctrl set size: {med_size}")
        if ann_parts:
            ax.text(
                0.98, 0.98, "\n".join(ann_parts),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=self.font_sizes["annotation"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb"),
            )

        # Graph stats as subtitle (outside plot area — avoids overlapping data)
        title_text = "Graph Permutation Distribution"
        if graph_stats:
            gs_text = (
                f"{graph_stats.get('n_nodes', '?')} nodes, "
                f"{graph_stats.get('n_edges', '?')} edges, "
                f"{graph_stats.get('n_regulators', '?')} regulators"
            )
            title_text += f"\n{gs_text}"
            ax.set_title(
                title_text, fontsize=self.font_sizes["title"],
                linespacing=1.6,
            )
            # Make the subtitle line smaller/gray via the title's secondary line
            # (matplotlib doesn't support mixed-size title lines natively, so we
            # use fig.text for the subtitle instead)
            ax.set_title(
                "Graph Permutation Distribution",
                fontsize=self.font_sizes["title"],
            )
            fig.text(
                0.5, 0.01, gs_text,
                ha="center", va="bottom",
                fontsize=self.font_sizes["annotation"] - 1,
                color="#6b7280",
            )
        else:
            ax.set_title(title_text, fontsize=self.font_sizes["title"])
        fig.tight_layout()

        return Figure(
            fig=fig,
            title="Graph Permutation Distribution",
            description="Does INDRA graph topology alone produce this signal?",
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 07: Proximity Decay
    # =========================================================================

    def plot_proximity_decay(
        self,
        phase6: dict,
        decay_df: pd.DataFrame | None = None,
        figsize: tuple[float, float] = (8, 5),
    ) -> Figure:
        """
        Bar chart of mean |t-stat| by shortest-path distance from seed gene.
        """
        fig, ax = plt.subplots(figsize=figsize)

        prox = phase6.get("proximity_decay", {})
        rho = prox.get("spearman_rho")
        perm_p = prox.get("permutation_pvalue")
        seed = prox.get("seed_gene", "seed")

        # Prefer CSV if available, fall back to JSON distance_bins
        if decay_df is not None and "distance" in decay_df.columns:
            df = decay_df.sort_values("distance")
        elif "distance_bins" in prox:
            rows = []
            for dist_str, vals in prox["distance_bins"].items():
                rows.append({"distance": int(dist_str), **vals})
            df = pd.DataFrame(rows).sort_values("distance")
        else:
            ax.text(0.5, 0.5, "Proximity data unavailable",
                    ha="center", va="center", transform=ax.transAxes)
            return Figure(fig=fig, title="Proximity Decay",
                          description="No data.", figure_type="matplotlib")

        distances = df["distance"].values
        means = df["mean_abs_t"].values
        stds = df.get("std_abs_t", pd.Series([0] * len(df))).values
        n_genes = df.get("n_genes", pd.Series([0] * len(df))).values

        # Gradient colors: blue at dist=1 → gray at max
        n_bars = len(distances)
        bar_colors = []
        for i in range(n_bars):
            frac = i / max(n_bars - 1, 1)
            r = int(0x25 + (0x94 - 0x25) * frac)
            g = int(0x63 + (0xa3 - 0x63) * frac)
            b = int(0xeb + (0xb8 - 0xeb) * frac)
            bar_colors.append(f"#{r:02x}{g:02x}{b:02x}")

        bars = ax.bar(
            distances, means, color=bar_colors,
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        ax.errorbar(
            distances, means, yerr=stds,
            fmt="none", color="#64748b", capsize=4, zorder=4,
        )

        # Gene count labels
        for x, y, n in zip(distances, means, n_genes):
            if n > 0:
                ax.text(
                    x, y + stds[list(distances).index(x)] + 0.02,
                    f"n={int(n)}",
                    ha="center", va="bottom",
                    fontsize=self.font_sizes["annotation"],
                    color="#6b7280",
                )

        # Correlation annotation
        ann_parts = []
        if rho is not None:
            ann_parts.append(f"Spearman ρ = {rho:.3f}")
        if perm_p is not None:
            ann_parts.append(f"Permutation {format_pvalue(perm_p)}")
        if ann_parts:
            ax.text(
                0.98, 0.98, "\n".join(ann_parts),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=self.font_sizes["annotation"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb"),
            )

        ax.set_xlabel(f"Shortest-path distance from {italicize_gene(seed)}",
                       fontsize=self.font_sizes["label"])
        ax.set_ylabel("Mean |t-statistic|", fontsize=self.font_sizes["label"])
        ax.set_xticks(distances)
        ax.set_title("Proximity Decay", fontsize=self.font_sizes["title"])
        fig.tight_layout()

        return Figure(
            fig=fig,
            title="Proximity Decay",
            description="Do genes closer to the seed show stronger differential expression?",
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 08: Reverse Causal Ranking
    # =========================================================================

    def plot_reverse_causal(
        self,
        phase6: dict,
        regulators_df: pd.DataFrame | None = None,
        figsize: tuple[float, float] = (8, 5),
    ) -> Figure:
        """
        Horizontal lollipop chart of upstream regulators ranked by z-score.
        """
        fig, ax = plt.subplots(figsize=figsize)

        rc = phase6.get("reverse_causal", {})
        query_gene = rc.get("query_gene", "?")
        query_rank = rc.get("query_gene_rank")
        n_tested = rc.get("n_regulators_tested", 0)
        n_up = rc.get("n_up_submitted", 0)
        n_down = rc.get("n_down_submitted", 0)

        # Get regulator data
        if regulators_df is not None and "regulator" in regulators_df.columns:
            df = regulators_df.sort_values("zscore", ascending=True)
        elif "top_regulators" in rc:
            df = pd.DataFrame(rc["top_regulators"]).sort_values("zscore", ascending=True)
        else:
            ax.text(0.5, 0.5, "Regulator data unavailable",
                    ha="center", va="center", transform=ax.transAxes)
            return Figure(fig=fig, title="Reverse Causal Ranking",
                          description="No data.", figure_type="matplotlib")

        names = df["regulator"].values
        zscores = df["zscore"].values
        y_pos = np.arange(len(names))

        # Color by sign
        colors = [
            VALIDATION_COLORS["observed"] if z < 0 else "#f97316"
            for z in zscores
        ]

        # Stems
        ax.hlines(y_pos, 0, zscores, colors=colors, linewidth=1.5, zorder=2)
        # Dots
        ax.scatter(zscores, y_pos, c=colors, s=50, zorder=3, edgecolors="white", linewidths=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [italicize_gene(n) for n in names],
            fontsize=self.font_sizes["label"],
        )
        ax.axvline(0, color="#e5e7eb", linewidth=1, zorder=1)
        ax.set_xlabel("z-score", fontsize=self.font_sizes["label"])

        # Query gene rank annotation
        ann_parts = []
        if query_rank is not None:
            ann_parts.append(f"{italicize_gene(query_gene)} rank: {query_rank}/{n_tested + 1}")
        ann_parts.append(f"Submitted: {n_up} up, {n_down} down")
        ax.text(
            0.98, 0.02, "\n".join(ann_parts),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=self.font_sizes["annotation"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb"),
        )

        ax.set_title("Reverse Causal Ranking", fontsize=self.font_sizes["title"])
        fig.tight_layout()

        return Figure(
            fig=fig,
            title="Reverse Causal Ranking",
            description="Which upstream regulators emerge from all DE genes?",
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 09: Differential Landscape (Volcano)
    # =========================================================================

    def plot_differential_landscape(
        self,
        protein_df: pd.DataFrame,
        figsize: tuple[float, float] = (10, 6),
    ) -> Figure:
        """
        Volcano plot with target genes highlighted.
        """
        fig, ax = plt.subplots(figsize=figsize)

        df = protein_df.copy()
        if "p_value" not in df.columns or "log2fc" not in df.columns:
            ax.text(0.5, 0.5, "Required columns missing (log2fc, p_value)",
                    ha="center", va="center", transform=ax.transAxes)
            return Figure(fig=fig, title="Differential Landscape",
                          description="No data.", figure_type="matplotlib")

        df["neg_log10p"] = -np.log10(df["p_value"].clip(lower=1e-300))

        # Background
        bg = df[df.get("is_target", False) != True]  # noqa: E712
        targets = df[df.get("is_target", False) == True]  # noqa: E712

        ax.scatter(
            bg["log2fc"], bg["neg_log10p"],
            c="#d1d5db", s=8, alpha=0.3, zorder=2,
            rasterized=True,
        )

        # Targets
        ax.scatter(
            targets["log2fc"], targets["neg_log10p"],
            c=VALIDATION_COLORS["observed"], s=40, alpha=0.8, zorder=3,
            edgecolors="white", linewidths=0.5,
        )

        # Label targets with adjustText to avoid overlaps
        if "gene_symbol" in targets.columns:
            from adjustText import adjust_text

            texts = []
            for _, row in targets.iterrows():
                label = row["gene_symbol"]
                if pd.notna(label) and str(label).strip():
                    texts.append(
                        ax.text(
                            row["log2fc"], row["neg_log10p"],
                            italicize_gene(str(label)),
                            fontsize=self.font_sizes["annotation"],
                            color=VALIDATION_COLORS["observed"],
                            zorder=4,
                        )
                    )
            if texts:
                # Only repel from target points (not 3000+ background points)
                target_x = list(targets["log2fc"].values)
                target_y = list(targets["neg_log10p"].values)
                adjust_text(
                    texts, x=target_x, y=target_y, ax=ax,
                    force_text=(3.0, 3.0),
                    force_points=(5.0, 5.0),
                    expand=(2.5, 2.5),
                    min_arrow_len=5,
                    lim=500,
                    arrowprops=dict(arrowstyle="-", color="#94a3b8", linewidth=0.5),
                )

        # Threshold lines
        sig_line = -np.log10(0.05)
        ax.axhline(sig_line, color="#e5e7eb", linestyle="--", linewidth=0.8, zorder=1)
        ax.text(
            ax.get_xlim()[1] * 0.98, sig_line + 0.1, "α = 0.05",
            ha="right", va="bottom",
            fontsize=self.font_sizes["annotation"] - 1, color="#9ca3af",
        )

        # Fold-change reference lines
        for fc in [-0.5, 0.5]:
            ax.axvline(fc, color="#f3f4f6", linestyle=":", linewidth=0.8, zorder=1)

        ax.set_xlabel("$\\log_2$(fold change)", fontsize=self.font_sizes["label"])
        ax.set_ylabel("$-\\log_{10}$(p-value)", fontsize=self.font_sizes["label"])
        ax.set_title("Differential Landscape", fontsize=self.font_sizes["title"])

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#d1d5db",
                   markersize=6, label=f"Background (n={len(bg)})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=VALIDATION_COLORS["observed"],
                   markersize=8, label=f"Targets (n={len(targets)})"),
        ]
        ax.legend(handles=legend_elements, fontsize=self.font_sizes["annotation"], loc="upper left")

        fig.tight_layout()
        return Figure(
            fig=fig,
            title="Differential Landscape",
            description="Volcano plot of all proteins with target genes highlighted.",
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 10: Evidence Summary Table
    # =========================================================================

    def plot_evidence_table(
        self,
        report: dict,
        figsize: tuple[float, float] = (14, 5),
    ) -> Figure:
        """
        Clean matplotlib table rendering all phase results.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        phases = report.get("phases", {})
        phase_details = report.get("phase_details", {})

        # Build table data
        col_labels = ["Phase", "Test", "Key Statistic", "P-value", "Status"]
        table_data = []
        row_colors = []

        for key, label, is_gate in _PHASE_META:
            if key not in phases and key not in phase_details:
                continue

            phase = phases.get(key, {})
            detail = phase_details.get(key, "")

            # Extract key statistic and p-value
            stat_str = ""
            p_str = ""
            status = "–"

            if key == "covariate_adjusted":
                stat_str = f"z = {phase.get('z_score', 0):.2f}"
                p_val = phase.get("empirical_pvalue")
                p_str = format_pvalue(p_val) if p_val else "–"
            elif key == "label_permutation":
                strat = phase.get("stratified", phase)
                stat_str = f"z = {strat.get('observed_z', 0):.2f}"
                p_val = strat.get("permutation_pvalue")
                p_str = format_pvalue(p_val) if p_val else "–"
            elif key == "specificity":
                stat_str = f"label: {phase.get('specificity_label', '?')}"
                int_test = phase.get("interaction_test", {})
                p_val = int_test.get("interaction_pvalue")
                p_str = format_pvalue(p_val) if p_val else "–"
            elif key == "matched_reanalysis":
                stat_str = f"z = {phase.get('z_score', 0):.2f}"
                p_val = phase.get("empirical_pvalue")
                p_str = format_pvalue(p_val) if p_val else "–"
            elif key == "negative_controls":
                cz = phase.get("competitive_z", {})
                stat_str = f"pctl = {cz.get('percentile', phase.get('target_percentile', '?'))}%"
                p_str = f"FPR = {cz.get('fpr', phase.get('fpr', '?'))}"
            elif key == "graph_permutation":
                stat_str = f"pctl = {phase.get('target_percentile', '?')}%"
                p_str = f"FPR = {phase.get('fpr', '?')}"
            elif key == "network_proximity":
                prox = phase.get("proximity_decay", {})
                stat_str = f"ρ = {prox.get('spearman_rho', 0):.3f}"
                p_val = prox.get("permutation_pvalue")
                p_str = format_pvalue(p_val) if p_val else "–"

            # Determine status from detail
            detail_lower = detail.lower() if detail else ""
            if "(pass)" in detail_lower:
                status = "PASS"
                row_colors.append(("#ecfdf5", VALIDATION_COLORS["pass"]))
            elif "(fail)" in detail_lower:
                status = "FAIL"
                row_colors.append(("#fef2f2", VALIDATION_COLORS["fail"]))
            else:
                status = "–"
                row_colors.append(("#ffffff", "#6b7280"))

            gate_marker = " [GATE]" if is_gate else ""
            table_data.append([f"{label}{gate_marker}", key.replace("_", " ").title(),
                              stat_str, p_str, status])

        if not table_data:
            ax.text(0.5, 0.5, "No phase data available",
                    ha="center", va="center", transform=ax.transAxes)
            return Figure(fig=fig, title="Evidence Summary",
                          description="No data.", figure_type="matplotlib")

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(self.font_sizes["label"])
        table.scale(1, 1.6)

        # Style header
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", fontweight="bold")

        # Style rows
        for i, (bg_color, text_color) in enumerate(row_colors):
            for j in range(len(col_labels)):
                cell = table[i + 1, j]
                cell.set_facecolor(bg_color)
                # Bold the status column
                if j == 4:
                    cell.set_text_props(color=text_color, fontweight="bold")
                # Bold gate phases
                if j == 0 and "[GATE]" in table_data[i][0]:
                    cell.set_text_props(fontweight="bold")

        ax.set_title(
            "Evidence Summary",
            fontsize=self.font_sizes["title"],
            fontweight="bold", pad=20,
        )

        fig.tight_layout()
        return Figure(
            fig=fig,
            title="Evidence Summary Table",
            description="All validation phases with key statistics in one table.",
            figure_type="matplotlib",
        )

    # =========================================================================
    # Act IV — Recursive Discovery (Figures 11-15)
    #
    # These figures visualize the multi-hop recursive discovery results.
    # The narrative arc: "How far does the signal reach, who carries it,
    # is it specific to C9orf72, and how do we know when to stop?"
    # =========================================================================

    # =========================================================================
    # Figure 11: Cascade Staircase — "How far does the signal reach?"
    # =========================================================================

    def plot_cascade_staircase(
        self,
        discovery: dict,
        figsize: tuple[float, float] = (10, 5.5),
    ) -> Figure:
        """
        Hero figure: hop-by-hop bar chart of tested/significant with π₀ line.

        Left y-axis: counts (bars). Right y-axis: π₀ (line).
        The visual contrast between tall "tested" bars and nearly-as-tall
        "significant" bars tells the story instantly: the signal persists.
        """
        hops_data = discovery.get("hops", [])
        if not hops_data:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No discovery data", ha="center", va="center",
                    transform=ax.transAxes)
            return Figure(fig=fig, title="Cascade Staircase",
                          description="No data.", figure_type="matplotlib")

        hop_nums = [h["hop"] for h in hops_data]
        n_tested = [h["n_intermediaries_tested"] for h in hops_data]
        n_sig = [h["n_significant"] for h in hops_data]
        pi0s = [h.get("pi0") for h in hops_data]

        fig, ax1 = plt.subplots(figsize=figsize)
        x = np.arange(len(hop_nums))
        bar_width = 0.35

        # Tested bars (recede — muted gray)
        ax1.bar(
            x - bar_width / 2, n_tested, bar_width,
            color="#e2e8f0", edgecolor="#94a3b8", linewidth=0.8,
            label="Tested", zorder=2,
        )
        # Significant bars (advance — signal blue)
        ax1.bar(
            x + bar_width / 2, n_sig, bar_width,
            color=VALIDATION_COLORS["observed"], edgecolor="#1d4ed8",
            linewidth=0.8, label="Significant", zorder=2,
        )

        ax1.set_xlabel("Hop", fontsize=self.font_sizes["label"])
        ax1.set_ylabel("Intermediaries", fontsize=self.font_sizes["label"])
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(h) for h in hop_nums])

        # Symlog scale: makes hop 1 (n=1) and hop 2 (n=46) visible
        # alongside hop 3-5 (n~2000) without losing proportionality
        max_count = max(max(n_tested), max(n_sig))
        if max_count > 200:
            ax1.set_yscale("symlog", linthresh=10)

        # Annotate counts on bars
        for i, (nt, ns) in enumerate(zip(n_tested, n_sig)):
            if nt > 0:
                pct = ns / nt * 100
                # Position annotation above the taller bar
                y_top = max(nt, ns)
                ax1.text(
                    i, y_top * 1.15 if y_top > 10 else y_top + 1,
                    f"{ns}/{nt}  ({pct:.0f}%)",
                    ha="center", va="bottom",
                    fontsize=self.font_sizes["annotation"],
                    color="#334155", fontweight="bold",
                )

        # π₀ on right axis (the methodological payoff line)
        valid_pi0 = [(i, p) for i, p in enumerate(pi0s) if p is not None]
        if valid_pi0:
            ax2 = ax1.twinx()
            pi0_x = [v[0] for v in valid_pi0]
            pi0_y = [v[1] for v in valid_pi0]
            ax2.plot(
                pi0_x, pi0_y,
                color="#dc2626", marker="o", markersize=7,
                linewidth=2.0, zorder=5, label=r"$\hat{\pi}_0$ (null fraction)",
            )
            ax2.set_ylabel(
                r"$\hat{\pi}_0$ (null fraction)",
                fontsize=self.font_sizes["label"], color="#dc2626",
            )
            ax2.set_ylim(-0.02, max(pi0_y) * 1.5 + 0.05)
            ax2.tick_params(axis="y", colors="#dc2626")

            # Annotate π₀ values (below the line to avoid bar label collisions)
            for px, py in zip(pi0_x, pi0_y):
                ax2.annotate(
                    f"{py:.3f}",
                    xy=(px, py), xytext=(6, -14),
                    textcoords="offset points",
                    fontsize=self.font_sizes["annotation"],
                    color="#dc2626", fontweight="bold",
                )

            ax2.legend(
                loc="lower right",
                fontsize=self.font_sizes["annotation"],
            )

        # Stop reason annotation — bottom-right, out of the data region
        last_hop = hops_data[-1]
        stop = last_hop.get("stop_reason", "")
        if stop:
            stop_labels = {
                "pi0_converged": "Stopped: pi0 converged (graph boundary)",
                "no_significant": "Stopped: no significant arms",
                "no_candidates": "Stopped: no candidates to extend",
                "graph_exhaustion": "Stopped: seed null (topology, not biology)",
            }
            stop_text = stop_labels.get(stop, stop)
            ax1.text(
                0.02, 0.02, stop_text,
                transform=ax1.transAxes,
                fontsize=self.font_sizes["annotation"],
                color="#6b7280", fontstyle="italic",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#e5e7eb", alpha=0.9),
            )

        seed = discovery.get("seed_gene", "seed")
        ax1.set_title(
            f"Recursive Discovery from {italicize_gene(seed)}",
            fontsize=self.font_sizes["title"], fontweight="bold",
        )
        ax1.legend(
            loc="upper left", fontsize=self.font_sizes["annotation"],
        )
        fig.tight_layout()

        return Figure(
            fig=fig,
            title="Cascade Staircase",
            description=(
                f"How far does {seed}'s regulatory signal reach through "
                f"the INDRA knowledge graph?"
            ),
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 12: Hop 2 Intermediary Detail — "Who carries the signal?"
    # =========================================================================

    def plot_hop2_intermediaries(
        self,
        discovery: dict,
        figsize: tuple[float, float] = (10, 10),
    ) -> Figure:
        """
        Horizontal lollipop: each of the hop-2 intermediaries, ordered by
        p-value. Point color encodes reliability; marker shape encodes
        direction (activation ▶ / repression ◀).

        The figure answers: which specific biological programs are enriched
        in C9orf72 carriers?
        """
        # Find hop 2
        hop2 = None
        for h in discovery.get("hops", []):
            if h["hop"] == 2:
                hop2 = h
                break

        if hop2 is None or not hop2.get("all_arms"):
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No hop 2 data", ha="center", va="center",
                    transform=ax.transAxes)
            return Figure(fig=fig, title="Hop 2 Intermediaries",
                          description="No data.", figure_type="matplotlib")

        arms = sorted(hop2["all_arms"], key=lambda a: a["p_value"])
        n_arms = len(arms)

        # Adaptive figsize for number of arms
        fig_h = max(5, n_arms * 0.28)
        fig, ax = plt.subplots(figsize=(figsize[0], fig_h))

        y_pos = np.arange(n_arms)
        p_values = [a["p_value"] for a in arms]
        reliabilities = [a.get("reliability", 0.5) for a in arms]
        directions = [a.get("direction", "unknown") for a in arms]
        n_targets = [a.get("n_targets", 0) for a in arms]
        names = [a["intermediary"] for a in arms]

        # Colormap: reliability → blue intensity
        # Low reliability = pale, high reliability = deep blue
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.Blues

        # Horizontal lollipop
        for i, (pv, rel, direction, nt, name) in enumerate(
            zip(p_values, reliabilities, directions, n_targets, names)
        ):
            # Stem
            ax.hlines(i, 0, -np.log10(pv), color="#cbd5e1", linewidth=1.0, zorder=1)
            # Marker: ▶ activation, ◀ repression, ● other
            if "activ" in direction.lower():
                marker = ">"
            elif "repress" in direction.lower():
                marker = "<"
            else:
                marker = "o"
            ax.scatter(
                -np.log10(pv), i,
                c=[rel], cmap=cmap, norm=norm,
                s=max(30, nt / 3),  # size ~ number of targets
                marker=marker, edgecolors="#1e40af", linewidth=0.6,
                zorder=3,
            )
            # Label
            ax.text(
                -np.log10(pv) + 0.08, i,
                f"{italicize_gene(name)}  ({nt})",
                va="center", fontsize=self.font_sizes["annotation"],
                color="#334155",
            )

        # FDR threshold line
        fdr_threshold = 0.05
        ax.axvline(
            -np.log10(fdr_threshold), color="#dc2626",
            linestyle="--", linewidth=1.0, alpha=0.7, zorder=2,
        )
        ax.text(
            -np.log10(fdr_threshold) + 0.05, n_arms - 0.5,
            "FDR = 0.05", color="#dc2626",
            fontsize=self.font_sizes["annotation"],
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([""] * n_arms)  # Labels are inline
        ax.set_xlabel("$-\\log_{10}$(p-value)", fontsize=self.font_sizes["label"])
        ax.invert_yaxis()

        # Colorbar for reliability
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.4, aspect=20, pad=0.02)
        cbar.set_label("Edge reliability", fontsize=self.font_sizes["annotation"])

        # Legend for direction markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker=">", color="w", markeredgecolor="#1e40af",
                   markerfacecolor="#93c5fd", markersize=8, label="Activation"),
            Line2D([0], [0], marker="<", color="w", markeredgecolor="#1e40af",
                   markerfacecolor="#93c5fd", markersize=8, label="Repression"),
            Line2D([0], [0], marker="o", color="w", markeredgecolor="#1e40af",
                   markerfacecolor="#93c5fd", markersize=6, label="Size = n targets"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=self.font_sizes["annotation"],
            framealpha=0.9,
        )

        seed = discovery.get("seed_gene", "seed")
        n_sig = hop2["n_significant"]
        ax.set_title(
            f"{italicize_gene(seed)} Hop 2: {n_sig}/{n_arms} intermediaries significant",
            fontsize=self.font_sizes["title"], fontweight="bold",
        )
        fig.tight_layout()

        return Figure(
            fig=fig,
            title="Hop 2 Intermediary Detail",
            description=(
                f"Each intermediary's downstream targets tested via ROAST. "
                f"Point color = edge reliability, shape = regulatory direction, "
                f"size = number of downstream targets."
            ),
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 13: π₀ Convergence — "How do we know when to stop?"
    # =========================================================================

    def plot_pi0_convergence(
        self,
        discovery: dict,
        figsize: tuple[float, float] = (7, 4.5),
    ) -> Figure:
        """
        Line plot of null fraction (π̂₀) across hops with convergence zone.

        The key insight: when π̂₀ stops changing between consecutive hops,
        every reachable regulatory gene has been tested.
        """
        hops_data = discovery.get("hops", [])
        valid = [(h["hop"], h["pi0"]) for h in hops_data if h.get("pi0") is not None]

        fig, ax = plt.subplots(figsize=figsize)

        if len(valid) < 2:
            ax.text(0.5, 0.5, "Insufficient pi0 data (need 2+ hops)",
                    ha="center", va="center", transform=ax.transAxes)
            return Figure(fig=fig, title="pi0 Convergence",
                          description="No data.", figure_type="matplotlib")

        hops_x, pi0_y = zip(*valid)

        # Main trajectory
        ax.plot(
            hops_x, pi0_y,
            color=VALIDATION_COLORS["observed"], marker="o",
            markersize=9, linewidth=2.5, zorder=5,
            markeredgecolor="white", markeredgewidth=1.5,
        )

        # Convergence band: shade the Δ<0.01 zone
        convergence_threshold = 0.01
        for i in range(1, len(pi0_y)):
            delta = abs(pi0_y[i] - pi0_y[i - 1])
            if delta < convergence_threshold:
                ax.axvspan(
                    hops_x[i - 1] - 0.3, hops_x[i] + 0.3,
                    color="#dcfce7", alpha=0.6, zorder=1,
                )
                ax.annotate(
                    f"Δ = {delta:.4f}\n< {convergence_threshold}",
                    xy=((hops_x[i - 1] + hops_x[i]) / 2, (pi0_y[i - 1] + pi0_y[i]) / 2),
                    xytext=(15, 20), textcoords="offset points",
                    fontsize=self.font_sizes["annotation"],
                    color="#059669", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7",
                              edgecolor="#059669", linewidth=0.8),
                )

        # Annotate each point
        for hx, py in zip(hops_x, pi0_y):
            ax.annotate(
                f"{py:.3f}",
                xy=(hx, py), xytext=(0, -18),
                textcoords="offset points", ha="center",
                fontsize=self.font_sizes["annotation"],
                fontweight="bold", color="#1e40af",
            )

        # Reference lines
        ax.axhline(0, color="#e2e8f0", linewidth=0.8, zorder=0)
        ax.axhline(1, color="#e2e8f0", linewidth=0.8, zorder=0)

        ax.set_xlabel("Hop", fontsize=self.font_sizes["label"])
        ax.set_ylabel(r"$\hat{\pi}_0$ (estimated null fraction)", fontsize=self.font_sizes["label"])
        ax.set_xticks(list(hops_x))
        ax.set_ylim(-0.02, min(1.05, max(pi0_y) * 1.5 + 0.05))

        seed = discovery.get("seed_gene", "seed")
        ax.set_title(
            f"Null Fraction Trajectory from {italicize_gene(seed)}",
            fontsize=self.font_sizes["title"], fontweight="bold",
        )
        fig.tight_layout()

        return Figure(
            fig=fig,
            title="Null Fraction Convergence",
            description=(
                "Estimated null fraction rises as the search moves away from "
                "the seed gene. Convergence (delta < 0.01) signals the graph boundary."
            ),
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 14: Specificity Triangle — "Is it C9-specific?"
    # =========================================================================

    def plot_specificity_triangle(
        self,
        contrasts: dict[str, dict],
        figsize: tuple[float, float] = (12, 5),
    ) -> Figure:
        """
        Three-contrast comparison at each hop level.

        Parameters
        ----------
        contrasts : dict
            Keys are contrast names (e.g. "C9 vs Sporadic"), values are
            discovery result dicts (output of run_discovery().to_dict()).

        The visual: grouped bars (one group per hop, one bar per contrast).
        The "aha" moment: C9 vs Sporadic fills the bars at every hop, while
        Sporadic vs Control is empty.
        """
        if not contrasts:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No contrast data", ha="center", va="center",
                    transform=ax.transAxes)
            return Figure(fig=fig, title="Specificity Triangle",
                          description="No data.", figure_type="matplotlib")

        # Semantic contrast colors
        contrast_colors = {
            "C9 vs Sporadic": "#2563eb",    # Signal blue — the primary finding
            "C9 vs Control": "#8b5cf6",     # Violet — intermediate
            "Sporadic vs Control": "#f97316",  # Orange — the null comparator
        }
        # Fallback for non-standard contrast names
        fallback_colors = ["#2563eb", "#8b5cf6", "#f97316", "#059669", "#dc2626"]

        contrast_names = list(contrasts.keys())
        colors = []
        for cn in contrast_names:
            if cn in contrast_colors:
                colors.append(contrast_colors[cn])
            else:
                colors.append(fallback_colors[len(colors) % len(fallback_colors)])

        # Collect all hop numbers across contrasts
        all_hops = set()
        for disc in contrasts.values():
            for h in disc.get("hops", []):
                all_hops.add(h["hop"])
        all_hops = sorted(all_hops)

        if not all_hops:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No hop data", ha="center", va="center",
                    transform=ax.transAxes)
            return Figure(fig=fig, title="Specificity Triangle",
                          description="No data.", figure_type="matplotlib")

        fig, (ax_pct, ax_pi0) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={"width_ratios": [3, 2], "wspace": 0.3},
        )

        n_contrasts = len(contrast_names)
        n_hops = len(all_hops)
        bar_width = 0.8 / n_contrasts
        x = np.arange(n_hops)

        # Left panel: % significant at each hop
        for ci, cn in enumerate(contrast_names):
            disc = contrasts[cn]
            hop_map = {h["hop"]: h for h in disc.get("hops", [])}
            pct_sig = []
            for hop in all_hops:
                h = hop_map.get(hop)
                if h and h["n_intermediaries_tested"] > 0:
                    pct_sig.append(h["n_significant"] / h["n_intermediaries_tested"] * 100)
                else:
                    pct_sig.append(0)

            offset = (ci - n_contrasts / 2 + 0.5) * bar_width
            bars = ax_pct.bar(
                x + offset, pct_sig, bar_width,
                color=colors[ci], edgecolor="white", linewidth=0.5,
                label=cn, zorder=2,
            )
            # Annotate n/N on bars
            for bi, hop in enumerate(all_hops):
                h = hop_map.get(hop)
                if h and h["n_significant"] > 0:
                    ax_pct.text(
                        x[bi] + offset, pct_sig[bi] + 1.5,
                        f"{h['n_significant']}/{h['n_intermediaries_tested']}",
                        ha="center", va="bottom",
                        fontsize=self.font_sizes["annotation"] - 1,
                        color=colors[ci], fontweight="bold",
                    )

        ax_pct.set_xlabel("Hop", fontsize=self.font_sizes["label"])
        ax_pct.set_ylabel("% Significant (FDR < 0.05)", fontsize=self.font_sizes["label"])
        ax_pct.set_xticks(x)
        ax_pct.set_xticklabels([str(h) for h in all_hops])
        ax_pct.set_ylim(0, 115)
        ax_pct.legend(fontsize=self.font_sizes["annotation"], loc="upper right")
        ax_pct.set_title("Signal Strength by Hop", fontsize=self.font_sizes["title"])

        # Right panel: π₀ trajectory per contrast
        for ci, cn in enumerate(contrast_names):
            disc = contrasts[cn]
            hop_pi0 = [
                (h["hop"], h["pi0"]) for h in disc.get("hops", [])
                if h.get("pi0") is not None
            ]
            if hop_pi0:
                hp, pi = zip(*hop_pi0)
                ax_pi0.plot(
                    hp, pi, marker="o", markersize=7,
                    linewidth=2.0, color=colors[ci], label=cn,
                    markeredgecolor="white", markeredgewidth=1.0,
                )
                # Annotate last point
                ax_pi0.annotate(
                    f"{pi[-1]:.2f}",
                    xy=(hp[-1], pi[-1]), xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=self.font_sizes["annotation"],
                    color=colors[ci], fontweight="bold",
                )

        ax_pi0.set_xlabel("Hop", fontsize=self.font_sizes["label"])
        ax_pi0.set_ylabel(r"$\hat{\pi}_0$ (null fraction)", fontsize=self.font_sizes["label"])
        ax_pi0.set_ylim(-0.02, 1.1)
        ax_pi0.axhline(1.0, color="#e2e8f0", linestyle=":", linewidth=0.8, zorder=0)
        ax_pi0.legend(fontsize=self.font_sizes["annotation"])
        ax_pi0.set_title("Null Fraction by Contrast", fontsize=self.font_sizes["title"])

        fig.suptitle(
            "Specificity Triangle: Three Contrasts, Same Graph",
            fontsize=self.font_sizes["title"] + 1, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        return Figure(
            fig=fig,
            title="Specificity Triangle",
            description=(
                "The same INDRA regulatory graph tested against three contrasts. "
                "C9orf72's cascade is C9-specific — not a general ALS feature."
            ),
            figure_type="matplotlib",
        )

    # =========================================================================
    # Figure 15: Hop 2 Specificity Heatmap — "Which arms are C9-specific?"
    # =========================================================================

    def plot_hop2_specificity_heatmap(
        self,
        contrasts: dict[str, dict],
        figsize: tuple[float, float] = (9, 10),
    ) -> Figure:
        """
        Heatmap of −log₁₀(p) for each hop-2 intermediary across contrasts.

        Rows: intermediaries (sorted by C9 vs Sporadic p-value).
        Columns: contrasts.

        The pattern: deep blue in the first column, pale in the second,
        white in the third.
        """
        if not contrasts:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No contrast data", ha="center", va="center",
                    transform=ax.transAxes)
            return Figure(fig=fig, title="Hop 2 Specificity Heatmap",
                          description="No data.", figure_type="matplotlib")

        contrast_names = list(contrasts.keys())

        # Extract hop 2 arms from each contrast
        arms_by_contrast: dict[str, dict[str, dict]] = {}
        for cn, disc in contrasts.items():
            for h in disc.get("hops", []):
                if h["hop"] == 2:
                    arms_by_contrast[cn] = {a["intermediary"]: a for a in h.get("all_arms", [])}
                    break

        if not arms_by_contrast:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No hop 2 arms", ha="center", va="center",
                    transform=ax.transAxes)
            return Figure(fig=fig, title="Hop 2 Specificity Heatmap",
                          description="No data.", figure_type="matplotlib")

        # Find all intermediaries present in any contrast; sort by primary contrast
        primary = contrast_names[0]
        all_intermediaries = set()
        for arms in arms_by_contrast.values():
            all_intermediaries.update(arms.keys())

        primary_arms = arms_by_contrast.get(primary, {})
        sorted_genes = sorted(
            all_intermediaries,
            key=lambda g: primary_arms.get(g, {}).get("p_value", 1.0),
        )

        # Build matrix: -log10(p)
        matrix = np.zeros((len(sorted_genes), len(contrast_names)))
        for ci, cn in enumerate(contrast_names):
            arms = arms_by_contrast.get(cn, {})
            for gi, gene in enumerate(sorted_genes):
                arm = arms.get(gene)
                if arm:
                    pv = arm["p_value"]
                    matrix[gi, ci] = -np.log10(max(pv, 1e-10))
                else:
                    matrix[gi, ci] = 0.0

        # Adaptive figsize
        fig_h = max(5, len(sorted_genes) * 0.22)
        fig, ax = plt.subplots(figsize=(figsize[0], fig_h))

        im = ax.imshow(
            matrix, aspect="auto", cmap="Blues",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(contrast_names)))
        ax.set_xticklabels(contrast_names, fontsize=self.font_sizes["label"], rotation=15, ha="right")
        ax.set_yticks(range(len(sorted_genes)))
        ax.set_yticklabels(
            [italicize_gene(g) for g in sorted_genes],
            fontsize=self.font_sizes["annotation"],
        )

        # Significance threshold line (−log10(0.05) ≈ 1.3)
        # Annotate cells that are significant
        for gi in range(len(sorted_genes)):
            for ci in range(len(contrast_names)):
                val = matrix[gi, ci]
                if val > -np.log10(0.05):
                    ax.text(
                        ci, gi, f"{10**(-val):.3f}",
                        ha="center", va="center",
                        fontsize=self.font_sizes["annotation"] - 2,
                        color="white" if val > 2.0 else "#334155",
                    )

        cbar = fig.colorbar(im, ax=ax, shrink=0.5, aspect=25, pad=0.02)
        cbar.set_label("$-\\log_{10}$(p)", fontsize=self.font_sizes["annotation"])

        ax.set_title(
            "Hop 2 Specificity: Intermediary × Contrast",
            fontsize=self.font_sizes["title"], fontweight="bold",
        )
        fig.tight_layout()

        return Figure(
            fig=fig,
            title="Hop 2 Specificity Heatmap",
            description=(
                "Each intermediary's enrichment p-value across three contrasts. "
                "Deep blue = strong signal; white = no signal."
            ),
            figure_type="matplotlib",
        )

    # =========================================================================
    # Orchestrator: Generate Report
    # =========================================================================

    def generate_report(
        self,
        data: dict[str, Any],
        *,
        skip_differential: bool = False,
        discovery_contrasts: dict[str, dict] | None = None,
    ) -> FigureCollection:
        """
        Generate all available validation figures.

        Parameters
        ----------
        data : dict
            Output of ``load_validation_data()``.
        skip_differential : bool
            Skip the volcano plot (large CSV).
        discovery_contrasts : dict, optional
            Multi-contrast discovery results for specificity figures.
            Keys are contrast names, values are discovery result dicts.

        Returns
        -------
        FigureCollection
            Numbered figures in narrative order.
        """
        collection = FigureCollection()
        report = data.get("report")

        if report is None:
            return collection

        # Act I: Verdict
        collection.add("01_verdict_scorecard", self.plot_verdict_scorecard(report))

        # Act II: Statistical Backbone
        if data.get("phase1"):
            collection.add(
                "02_covariate_enrichment",
                self.plot_covariate_enrichment(data["phase1"], data.get("protein_df")),
            )

        if data.get("phase3"):
            collection.add(
                "03_label_permutation",
                self.plot_label_permutation(data["phase3"], data.get("label_perm_dist")),
            )

        if data.get("phase2"):
            collection.add(
                "04_specificity",
                self.plot_specificity(data["phase2"]),
            )

        if data.get("phase5a"):
            collection.add(
                "05_negative_controls",
                self.plot_negative_controls(data["phase5a"], data.get("neg_ctrl_dist")),
            )

        if data.get("phase5b"):
            collection.add(
                "06_graph_permutation",
                self.plot_graph_permutation(data["phase5b"], data.get("graph_perm_dist")),
            )

        # Act III: Mechanistic Corroboration
        if data.get("phase6"):
            phase6 = data["phase6"]

            if "proximity_decay" in phase6:
                collection.add(
                    "07_proximity_decay",
                    self.plot_proximity_decay(phase6, data.get("decay_curve")),
                )

            if "reverse_causal" in phase6:
                collection.add(
                    "08_reverse_causal",
                    self.plot_reverse_causal(phase6, data.get("regulators")),
                )

        if not skip_differential and data.get("protein_df") is not None:
            collection.add(
                "09_differential_landscape",
                self.plot_differential_landscape(data["protein_df"]),
            )

        # Epilogue
        collection.add("10_evidence_table", self.plot_evidence_table(report))

        # Act IV: Recursive Discovery
        discovery = data.get("discovery")
        if discovery and discovery.get("hops"):
            collection.add(
                "11_cascade_staircase",
                self.plot_cascade_staircase(discovery),
            )
            collection.add(
                "12_hop2_intermediaries",
                self.plot_hop2_intermediaries(discovery),
            )
            collection.add(
                "13_pi0_convergence",
                self.plot_pi0_convergence(discovery),
            )

        if discovery_contrasts:
            collection.add(
                "14_specificity_triangle",
                self.plot_specificity_triangle(discovery_contrasts),
            )
            collection.add(
                "15_hop2_specificity_heatmap",
                self.plot_hop2_specificity_heatmap(discovery_contrasts),
            )

        return collection


# =============================================================================
# Helpers
# =============================================================================


def _wrap_text(text: str, max_chars: int = 40) -> str:
    """Wrap text at word boundaries."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > max_chars and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = f"{current_line} {word}".strip()
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)
