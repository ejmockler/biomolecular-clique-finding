"""
Differential clique abundance visualizations.

This module provides visualizations for ROAST-based differential clique analysis,
following perceptual engineering principles to support specific cognitive operations:

Visualization Hierarchy:
========================

Level 0 - Overview Dashboard (plot_significance_overview):
    Cognitive operation: "How did this experiment go overall?"
    - P-value distribution with threshold markers
    - Counts: total tested, significant at various thresholds
    - Direction breakdown: up vs down vs bidirectional

Level 1 - Ranked Clique Plot (plot_ranked_cliques):
    Cognitive operation: "Which cliques should I investigate?"
    - Manhattan-style or volcano-style ranking
    - Color encodes direction, size encodes clique size
    - Labels for top hits, threshold lines for significance

Level 2 - Direction Composition (plot_direction_composition):
    Cognitive operation: "Are these cliques going up, down, or mixed?"
    - Stacked bars showing active_proportion_up vs down
    - Identifies bidirectional regulation candidates
    - Sorted by significance

Level 3 - Single Clique Deep Dive (plot_clique_deep_dive):
    Cognitive operation: "What's driving this specific clique?"
    - Gene-level effect sizes
    - Mini correlation heatmap
    - Expression patterns by condition

Level 4 - Cross-Experiment Comparison (plot_experiment_comparison):
    Cognitive operation: "How do male vs female compare?"
    - Scatter: X = male -log10(p), Y = female -log10(p)
    - Quadrants identify sex-specific vs shared effects
    - Highlights interaction results

Design Philosophy:
==================
- Eye lands and knows where to go in <200ms
- Color is semantic (up=teal, down=orange, bidirectional=purple)
- Space encodes relationships (no lines/boxes needed where possible)
- Complexity visible but structure graspable in 2-3 seconds
- Working memory respected: max 4 simultaneous comparisons
"""

from __future__ import annotations

from typing import Optional, Literal
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
import seaborn as sns
from adjustText import adjust_text

from cliquefinder.viz.core import Figure, FigureCollection
from cliquefinder.viz.styles import Palette, PALETTES, configure_style, italicize_gene


# =============================================================================
# Color Semantics for Differential Analysis
# =============================================================================

DIFFERENTIAL_COLORS = {
    # Direction semantics
    "up": "#0d9488",        # Teal - upregulated in disease
    "down": "#f97316",      # Orange - downregulated in disease
    "bidirectional": "#7c3aed",  # Violet - mixed direction
    "neutral": "#94a3b8",   # Slate - not significant

    # Significance levels
    "sig_001": "#dc2626",   # Red - p < 0.001
    "sig_01": "#f59e0b",    # Amber - p < 0.01
    "sig_05": "#84cc16",    # Lime - p < 0.05
    "nonsig": "#e2e8f0",    # Light gray - not significant

    # Experiment comparison
    "male_only": "#0ea5e9",     # Sky blue
    "female_only": "#ec4899",   # Pink
    "both_sexes": "#8b5cf6",    # Purple
    "interaction": "#10b981",   # Emerald
}


class DifferentialCliqueVisualizer:
    """
    Visualizations for differential clique abundance analysis.

    Designed for ROAST rotation-based gene set test results.
    Follows perceptual engineering principles for cognitive alignment.

    Usage:
        viz = DifferentialCliqueVisualizer()

        # Load results
        results_df = pd.read_csv("roast_clique_results.csv")

        # Generate overview
        fig = viz.plot_significance_overview(results_df, title="Female sALS")
        fig.save("overview.png")

        # Generate ranked plot
        fig = viz.plot_ranked_cliques(results_df)
        fig.save("ranked.png")

        # Compare experiments
        fig = viz.plot_experiment_comparison(male_df, female_df)
        fig.save("comparison.png")
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

        # Typography scale (derived from base)
        self.font_sizes = {
            "paper": {"title": 14, "label": 11, "tick": 9, "annotation": 8},
            "presentation": {"title": 18, "label": 14, "tick": 12, "annotation": 10},
            "notebook": {"title": 12, "label": 10, "tick": 8, "annotation": 7},
        }[style]

    # =========================================================================
    # Level 0: Significance Overview Dashboard
    # =========================================================================

    def plot_significance_overview(
        self,
        results: pd.DataFrame,
        title: str = "Differential Clique Analysis",
        p_column: str = "pvalue_msq_mixed",
        figsize: tuple[float, float] = (10, 6),
    ) -> Figure:
        """
        Overview dashboard showing experiment summary statistics.

        Cognitive operation: "How did this experiment go overall?"

        Layout (3-panel horizontal):
        - Left: P-value distribution histogram
        - Center: Significance counts (bar chart)
        - Right: Direction breakdown (pie/donut)

        Args:
            results: DataFrame from ROAST analysis
            title: Plot title (e.g., "Female sALS CASE vs CTRL")
            p_column: Which p-value column to use
            figsize: Figure dimensions

        Returns:
            Figure wrapper with matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1, 1], wspace=0.3)

        # Extract p-values
        pvals = results[p_column].dropna()
        n_total = len(pvals)

        # Count significance levels
        n_001 = (pvals < 0.001).sum()
        n_01 = ((pvals >= 0.001) & (pvals < 0.01)).sum()
        n_05 = ((pvals >= 0.01) & (pvals < 0.05)).sum()
        n_ns = (pvals >= 0.05).sum()

        # Determine direction for significant cliques
        sig_mask = pvals < 0.05
        if "pvalue_mean_down" in results.columns and "pvalue_mean_up" in results.columns:
            down_dominant = (results["pvalue_mean_down"] < results["pvalue_mean_up"]) & sig_mask
            up_dominant = (results["pvalue_mean_up"] < results["pvalue_mean_down"]) & sig_mask

            # Bidirectional: both directions show signal
            bidir_mask = sig_mask & (results.get("active_proportion_up", 0) > 0.1) & (results.get("active_proportion_down", 0) > 0.1)

            n_up = up_dominant.sum() - bidir_mask.sum()
            n_down = down_dominant.sum() - bidir_mask.sum()
            n_bidir = bidir_mask.sum()
        else:
            n_up = n_down = n_bidir = 0

        # Panel 1: P-value histogram
        ax1 = fig.add_subplot(gs[0])
        bins = np.linspace(0, 1, 51)

        # Color bars by significance
        n, bin_edges, patches = ax1.hist(pvals, bins=bins, edgecolor="white", linewidth=0.5)
        for i, patch in enumerate(patches):
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            if bin_center < 0.001:
                patch.set_facecolor(DIFFERENTIAL_COLORS["sig_001"])
            elif bin_center < 0.01:
                patch.set_facecolor(DIFFERENTIAL_COLORS["sig_01"])
            elif bin_center < 0.05:
                patch.set_facecolor(DIFFERENTIAL_COLORS["sig_05"])
            else:
                patch.set_facecolor(DIFFERENTIAL_COLORS["nonsig"])

        # Add threshold lines
        ax1.axvline(0.001, color=DIFFERENTIAL_COLORS["sig_001"], linestyle="--", linewidth=1.5, alpha=0.7)
        ax1.axvline(0.01, color=DIFFERENTIAL_COLORS["sig_01"], linestyle="--", linewidth=1.5, alpha=0.7)
        ax1.axvline(0.05, color=DIFFERENTIAL_COLORS["sig_05"], linestyle="--", linewidth=1.5, alpha=0.7)

        ax1.set_xlabel("P-value", fontsize=self.font_sizes["label"])
        ax1.set_ylabel("Number of cliques", fontsize=self.font_sizes["label"])
        ax1.set_title("P-value Distribution", fontsize=self.font_sizes["title"], fontweight="bold")
        ax1.tick_params(labelsize=self.font_sizes["tick"])

        # Panel 2: Significance counts
        ax2 = fig.add_subplot(gs[1])
        categories = ["p < 0.001", "p < 0.01", "p < 0.05", "n.s."]
        counts = [n_001, n_01, n_05, n_ns]
        colors = [DIFFERENTIAL_COLORS["sig_001"], DIFFERENTIAL_COLORS["sig_01"],
                  DIFFERENTIAL_COLORS["sig_05"], DIFFERENTIAL_COLORS["nonsig"]]

        bars = ax2.bar(categories, counts, color=colors, edgecolor="white", linewidth=1)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.annotate(f"{count}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=self.font_sizes["annotation"],
                        fontweight="bold")

        ax2.set_ylabel("Number of cliques", fontsize=self.font_sizes["label"])
        ax2.set_title("Significance Levels", fontsize=self.font_sizes["title"], fontweight="bold")
        ax2.tick_params(labelsize=self.font_sizes["tick"])
        ax2.set_ylim(0, max(counts) * 1.15)

        # Panel 3: Direction breakdown (only if significant cliques exist)
        ax3 = fig.add_subplot(gs[2])

        if n_up + n_down + n_bidir > 0:
            direction_counts = [n_up, n_down, n_bidir]
            direction_labels = [f"Up\n({n_up})", f"Down\n({n_down})", f"Bidirectional\n({n_bidir})"]
            direction_colors = [DIFFERENTIAL_COLORS["up"], DIFFERENTIAL_COLORS["down"],
                               DIFFERENTIAL_COLORS["bidirectional"]]

            # Filter out zeros
            nonzero = [(c, l, col) for c, l, col in zip(direction_counts, direction_labels, direction_colors) if c > 0]
            if nonzero:
                counts_nz, labels_nz, colors_nz = zip(*nonzero)
                wedges, texts, autotexts = ax3.pie(
                    counts_nz, labels=labels_nz, colors=colors_nz,
                    autopct=lambda pct: f"{pct:.0f}%" if pct > 5 else "",
                    startangle=90,
                    wedgeprops={"edgecolor": "white", "linewidth": 2},
                    textprops={"fontsize": self.font_sizes["annotation"]}
                )
            else:
                ax3.text(0.5, 0.5, "No significant\ncliques", ha="center", va="center",
                        fontsize=self.font_sizes["label"], transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, "Direction data\nnot available", ha="center", va="center",
                    fontsize=self.font_sizes["label"], transform=ax3.transAxes)

        ax3.set_title("Effect Direction\n(p < 0.05)", fontsize=self.font_sizes["title"], fontweight="bold")

        # Overall title
        fig.suptitle(title, fontsize=self.font_sizes["title"] + 2, fontweight="bold", y=1.02)

        # Summary annotation
        summary_text = f"n = {n_total} cliques tested"
        fig.text(0.5, -0.02, summary_text, ha="center", fontsize=self.font_sizes["annotation"],
                style="italic", color="#64748b")

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Significance Overview",
            description=f"Differential clique analysis overview: {title}",
            figure_type="matplotlib",
            metadata={"n_total": n_total, "n_sig_05": int((pvals < 0.05).sum())}
        )

    # =========================================================================
    # Level 1: Ranked Clique Plot (Manhattan/Volcano style)
    # =========================================================================

    def plot_ranked_cliques(
        self,
        results: pd.DataFrame,
        style: Literal["manhattan", "volcano"] = "volcano",
        p_column: str = "pvalue_msq_mixed",
        effect_column: str = "observed_mean",
        n_labels: int = 15,
        significance_thresholds: list[float] = [0.001, 0.01, 0.05],
        figsize: tuple[float, float] = (12, 8),
        title: str = "Clique Differential Abundance",
    ) -> Figure:
        """
        Ranked visualization of all cliques by significance.

        Cognitive operation: "Which cliques should I investigate?"

        Manhattan style: X = clique index (ranked), Y = -log10(p)
        Volcano style: X = effect size, Y = -log10(p)

        Args:
            results: DataFrame from ROAST analysis
            style: "manhattan" or "volcano"
            p_column: Which p-value column to use
            effect_column: Which effect size column to use (for volcano)
            n_labels: Number of top cliques to label
            significance_thresholds: List of p-value thresholds to draw
            figsize: Figure dimensions
            title: Plot title

        Returns:
            Figure wrapper with matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data
        df = results.copy()
        df["neglog10p"] = -np.log10(df[p_column].clip(lower=1e-10))

        # Determine direction for coloring
        if "pvalue_mean_down" in df.columns and "pvalue_mean_up" in df.columns:
            df["direction"] = "neutral"
            sig_mask = df[p_column] < 0.05
            df.loc[sig_mask & (df["pvalue_mean_down"] < df["pvalue_mean_up"]), "direction"] = "down"
            df.loc[sig_mask & (df["pvalue_mean_up"] < df["pvalue_mean_down"]), "direction"] = "up"

            # Bidirectional
            if "active_proportion_up" in df.columns:
                bidir_mask = sig_mask & (df["active_proportion_up"] > 0.1) & (df["active_proportion_down"] > 0.1)
                df.loc[bidir_mask, "direction"] = "bidirectional"
        else:
            df["direction"] = "neutral"
            df.loc[df[p_column] < 0.05, "direction"] = "up"

        # Color mapping
        color_map = {
            "up": DIFFERENTIAL_COLORS["up"],
            "down": DIFFERENTIAL_COLORS["down"],
            "bidirectional": DIFFERENTIAL_COLORS["bidirectional"],
            "neutral": DIFFERENTIAL_COLORS["neutral"],
        }
        colors = df["direction"].map(color_map)

        # Size by clique size
        sizes = 20 + df["n_genes_found"] * 8
        sizes = sizes.clip(upper=200)

        if style == "manhattan":
            # Sort by p-value for x-axis ordering
            df = df.sort_values(p_column)
            df["rank"] = range(len(df))
            x = df["rank"]
            xlabel = "Clique rank (by p-value)"
        else:  # volcano
            # Parse effect size from dict string
            if effect_column in df.columns:
                effect_vals = []
                for val in df[effect_column]:
                    if isinstance(val, str) and "down" in val:
                        try:
                            parsed = eval(val)
                            # Use signed effect: positive for up, negative for down
                            effect_vals.append(parsed.get("down", 0))
                        except:
                            effect_vals.append(0)
                    else:
                        effect_vals.append(0)
                df["effect_size"] = effect_vals
                x = df["effect_size"]
            else:
                x = np.zeros(len(df))
            xlabel = "Effect size (mean t-statistic)"

        # Scatter plot
        scatter = ax.scatter(
            x, df["neglog10p"],
            c=colors,
            s=sizes,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

        # Threshold lines
        for thresh in significance_thresholds:
            y_thresh = -np.log10(thresh)
            ax.axhline(y_thresh, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(ax.get_xlim()[1], y_thresh, f" p={thresh}",
                   va="center", ha="left", fontsize=self.font_sizes["annotation"],
                   color="#64748b")

        # Label top cliques using adjustText for non-overlapping placement
        top_cliques = df.nsmallest(n_labels, p_column)

        texts = []
        for _, row in top_cliques.iterrows():
            x_pos = row["rank"] if style == "manhattan" else row.get("effect_size", 0)
            y_pos = row["neglog10p"]
            label = row["feature_set_id"]

            txt = ax.text(
                x_pos, y_pos,
                italicize_gene(label),
                fontsize=self.font_sizes["annotation"],
                fontweight="bold",
                color="#1e293b",
            )
            texts.append(txt)

        # Adjust text positions to avoid overlaps
        if texts:
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.5),
                expand_points=(1.5, 1.5),
                force_points=(0.5, 0.5),
                force_text=(0.5, 0.5),
            )

        # Axis labels and title
        ax.set_xlabel(xlabel, fontsize=self.font_sizes["label"])
        ax.set_ylabel(r"-log$_{10}$(p-value)", fontsize=self.font_sizes["label"])
        ax.set_title(title, fontsize=self.font_sizes["title"], fontweight="bold")
        ax.tick_params(labelsize=self.font_sizes["tick"])

        # Legend for direction
        legend_elements = [
            mpatches.Patch(color=DIFFERENTIAL_COLORS["up"], label="Upregulated"),
            mpatches.Patch(color=DIFFERENTIAL_COLORS["down"], label="Downregulated"),
            mpatches.Patch(color=DIFFERENTIAL_COLORS["bidirectional"], label="Bidirectional"),
            mpatches.Patch(color=DIFFERENTIAL_COLORS["neutral"], label="Not significant"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=self.font_sizes["annotation"])

        # Clean up
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Ranked Cliques",
            description=f"Ranked clique plot ({style} style)",
            figure_type="matplotlib",
            metadata={"style": style, "n_cliques": len(df)}
        )

    # =========================================================================
    # Level 2: Direction Composition
    # =========================================================================

    def plot_direction_composition(
        self,
        results: pd.DataFrame,
        p_threshold: float = 0.05,
        max_cliques: int = 30,
        figsize: tuple[float, float] = (14, 8),
        title: str = "Effect Direction Composition",
    ) -> Figure:
        """
        Stacked bar chart showing direction composition for significant cliques.

        Cognitive operation: "Are cliques going up, down, or mixed?"

        Each bar shows the fraction of genes upregulated vs downregulated
        within that clique. Bidirectional cliques have both colors visible.

        Args:
            results: DataFrame from ROAST analysis
            p_threshold: Significance threshold for inclusion
            max_cliques: Maximum number of cliques to show
            figsize: Figure dimensions
            title: Plot title

        Returns:
            Figure wrapper with matplotlib figure
        """
        # Filter significant cliques
        df = results[results["pvalue_msq_mixed"] < p_threshold].copy()

        if len(df) == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"No cliques with p < {p_threshold}",
                   ha="center", va="center", fontsize=self.font_sizes["label"])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return Figure(fig=fig, title="Direction Composition",
                         description="No significant cliques", figure_type="matplotlib")

        # Sort by significance and limit
        df = df.nsmallest(max_cliques, "pvalue_msq_mixed")

        # Get active proportions
        up_prop = df["active_proportion_up"].fillna(0)
        down_prop = df["active_proportion_down"].fillna(0)

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(df))
        width = 0.8

        # Stacked bars
        bars_up = ax.bar(x, up_prop, width, label="Upregulated",
                        color=DIFFERENTIAL_COLORS["up"], edgecolor="white")
        bars_down = ax.bar(x, down_prop, width, bottom=up_prop, label="Downregulated",
                          color=DIFFERENTIAL_COLORS["down"], edgecolor="white")

        # X-axis labels (clique names)
        ax.set_xticks(x)
        ax.set_xticklabels([italicize_gene(name) for name in df["feature_set_id"]],
                          rotation=45, ha="right", fontsize=self.font_sizes["tick"])

        # Y-axis
        ax.set_ylabel("Active proportion", fontsize=self.font_sizes["label"])
        ax.set_ylim(0, 1.1)

        # Add p-value annotations on top
        for i, (_, row) in enumerate(df.iterrows()):
            p = row["pvalue_msq_mixed"]
            if p < 0.001:
                label = "***"
            elif p < 0.01:
                label = "**"
            else:
                label = "*"

            total_height = row["active_proportion_up"] + row["active_proportion_down"]
            ax.text(i, min(total_height + 0.02, 1.05), label,
                   ha="center", va="bottom", fontsize=self.font_sizes["annotation"])

        ax.set_title(title, fontsize=self.font_sizes["title"], fontweight="bold")
        ax.legend(loc="upper right", fontsize=self.font_sizes["annotation"])

        # Clean up
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Direction Composition",
            description="Active proportion by direction for significant cliques",
            figure_type="matplotlib",
            metadata={"n_shown": len(df), "p_threshold": p_threshold}
        )

    # =========================================================================
    # Level 4: Cross-Experiment Comparison
    # =========================================================================

    def plot_experiment_comparison(
        self,
        results_a: pd.DataFrame,
        results_b: pd.DataFrame,
        label_a: str = "Experiment A",
        label_b: str = "Experiment B",
        p_column: str = "pvalue_msq_mixed",
        significance_threshold: float = 0.05,
        n_labels: int = 10,
        figsize: tuple[float, float] = (10, 10),
        title: str = "Cross-Experiment Comparison",
    ) -> Figure:
        """
        Scatter plot comparing p-values between two experiments.

        Cognitive operation: "How do experiments A and B compare?"

        Quadrants:
        - Top-right: Significant in BOTH
        - Top-left: Significant only in B
        - Bottom-right: Significant only in A
        - Bottom-left: Not significant in either

        Args:
            results_a: DataFrame from first experiment
            results_b: DataFrame from second experiment
            label_a: Label for first experiment (e.g., "Male")
            label_b: Label for second experiment (e.g., "Female")
            p_column: Which p-value column to use
            significance_threshold: P-value threshold for quadrant coloring
            n_labels: Number of interesting cliques to label
            figsize: Figure dimensions
            title: Plot title

        Returns:
            Figure wrapper with matplotlib figure
        """
        # Merge on clique ID
        merged = pd.merge(
            results_a[["feature_set_id", p_column]].rename(columns={p_column: "p_a"}),
            results_b[["feature_set_id", p_column]].rename(columns={p_column: "p_b"}),
            on="feature_set_id",
            how="inner"
        )

        # Transform to -log10
        merged["neglog_a"] = -np.log10(merged["p_a"].clip(lower=1e-10))
        merged["neglog_b"] = -np.log10(merged["p_b"].clip(lower=1e-10))

        # Categorize quadrants
        thresh = -np.log10(significance_threshold)
        merged["category"] = "neither"
        merged.loc[(merged["neglog_a"] >= thresh) & (merged["neglog_b"] < thresh), "category"] = "a_only"
        merged.loc[(merged["neglog_a"] < thresh) & (merged["neglog_b"] >= thresh), "category"] = "b_only"
        merged.loc[(merged["neglog_a"] >= thresh) & (merged["neglog_b"] >= thresh), "category"] = "both"

        # Color mapping
        cat_colors = {
            "a_only": DIFFERENTIAL_COLORS["male_only"],
            "b_only": DIFFERENTIAL_COLORS["female_only"],
            "both": DIFFERENTIAL_COLORS["both_sexes"],
            "neither": DIFFERENTIAL_COLORS["neutral"],
        }

        fig, ax = plt.subplots(figsize=figsize)

        # Plot each category
        for cat, color in cat_colors.items():
            subset = merged[merged["category"] == cat]
            ax.scatter(
                subset["neglog_a"], subset["neglog_b"],
                c=color, alpha=0.6, s=50, label=f"{cat.replace('_', ' ').title()} (n={len(subset)})",
                edgecolors="white", linewidths=0.5
            )

        # Threshold lines
        ax.axhline(thresh, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(thresh, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7)

        # Diagonal reference line
        max_val = max(merged["neglog_a"].max(), merged["neglog_b"].max())
        ax.plot([0, max_val], [0, max_val], color="#cbd5e1", linestyle=":", linewidth=1)

        # Label interesting cliques (significant in at least one)
        interesting = merged[merged["category"] != "neither"].copy()
        # Label interesting cliques using adjustText
        texts = []
        if len(interesting) > 0:
            # Score by maximum significance
            interesting["max_sig"] = interesting[["neglog_a", "neglog_b"]].max(axis=1)
            top = interesting.nlargest(n_labels, "max_sig")

            for _, row in top.iterrows():
                txt = ax.text(
                    row["neglog_a"], row["neglog_b"],
                    italicize_gene(row["feature_set_id"]),
                    fontsize=self.font_sizes["annotation"],
                    fontweight="bold",
                    color="#1e293b",
                )
                texts.append(txt)

        # Adjust text positions to avoid overlaps
        if texts:
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.5),
                expand_points=(1.5, 1.5),
                force_points=(0.5, 0.5),
                force_text=(0.5, 0.5),
            )

        # Axis labels
        ax.set_xlabel(r"-log$_{10}$(p) in " + label_a, fontsize=self.font_sizes["label"])
        ax.set_ylabel(r"-log$_{10}$(p) in " + label_b, fontsize=self.font_sizes["label"])
        ax.set_title(title, fontsize=self.font_sizes["title"], fontweight="bold")
        ax.tick_params(labelsize=self.font_sizes["tick"])

        # Legend
        ax.legend(loc="lower right", fontsize=self.font_sizes["annotation"])

        # Quadrant labels
        ax.text(0.95, 0.05, f"{label_a} only", transform=ax.transAxes,
               ha="right", va="bottom", fontsize=self.font_sizes["annotation"],
               color=DIFFERENTIAL_COLORS["male_only"], fontweight="bold")
        ax.text(0.05, 0.95, f"{label_b} only", transform=ax.transAxes,
               ha="left", va="top", fontsize=self.font_sizes["annotation"],
               color=DIFFERENTIAL_COLORS["female_only"], fontweight="bold")
        ax.text(0.95, 0.95, "Both", transform=ax.transAxes,
               ha="right", va="top", fontsize=self.font_sizes["annotation"],
               color=DIFFERENTIAL_COLORS["both_sexes"], fontweight="bold")

        # Clean up
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        plt.tight_layout()

        # Count summary
        counts = merged["category"].value_counts()

        return Figure(
            fig=fig,
            title="Experiment Comparison",
            description=f"Comparison: {label_a} vs {label_b}",
            figure_type="matplotlib",
            metadata={
                "n_both": int(counts.get("both", 0)),
                "n_a_only": int(counts.get("a_only", 0)),
                "n_b_only": int(counts.get("b_only", 0)),
                "n_neither": int(counts.get("neither", 0)),
            }
        )

    # =========================================================================
    # Level 4b: Interaction Effect Visualization
    # =========================================================================

    def plot_interaction_effects(
        self,
        interaction_results: pd.DataFrame,
        male_results: pd.DataFrame,
        female_results: pd.DataFrame,
        p_threshold: float = 0.05,
        n_top: int = 20,
        figsize: tuple[float, float] = (14, 10),
        title: str = "Sex × Disease Interaction Effects",
    ) -> Figure:
        """
        Visualize interaction effects with sex-stratified context.

        Cognitive operation: "Which cliques show sex-dependent disease effects?"

        Layout:
        - Main panel: Interaction p-values vs sex-stratified effects
        - Shows which direction the interaction goes

        Args:
            interaction_results: DataFrame from interaction ROAST
            male_results: DataFrame from male-only ROAST
            female_results: DataFrame from female-only ROAST
            p_threshold: Significance threshold for interaction
            n_top: Number of top interaction cliques to highlight
            figsize: Figure dimensions
            title: Plot title

        Returns:
            Figure wrapper with matplotlib figure
        """
        # Merge all results
        merged = pd.merge(
            interaction_results[["feature_set_id", "pvalue_msq_mixed", "observed_mean"]],
            male_results[["feature_set_id", "pvalue_msq_mixed"]].rename(
                columns={"pvalue_msq_mixed": "p_male"}
            ),
            on="feature_set_id",
            how="inner"
        )
        merged = pd.merge(
            merged,
            female_results[["feature_set_id", "pvalue_msq_mixed"]].rename(
                columns={"pvalue_msq_mixed": "p_female"}
            ),
            on="feature_set_id",
            how="inner"
        )

        # Transform p-values
        merged["neglog_interaction"] = -np.log10(merged["pvalue_msq_mixed"].clip(lower=1e-10))
        merged["neglog_male"] = -np.log10(merged["p_male"].clip(lower=1e-10))
        merged["neglog_female"] = -np.log10(merged["p_female"].clip(lower=1e-10))

        # Compute sex difference (male - female effect)
        merged["sex_diff"] = merged["neglog_male"] - merged["neglog_female"]

        fig, ax = plt.subplots(figsize=figsize)

        # Significant interactions
        sig_mask = merged["pvalue_msq_mixed"] < p_threshold

        # Color by sex-specificity
        colors = np.where(
            merged["sex_diff"] > 0,
            DIFFERENTIAL_COLORS["male_only"],
            DIFFERENTIAL_COLORS["female_only"]
        )
        colors = np.where(~sig_mask, DIFFERENTIAL_COLORS["neutral"], colors)

        # Size by interaction significance
        sizes = 20 + merged["neglog_interaction"] * 5
        sizes = sizes.clip(upper=200)

        scatter = ax.scatter(
            merged["sex_diff"],
            merged["neglog_interaction"],
            c=colors,
            s=sizes,
            alpha=0.6,
            edgecolors="white",
            linewidths=0.5,
        )

        # Reference lines
        ax.axhline(-np.log10(p_threshold), color="#94a3b8", linestyle="--",
                  linewidth=1, alpha=0.7)
        ax.axvline(0, color="#94a3b8", linestyle=":", linewidth=1, alpha=0.5)

        # Label top interactions using adjustText
        top_interactions = merged.nlargest(n_top, "neglog_interaction")
        texts = []
        for _, row in top_interactions.iterrows():
            txt = ax.text(
                row["sex_diff"], row["neglog_interaction"],
                italicize_gene(row["feature_set_id"]),
                fontsize=self.font_sizes["annotation"],
                fontweight="bold",
                color="#1e293b",
            )
            texts.append(txt)

        # Adjust text positions to avoid overlaps
        if texts:
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.5),
                expand_points=(1.5, 1.5),
                force_points=(0.5, 0.5),
                force_text=(0.5, 0.5),
            )

        # Axis labels
        ax.set_xlabel("Sex specificity\n(←Female-stronger | Male-stronger→)",
                     fontsize=self.font_sizes["label"])
        ax.set_ylabel(r"-log$_{10}$(interaction p-value)", fontsize=self.font_sizes["label"])
        ax.set_title(title, fontsize=self.font_sizes["title"], fontweight="bold")
        ax.tick_params(labelsize=self.font_sizes["tick"])

        # Legend
        legend_elements = [
            mpatches.Patch(color=DIFFERENTIAL_COLORS["male_only"], label="Male-dominant effect"),
            mpatches.Patch(color=DIFFERENTIAL_COLORS["female_only"], label="Female-dominant effect"),
            mpatches.Patch(color=DIFFERENTIAL_COLORS["neutral"], label="Not significant"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=self.font_sizes["annotation"])

        # Clean up
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        n_sig = sig_mask.sum()

        return Figure(
            fig=fig,
            title="Interaction Effects",
            description="Sex × Disease interaction visualization",
            figure_type="matplotlib",
            metadata={"n_significant_interactions": int(n_sig)}
        )

    # =========================================================================
    # Batch Report Generation
    # =========================================================================

    def generate_report(
        self,
        results: pd.DataFrame,
        output_dir: Path,
        experiment_name: str = "Differential Clique Analysis",
        male_results: pd.DataFrame | None = None,
        female_results: pd.DataFrame | None = None,
        interaction_results: pd.DataFrame | None = None,
    ) -> FigureCollection:
        """
        Generate a complete report with all visualization levels.

        Args:
            results: Primary ROAST results DataFrame
            output_dir: Directory to save figures
            experiment_name: Name for the analysis
            male_results: Optional male-stratified results
            female_results: Optional female-stratified results
            interaction_results: Optional interaction results

        Returns:
            FigureCollection with all generated figures
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        collection = FigureCollection()

        # Level 0: Overview
        fig = self.plot_significance_overview(results, title=experiment_name)
        fig.save(output_dir / "01_overview.png", dpi=150)
        collection.add(fig)

        # Level 1: Ranked plot (volcano)
        fig = self.plot_ranked_cliques(results, style="volcano", title=f"{experiment_name} - Volcano")
        fig.save(output_dir / "02_volcano.png", dpi=150)
        collection.add(fig)

        # Level 1b: Ranked plot (manhattan)
        fig = self.plot_ranked_cliques(results, style="manhattan", title=f"{experiment_name} - Manhattan")
        fig.save(output_dir / "03_manhattan.png", dpi=150)
        collection.add(fig)

        # Level 2: Direction composition
        fig = self.plot_direction_composition(results, title=f"{experiment_name} - Direction")
        fig.save(output_dir / "04_direction.png", dpi=150)
        collection.add(fig)

        # Level 4: Cross-experiment comparison (if available)
        if male_results is not None and female_results is not None:
            fig = self.plot_experiment_comparison(
                male_results, female_results,
                label_a="Male", label_b="Female",
                title="Male vs Female Comparison"
            )
            fig.save(output_dir / "05_male_vs_female.png", dpi=150)
            collection.add(fig)

        # Interaction visualization (if available)
        if interaction_results is not None and male_results is not None and female_results is not None:
            fig = self.plot_interaction_effects(
                interaction_results, male_results, female_results,
                title="Sex × Disease Interaction"
            )
            fig.save(output_dir / "06_interaction.png", dpi=150)
            collection.add(fig)

        return collection
