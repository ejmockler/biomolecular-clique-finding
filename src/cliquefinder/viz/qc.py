"""
Quality control visualizations for expression data.

Design Philosophy (Perceptual Engineering):
    These visualizations answer narrative questions about data transformations,
    NOT "what happened to sample X". With hundreds of samples, individual-level
    views are noise. We show DISTRIBUTIONS and PATTERNS that reveal:

    - What intervention did we make?
    - Did we preserve biological structure?
    - Where is uncertainty concentrated?
    - What's the story of this transformation?

Outlier Imputation Narrative:
    "We detected extreme values and pulled them back to reasonable bounds,
    preserving the biological difference between groups."

Sex Classification Narrative:
    "Biological sex creates measurable expression differences; we leveraged
    these to classify samples with unknown sex labels."
"""

from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag
from cliquefinder.viz.core import Figure
from cliquefinder.viz.styles import Palette, PALETTES, configure_style, get_stratum_colors
from cliquefinder.viz.id_mapper import get_gene_symbol, format_feature_label


def fmt_num(n: float) -> str:
    """Format number with scientific notation for large values."""
    if abs(n) >= 1e6:
        return f'{n:.2e}'
    elif abs(n) >= 1000:
        return f'{n:,.0f}'
    else:
        return f'{n:.1f}'


class QCVisualizer:
    """
    Aggregate visualizations for QC narratives.

    All plots show distributions and patterns, never individual samples.
    Designed for datasets with hundreds of samples where sample-level
    inspection is meaningless noise.
    """

    def __init__(
        self,
        palette: str | Palette = "default",
        style: Literal["paper", "presentation", "notebook"] = "paper"
    ):
        if isinstance(palette, str):
            self.palette = PALETTES.get(palette, PALETTES["default"])
        else:
            self.palette = palette
        self.style = style
        configure_style(style=style, palette=self.palette)

    # =========================================================================
    # OUTLIER IMPUTATION NARRATIVE
    # =========================================================================

    def plot_intervention_magnitude(
        self,
        original_values: np.ndarray,
        imputed_values: np.ndarray,
        outlier_mask: np.ndarray,
        strata: Optional[np.ndarray] = None,
        figsize: tuple[float, float] = (12, 8)
    ) -> Figure:
        """
        Show the winsorization pattern: extreme values pulled toward bounds.

        This is a 2-panel figure:
        - Left: Original vs Imputed (shows where values land after correction)
        - Right: Histogram of change magnitudes

        Key insight: Points above the diagonal were pulled DOWN, below were pulled UP.
        The further from diagonal, the more extreme the original outlier was.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Extract only outlier values
        orig = original_values[outlier_mask]
        imputed = imputed_values[outlier_mask]
        delta = imputed - orig

        n_positive = (delta > 0).sum()
        n_negative = (delta < 0).sum()

        # === Left panel: Original vs Imputed ===
        ax1 = axes[0]

        # Color by direction of change
        pulled_down_mask = delta < 0
        pulled_up_mask = delta > 0

        ax1.scatter(orig[pulled_down_mask], imputed[pulled_down_mask],
                   alpha=0.4, s=15, c=self.palette.case, label=f'Pulled down (n={n_negative:,})',
                   edgecolors='none')
        ax1.scatter(orig[pulled_up_mask], imputed[pulled_up_mask],
                   alpha=0.4, s=15, c=self.palette.ctrl, label=f'Pulled up (n={n_positive:,})',
                   edgecolors='none')

        # Identity line (no change)
        all_vals = np.concatenate([orig, imputed])
        lims = [all_vals.min(), all_vals.max()]
        ax1.plot(lims, lims, 'k-', linewidth=2, alpha=0.7, label='No change line')

        ax1.set_xlabel("Original Value (outlier)")
        ax1.set_ylabel("Imputed Value (corrected)")
        ax1.set_title(f"Winsorization: {len(orig):,} outliers corrected")
        ax1.legend(loc='upper left', fontsize=8)

        # Use log scale if data spans multiple orders of magnitude
        if all_vals.max() / (all_vals.min() + 1) > 100:
            ax1.set_xscale('log')
            ax1.set_yscale('log')

        # === Right panel: Magnitude of changes ===
        ax2 = axes[1]

        # Histogram of absolute changes
        abs_delta = np.abs(delta)

        ax2.hist(abs_delta, bins=50, color=self.palette.neutral, edgecolor='white', alpha=0.8)
        ax2.axvline(np.median(abs_delta), color='black', linestyle='-', linewidth=2,
                   label=f'Median: {fmt_num(np.median(abs_delta))}')
        ax2.axvline(np.percentile(abs_delta, 95), color=self.palette.outlier, linestyle='--', linewidth=2,
                   label=f'95th pctl: {fmt_num(np.percentile(abs_delta, 95))}')

        ax2.set_xlabel("|Δ| = |Imputed − Original|")
        ax2.set_ylabel("Count")
        ax2.set_title("Magnitude of Corrections")
        ax2.legend(loc='upper right', fontsize=8)

        # Summary annotation
        ax2.text(0.98, 0.75, f"Total corrections: {len(orig):,}\n"
                            f"Pulled down: {n_negative:,} ({100*n_negative/len(orig):.0f}%)\n"
                            f"Pulled up: {n_positive:,} ({100*n_positive/len(orig):.0f}%)",
                transform=ax2.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Intervention Magnitude",
            description=f"{len(orig):,} outliers: {n_negative:,} pulled down, {n_positive:,} pulled up",
            figure_type="matplotlib"
        )

    def plot_distribution_preservation(
        self,
        matrix_before: BioMatrix,
        matrix_after: BioMatrix,
        figsize: tuple[float, float] = (14, 8)
    ) -> Figure:
        """
        Show WHERE and HOW MUCH values changed during imputation.

        Since 99% of values are unchanged, overlaying before/after KDEs is useless.
        Instead, we show:
        - Left: Histogram of changes (Δ = after - before) for modified values only
        - Right: Scatter of original vs imputed for modified values, colored by stratum

        Reveals: The magnitude and direction of corrections, and whether strata differ.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Find changed values
        changed_mask = matrix_before.data != matrix_after.data
        n_changed = changed_mask.sum()
        pct_changed = 100 * n_changed / matrix_before.data.size

        before_changed = matrix_before.data[changed_mask]
        after_changed = matrix_after.data[changed_mask]
        delta = after_changed - before_changed

        # === Left panel: Distribution of changes ===
        ax1 = axes[0]

        # Separate positive and negative changes
        pulled_down = delta[delta < 0]
        pulled_up = delta[delta > 0]

        # Use percentile-based limits to focus on bulk of distribution
        lo, hi = np.percentile(delta, [1, 99])
        bound = max(abs(lo), abs(hi))

        # Create bins within the visible range
        bins = np.linspace(-bound, bound, 51)

        # Count outliers beyond display range
        n_clipped = ((delta < -bound) | (delta > bound)).sum()

        if len(pulled_down) > 0:
            ax1.hist(np.clip(pulled_down, -bound, bound), bins=bins, alpha=0.7,
                    color=self.palette.case,
                    label=f'Pulled down (n={len(pulled_down):,})', edgecolor='white')
        if len(pulled_up) > 0:
            ax1.hist(np.clip(pulled_up, -bound, bound), bins=bins, alpha=0.7,
                    color=self.palette.ctrl,
                    label=f'Pulled up (n={len(pulled_up):,})', edgecolor='white')

        ax1.axvline(0, color='black', linestyle='-', linewidth=2)
        ax1.set_xlim(-bound, bound)
        ax1.set_xlabel("Change (Δ = Imputed − Original)")
        ax1.set_ylabel("Count")
        ax1.set_title(f"Distribution of {n_changed:,} Changes ({pct_changed:.2f}%)")
        ax1.legend(loc='upper right')

        # Add summary stats
        median_down = np.median(pulled_down) if len(pulled_down) > 0 else 0
        median_up = np.median(pulled_up) if len(pulled_up) > 0 else 0
        stats_text = f"Median: {fmt_num(median_down)} / +{fmt_num(median_up)}"
        if n_clipped > 0:
            stats_text += f"\n({n_clipped:,} extreme values clipped)"
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # === Right panel: Original vs Imputed scatter ===
        ax2 = axes[1]

        # Color by stratum if metadata available
        if matrix_before.sample_metadata is not None and "phenotype" in matrix_before.sample_metadata.columns:
            phenotype = matrix_before.sample_metadata["phenotype"].values

            # Get stratum for each changed value
            row_idx, col_idx = np.where(changed_mask)
            strata = phenotype[col_idx]

            for pheno in ["CASE", "CTRL"]:
                mask = strata == pheno
                if mask.any():
                    color = self.palette.phenotype.get(pheno, self.palette.neutral)
                    ax2.scatter(before_changed[mask], after_changed[mask],
                               alpha=0.3, s=10, c=color, label=pheno, edgecolors='none')
        else:
            ax2.scatter(before_changed, after_changed, alpha=0.3, s=10,
                       c=self.palette.neutral, edgecolors='none')

        # Add identity line
        lims = [min(before_changed.min(), after_changed.min()),
                max(before_changed.max(), after_changed.max())]
        ax2.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='No change')

        ax2.set_xlabel("Original Value")
        ax2.set_ylabel("Imputed Value")
        ax2.set_title("Original vs Imputed (changed values only)")
        ax2.legend(loc='upper left', fontsize=9)

        # Use same scale for both axes
        ax2.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Imputation Changes",
            description=f"{n_changed:,} values ({pct_changed:.2f}%) modified",
            figure_type="matplotlib"
        )

    def plot_protein_vulnerability(
        self,
        outlier_mask: np.ndarray,
        feature_ids: np.ndarray,
        n_top: int = 20,
        figsize: tuple[float, float] = (10, 6)
    ) -> Figure:
        """
        Which proteins are outlier-prone?

        Histogram: % of samples flagged as outlier per protein
        Annotation: Most vulnerable proteins labeled

        Reveals: Most proteins have few outliers; some are systematically problematic.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})

        # Compute outlier % per protein
        n_samples = outlier_mask.shape[1]
        outlier_pct = 100 * outlier_mask.sum(axis=1) / n_samples

        # Left panel: Histogram
        ax1 = axes[0]
        ax1.hist(outlier_pct, bins=50, color=self.palette.case, edgecolor='white',
                 linewidth=0.5, alpha=0.8)

        # Annotations
        median_pct = np.median(outlier_pct)
        p95 = np.percentile(outlier_pct, 95)

        ax1.axvline(median_pct, color='black', linestyle='-',
                   linewidth=2, label=f'Median: {median_pct:.1f}%')
        ax1.axvline(p95, color=self.palette.outlier, linestyle='--',
                   linewidth=2, label=f'95th pctl: {p95:.1f}%')

        ax1.set_xlabel("Outlier Rate (% of samples)")
        ax1.set_ylabel("Number of Proteins")
        ax1.set_title("Outlier Prevalence Distribution")
        ax1.legend(loc='upper right')

        # Right panel: Top vulnerable proteins
        ax2 = axes[1]
        top_idx = np.argsort(outlier_pct)[-n_top:][::-1]
        top_proteins = feature_ids[top_idx]
        top_pcts = outlier_pct[top_idx]

        # Convert to gene symbols for readability
        top_gene_symbols = [get_gene_symbol(p) for p in top_proteins]

        y_pos = np.arange(len(top_proteins))
        ax2.barh(y_pos, top_pcts, color=self.palette.outlier, edgecolor='white')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_gene_symbols, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel("Outlier Rate (%)")
        ax2.set_title(f"Top {n_top} Vulnerable Proteins")

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Protein Vulnerability",
            description=f"Median outlier rate: {median_pct:.1f}%, {(outlier_pct > 5).sum()} proteins above 5%",
            figure_type="matplotlib"
        )

    def plot_outlier_distribution_by_stratum(
        self,
        matrix_before: BioMatrix,
        matrix_after: BioMatrix,
        figsize: tuple[float, float] = (14, 10)
    ) -> Figure:
        """
        Show where outliers occur AND where they were winsorized to.

        Data-driven: Uses actual outlier flags from MAD-z detection.
        Shows original distribution with outlier original values (orange)
        and vertical lines showing the actual winsorization bounds
        (min/max of imputed values for outliers).
        """
        if matrix_before.sample_metadata is None:
            raise ValueError("sample_metadata required")

        metadata = matrix_before.sample_metadata
        phenotype = metadata.get("phenotype", pd.Series(["ALL"] * matrix_before.n_samples))

        sex_col = None
        for col in ["Sex_predicted", "SEX"]:
            if col in metadata.columns:
                sex_col = col
                break

        if sex_col:
            sex = metadata[sex_col].fillna("?")
            strata = phenotype.astype(str) + "_" + sex.astype(str)
        else:
            strata = phenotype.astype(str)

        unique_strata = sorted(strata.unique())
        n_strata = len(unique_strata)

        # Get actual outlier mask from data comparison
        outlier_mask = matrix_before.data != matrix_after.data

        # Create grid layout
        n_cols = min(2, n_strata)
        n_rows = (n_strata + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        colors = get_stratum_colors(unique_strata, self.palette)

        for i, stratum in enumerate(unique_strata):
            ax = axes[i]
            sample_mask = (strata == stratum).values

            # Get values for this stratum (both before and after)
            stratum_before = matrix_before.data[:, sample_mask].flatten()
            stratum_after = matrix_after.data[:, sample_mask].flatten()
            stratum_outlier_mask = outlier_mask[:, sample_mask].flatten()

            # Filter to positive values for log scale
            all_pos = stratum_before[stratum_before > 0]
            if len(all_pos) == 0:
                continue

            # Get imputed values for outliers to find actual bounds
            outlier_imputed = stratum_after[stratum_outlier_mask]
            outlier_imputed_pos = outlier_imputed[outlier_imputed > 0]

            # Count statistics
            n_total = len(stratum_before)
            n_outliers = stratum_outlier_mask.sum()
            pct_outliers = 100 * n_outliers / n_total if n_total > 0 else 0

            # Data range
            data_min, data_max = all_pos.min(), all_pos.max()

            # Log-spaced bins spanning the data range
            bins = np.logspace(np.log10(data_min), np.log10(data_max), 101)

            # Single histogram of entire original distribution
            color = colors[stratum]
            ax.hist(all_pos, bins=bins, color=color, alpha=0.7,
                   edgecolor='white', linewidth=0.3)

            # Show actual winsorization bounds as vertical lines
            # These are the min/max values that outliers were clipped TO
            if len(outlier_imputed_pos) > 0:
                lower_bound = outlier_imputed_pos.min()
                upper_bound = outlier_imputed_pos.max()

                ax.axvline(lower_bound, color='black', linestyle='--', linewidth=2)
                ax.axvline(upper_bound, color='black', linestyle='-', linewidth=2)

                # Shade regions outside bounds (what got clipped)
                ax.axvspan(data_min * 0.9, lower_bound, alpha=0.3, color=self.palette.outlier)
                ax.axvspan(upper_bound, data_max * 1.1, alpha=0.3, color=self.palette.outlier)

                # Count what's in each tail
                n_below = (all_pos < lower_bound).sum()
                n_above = (all_pos > upper_bound).sum()

                # All bound info in annotation box
                ax.text(0.02, 0.98,
                       f"Lower: {fmt_num(lower_bound)} ({n_below:,} clipped)\n"
                       f"Upper: {fmt_num(upper_bound)} ({n_above:,} clipped)",
                       transform=ax.transAxes, va='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(data_min * 0.9, data_max * 1.1)
            ax.set_xlabel("Expression Value (log)")
            ax.set_ylabel("Count (log)")
            ax.set_title(f"{stratum} (n={sample_mask.sum():,} samples, {pct_outliers:.1f}% clipped)")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        total_outliers = outlier_mask.sum()
        total_values = matrix_before.data.size
        overall_pct = 100 * total_outliers / total_values

        fig.suptitle(f"Outlier Distribution with Winsorization Bounds\n"
                    f"({total_outliers:,} outliers = {overall_pct:.2f}%, bounds from actual imputed values)",
                    fontweight='bold', fontsize=12)

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Outlier Distribution by Stratum",
            description=f"{total_outliers:,} outliers ({overall_pct:.2f}%) with actual winsorization bounds",
            figure_type="matplotlib"
        )

    def plot_outlier_summary_card(
        self,
        matrix_before: BioMatrix,
        matrix_after: BioMatrix,
        figsize: tuple[float, float] = (12, 8)
    ) -> Figure:
        """
        Single-figure summary of outlier imputation narrative.

        4-panel layout:
        - Top-left: Intervention magnitude (2D density)
        - Top-right: Protein vulnerability histogram
        - Bottom: Distribution preservation by stratum
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

        # Get outlier mask
        if matrix_after.quality_flags is not None:
            outlier_mask = (matrix_after.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        else:
            # Fallback: compute from difference
            outlier_mask = matrix_before.data != matrix_after.data

        n_outliers = outlier_mask.sum()
        total = matrix_before.data.size
        pct = 100 * n_outliers / total

        # === Panel 1: Intervention magnitude (Original vs Imputed scatter) ===
        ax1 = fig.add_subplot(gs[0, 0])

        orig = matrix_before.data[outlier_mask]
        imputed = matrix_after.data[outlier_mask]
        delta = imputed - orig

        if len(orig) > 0:
            # Color by direction of change
            pulled_down = delta < 0
            pulled_up = delta > 0

            ax1.scatter(orig[pulled_down], imputed[pulled_down],
                       alpha=0.4, s=12, c=self.palette.case,
                       label=f'Down ({pulled_down.sum():,})', edgecolors='none')
            ax1.scatter(orig[pulled_up], imputed[pulled_up],
                       alpha=0.4, s=12, c=self.palette.ctrl,
                       label=f'Up ({pulled_up.sum():,})', edgecolors='none')

            # Identity line (no change)
            all_vals = np.concatenate([orig, imputed])
            lims = [all_vals.min(), all_vals.max()]
            ax1.plot(lims, lims, 'k-', linewidth=1.5, alpha=0.7)

            # Use log scale if data spans multiple orders of magnitude
            if all_vals.max() / (all_vals.min() + 1) > 100:
                ax1.set_xscale('log')
                ax1.set_yscale('log')

        ax1.set_xlabel("Original")
        ax1.set_ylabel("Imputed")
        ax1.set_title("Winsorization Pattern")
        ax1.legend(loc='upper left', fontsize=7)

        # === Panel 2: Protein vulnerability ===
        ax2 = fig.add_subplot(gs[0, 1])

        outlier_pct = 100 * outlier_mask.sum(axis=1) / matrix_before.n_samples
        ax2.hist(outlier_pct, bins=40, color=self.palette.case, edgecolor='white', alpha=0.8)
        ax2.axvline(np.median(outlier_pct), color='black', linestyle='-', linewidth=2)
        ax2.set_xlabel("Outlier Rate (% samples)")
        ax2.set_ylabel("# Proteins")
        ax2.set_title("Protein Vulnerability")

        # === Panel 3: Distribution of changes (histogram of Δ values) ===
        ax3 = fig.add_subplot(gs[1, :])

        # Show histogram of changes, which is more informative than overlapping KDEs
        pulled_down = delta[delta < 0]
        pulled_up = delta[delta > 0]

        if len(delta) > 0:
            # Use percentile-based limits to focus on bulk of distribution
            lo, hi = np.percentile(delta, [1, 99])
            # Ensure we include 0 and have symmetric-ish bounds
            bound = max(abs(lo), abs(hi))

            # Create bins within the visible range
            bins = np.linspace(-bound, bound, 51)

            # Count outliers beyond display range
            n_below = (delta < -bound).sum()
            n_above = (delta > bound).sum()

            if len(pulled_down) > 0:
                ax3.hist(np.clip(pulled_down, -bound, bound), bins=bins, alpha=0.7,
                        color=self.palette.case,
                        label=f'Pulled down (n={len(pulled_down):,})', edgecolor='white')
            if len(pulled_up) > 0:
                ax3.hist(np.clip(pulled_up, -bound, bound), bins=bins, alpha=0.7,
                        color=self.palette.ctrl,
                        label=f'Pulled up (n={len(pulled_up):,})', edgecolor='white')

            ax3.axvline(0, color='black', linestyle='-', linewidth=2)
            ax3.set_xlim(-bound, bound)

            # Add summary stats
            median_down = np.median(pulled_down) if len(pulled_down) > 0 else 0
            median_up = np.median(pulled_up) if len(pulled_up) > 0 else 0

            stats_text = f"Median: {fmt_num(median_down)} / +{fmt_num(median_up)}"
            if n_below + n_above > 0:
                stats_text += f"\n({n_below + n_above:,} extreme values clipped)"

            ax3.text(0.02, 0.98, stats_text,
                    transform=ax3.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax3.set_xlabel("Change (Δ = Imputed − Original)")
        ax3.set_ylabel("Count")
        ax3.set_title("Distribution of Corrections")
        ax3.legend(loc='upper right', fontsize=8)

        fig.suptitle(f"Outlier Imputation Summary: {n_outliers:,} values ({pct:.2f}%) adjusted",
                    fontweight='bold', fontsize=12)

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Outlier Imputation Summary",
            description=f"{n_outliers:,} outliers ({pct:.2f}%) winsorized",
            figure_type="matplotlib"
        )

    # =========================================================================
    # ADAPTIVE OUTLIER HANDLING NARRATIVES
    # =========================================================================

    def plot_skewness_adaptation(
        self,
        medcouples: np.ndarray,
        lower_fences: np.ndarray,
        upper_fences: np.ndarray,
        feature_ids: np.ndarray,
        figsize: tuple[float, float] = (14, 8)
    ) -> Figure:
        """
        Visualize medcouple-based asymmetric fence adaptation.

        Narrative: "We detected distribution asymmetry and adapted outlier bounds
        accordingly—right-skewed features get wider upper fences, left-skewed
        features get wider lower fences."

        Cognitive Task: Understand WHY different proteins have different fence widths.

        Perceptual Channels:
        - X-position: Medcouple (skewness measure)
        - Y-position: Fence asymmetry ratio
        - Color: Skewness direction (blue=left, red=right, gray=symmetric)
        - Vertical position: Shows how bounds expand/contract with skewness

        Parameters
        ----------
        medcouples : np.ndarray
            Medcouple values per feature (range [-1, 1])
        lower_fences : np.ndarray
            Lower outlier fence per feature
        upper_fences : np.ndarray
            Upper outlier fence per feature
        feature_ids : np.ndarray
            Feature identifiers for labeling
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # === Panel 1: Medcouple distribution (skewness landscape) ===
        ax1 = axes[0]

        # Color by skewness direction
        colors = np.where(
            medcouples > 0.1, self.palette.ctrl,  # Right-skewed (orange)
            np.where(medcouples < -0.1, self.palette.case, self.palette.neutral)  # Left-skewed (blue) / symmetric (gray)
        )

        ax1.hist(medcouples, bins=50, color=self.palette.neutral, edgecolor='white', alpha=0.7)

        # Shade skewness regions
        ax1.axvspan(-1, -0.1, alpha=0.15, color=self.palette.case, label='Left-skewed')
        ax1.axvspan(-0.1, 0.1, alpha=0.15, color=self.palette.neutral, label='Symmetric')
        ax1.axvspan(0.1, 1, alpha=0.15, color=self.palette.ctrl, label='Right-skewed')

        ax1.axvline(0, color='black', linestyle='-', linewidth=2)

        # Summary stats
        n_left = (medcouples < -0.1).sum()
        n_sym = ((medcouples >= -0.1) & (medcouples <= 0.1)).sum()
        n_right = (medcouples > 0.1).sum()

        ax1.text(0.02, 0.98,
                f"Left-skewed: {n_left:,} ({100*n_left/len(medcouples):.0f}%)\n"
                f"Symmetric: {n_sym:,} ({100*n_sym/len(medcouples):.0f}%)\n"
                f"Right-skewed: {n_right:,} ({100*n_right/len(medcouples):.0f}%)",
                transform=ax1.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax1.set_xlabel("Medcouple (MC)")
        ax1.set_ylabel("Number of Features")
        ax1.set_title("Distribution Skewness Landscape")
        ax1.set_xlim(-1, 1)
        ax1.legend(loc='upper right', fontsize=8)

        # === Panel 2: Fence asymmetry vs medcouple (the adaptation story) ===
        ax2 = axes[1]

        # Compute fence asymmetry: ratio of upper to lower fence width
        # Higher ratio = wider upper fence relative to lower
        fence_ratio = np.log2((upper_fences + 1) / (lower_fences + 1 + 1e-10))

        # Scatter with medcouple coloring
        scatter = ax2.scatter(medcouples, fence_ratio, c=medcouples, cmap='RdBu_r',
                             alpha=0.6, s=20, edgecolors='none', vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
        cbar.set_label('Medcouple')

        # Trend line (theoretical)
        mc_range = np.linspace(-0.8, 0.8, 100)
        # Hubert-Vandervieren: right-skew (MC>0) → expand upper fence
        expected_ratio = mc_range * 2  # Simplified visualization of adaptation
        ax2.plot(mc_range, expected_ratio, 'k--', linewidth=2, alpha=0.7, label='Expected adaptation')

        ax2.axhline(0, color='gray', linestyle='-', linewidth=1)
        ax2.axvline(0, color='gray', linestyle='-', linewidth=1)

        ax2.set_xlabel("Medcouple (Skewness)")
        ax2.set_ylabel("log₂(Upper/Lower Fence Ratio)")
        ax2.set_title("Fence Adaptation to Skewness")
        ax2.legend(loc='upper left', fontsize=8)

        # === Panel 3: Top asymmetric features (who needed adaptation?) ===
        ax3 = axes[2]

        # Find most skewed features
        skewness_magnitude = np.abs(medcouples)
        top_idx = np.argsort(skewness_magnitude)[-15:][::-1]

        top_features = feature_ids[top_idx]
        top_mc = medcouples[top_idx]

        # Convert to gene symbols
        top_gene_symbols = [get_gene_symbol(f) for f in top_features]

        y_pos = np.arange(len(top_features))
        colors_bar = [self.palette.ctrl if mc > 0 else self.palette.case for mc in top_mc]

        ax3.barh(y_pos, top_mc, color=colors_bar, edgecolor='white')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(top_gene_symbols, fontsize=8)
        ax3.invert_yaxis()
        ax3.axvline(0, color='black', linestyle='-', linewidth=2)
        ax3.set_xlabel("Medcouple")
        ax3.set_title("Most Asymmetric Features")

        # Add skewness direction labels
        ax3.text(0.6, 1.02, "Right-skewed →", transform=ax3.transAxes, fontsize=8,
                color=self.palette.ctrl, ha='center')
        ax3.text(-0.1, 1.02, "← Left-skewed", transform=ax3.transAxes, fontsize=8,
                color=self.palette.case, ha='center')

        fig.suptitle("Medcouple-Adjusted Outlier Detection:\nAdapting to Distribution Shape",
                    fontweight='bold', fontsize=12)
        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Skewness Adaptation",
            description=f"{n_left:,} left-skewed, {n_sym:,} symmetric, {n_right:,} right-skewed features",
            figure_type="matplotlib"
        )

    def plot_probabilistic_scores(
        self,
        outlier_pvalues: np.ndarray,
        outlier_mask: np.ndarray,
        degrees_of_freedom: float,
        figsize: tuple[float, float] = (14, 8)
    ) -> Figure:
        """
        Visualize Student's t probabilistic outlier scores.

        Narrative: "Instead of binary outlier flags, we assigned probability
        scores using heavy-tailed Student's t distributions. This captures
        uncertainty—borderline values get intermediate scores."

        Cognitive Task: See the gradient from confident to uncertain outliers.

        Perceptual Channels:
        - X-position: p-value (probability of being this extreme)
        - Color gradient: Outlier confidence (red=confident, yellow=uncertain)
        - Histogram shape: Shows concentration of confidence levels

        Parameters
        ----------
        outlier_pvalues : np.ndarray
            P-values from Student's t distribution (shape: n_features × n_samples)
        outlier_mask : np.ndarray
            Binary outlier mask for comparison
        degrees_of_freedom : float
            Fitted df parameter (lower = heavier tails)
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Flatten arrays for overall analysis
        pvals_flat = outlier_pvalues.flatten()
        mask_flat = outlier_mask.flatten()

        # === Panel 1: P-value distribution (where is confidence?) ===
        ax1 = axes[0]

        # Log-transform p-values for better visualization
        log_pvals = -np.log10(pvals_flat + 1e-300)  # Avoid log(0)

        # Histogram with gradient coloring based on significance
        bins = np.linspace(0, min(50, log_pvals.max()), 100)
        n, bins_edges, patches = ax1.hist(log_pvals, bins=bins, edgecolor='white', linewidth=0.3)

        # Color patches by significance level
        cmap = plt.cm.YlOrRd
        for i, patch in enumerate(patches):
            bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
            # Normalize to [0, 1] for colormap
            intensity = min(1.0, bin_center / 10)  # Cap at -log10(p) = 10
            patch.set_facecolor(cmap(intensity))

        # Add significance thresholds
        ax1.axvline(-np.log10(0.05), color='black', linestyle='--', linewidth=2,
                   label='p = 0.05')
        ax1.axvline(-np.log10(0.01), color='black', linestyle=':', linewidth=2,
                   label='p = 0.01')
        ax1.axvline(-np.log10(0.001), color='black', linestyle='-.', linewidth=2,
                   label='p = 0.001')

        ax1.set_xlabel("-log₁₀(p-value)")
        ax1.set_ylabel("Count")
        ax1.set_title("Outlier Confidence Distribution")
        ax1.legend(loc='upper right', fontsize=8)

        # Summary stats
        n_p05 = (pvals_flat < 0.05).sum()
        n_p01 = (pvals_flat < 0.01).sum()
        n_p001 = (pvals_flat < 0.001).sum()
        ax1.text(0.98, 0.98,
                f"p < 0.05: {n_p05:,} ({100*n_p05/len(pvals_flat):.2f}%)\n"
                f"p < 0.01: {n_p01:,} ({100*n_p01/len(pvals_flat):.2f}%)\n"
                f"p < 0.001: {n_p001:,} ({100*n_p001/len(pvals_flat):.2f}%)",
                transform=ax1.transAxes, va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # === Panel 2: Binary vs probabilistic comparison ===
        ax2 = axes[1]

        # For values flagged as outliers by binary method, show their p-value distribution
        binary_outlier_pvals = pvals_flat[mask_flat]
        binary_inlier_pvals = pvals_flat[~mask_flat]

        if len(binary_outlier_pvals) > 0:
            ax2.hist(-np.log10(binary_outlier_pvals + 1e-300), bins=50, alpha=0.7,
                    color=self.palette.outlier, label=f'Binary outliers (n={len(binary_outlier_pvals):,})',
                    edgecolor='white', density=True)

        if len(binary_inlier_pvals) > 0:
            ax2.hist(-np.log10(binary_inlier_pvals + 1e-300), bins=50, alpha=0.5,
                    color=self.palette.neutral, label=f'Binary inliers (n={len(binary_inlier_pvals):,})',
                    edgecolor='white', density=True)

        ax2.set_xlabel("-log₁₀(p-value)")
        ax2.set_ylabel("Density")
        ax2.set_title("Binary vs Probabilistic: Outlier Confidence")
        ax2.legend(loc='upper right', fontsize=8)

        # === Panel 3: Degrees of freedom impact (tail heaviness) ===
        ax3 = axes[2]

        # Show Student's t PDF with fitted df vs Normal
        x = np.linspace(-6, 6, 200)

        # Normal distribution
        normal_pdf = stats.norm.pdf(x)

        # Student's t with fitted df
        t_pdf = stats.t.pdf(x, df=degrees_of_freedom)

        # Reference: t with df=3 (heavy tails) and df=30 (near-normal)
        t3_pdf = stats.t.pdf(x, df=3)
        t30_pdf = stats.t.pdf(x, df=30)

        ax3.plot(x, normal_pdf, 'k-', linewidth=2, label='Normal', alpha=0.7)
        ax3.plot(x, t30_pdf, '--', color=self.palette.neutral, linewidth=1.5,
                label='t (df=30, light tails)', alpha=0.7)
        ax3.plot(x, t_pdf, '-', color=self.palette.highlight, linewidth=3,
                label=f't (df={degrees_of_freedom:.1f}, fitted)')
        ax3.plot(x, t3_pdf, '--', color=self.palette.outlier, linewidth=1.5,
                label='t (df=3, heavy tails)', alpha=0.7)

        # Shade tail regions
        tail_x = x[np.abs(x) > 2.5]
        ax3.fill_between(tail_x, 0, stats.t.pdf(tail_x, df=degrees_of_freedom),
                        alpha=0.3, color=self.palette.highlight, label='Tail mass')

        ax3.set_xlabel("Standardized Value (z)")
        ax3.set_ylabel("Probability Density")
        ax3.set_title(f"Tail Heaviness: Fitted df = {degrees_of_freedom:.1f}")
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_xlim(-6, 6)
        ax3.set_ylim(0, 0.45)

        # Interpretation
        if degrees_of_freedom < 10:
            tail_interpretation = "Heavy tails → tolerant of extremes"
        elif degrees_of_freedom < 30:
            tail_interpretation = "Moderate tails → balanced"
        else:
            tail_interpretation = "Light tails → strict outlier detection"

        ax3.text(0.5, 0.02, tail_interpretation, transform=ax3.transAxes,
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle("Student's t Probabilistic Outlier Scoring:\nCapturing Uncertainty in Extreme Values",
                    fontweight='bold', fontsize=12)
        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Probabilistic Outlier Scores",
            description=f"df={degrees_of_freedom:.1f}, {n_p001:,} values with p < 0.001",
            figure_type="matplotlib"
        )

    def plot_soft_clip_transformation(
        self,
        original_values: np.ndarray,
        clipped_values: np.ndarray,
        outlier_mask: np.ndarray,
        sharpness: float = 5.0,
        figsize: tuple[float, float] = (14, 8)
    ) -> Figure:
        """
        Visualize soft clipping transformation curve and rank preservation.

        Narrative: "We smoothly compressed extreme values using a sigmoid function,
        preserving rank order while reducing their influence. No information is
        completely lost—extremes are just pulled closer together."

        Cognitive Task: Understand the transformation curve, verify rank preservation.

        Perceptual Channels:
        - Curve shape: Shows smooth compression (not hard cutoff)
        - Before/after scatter: Shows transformation applied
        - Rank comparison: Verifies ordering preserved

        Parameters
        ----------
        original_values : np.ndarray
            Original expression values (flat or 2D)
        clipped_values : np.ndarray
            Soft-clipped values
        outlier_mask : np.ndarray
            Boolean mask of outlier positions
        sharpness : float
            Sharpness parameter used in tanh transformation
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Flatten if needed
        orig_flat = original_values.flatten()
        clip_flat = clipped_values.flatten()
        mask_flat = outlier_mask.flatten()

        # Focus on outliers only
        orig_out = orig_flat[mask_flat]
        clip_out = clip_flat[mask_flat]

        # === Panel 1: Transformation curve (the core insight) ===
        ax1 = axes[0]

        # Create a range spanning the data
        x_range = np.linspace(orig_flat.min(), orig_flat.max(), 500)

        # Compute bounds for soft clipping (approximate from data)
        non_outlier = orig_flat[~mask_flat]
        if len(non_outlier) > 0:
            lower_bound = np.percentile(non_outlier, 1)
            upper_bound = np.percentile(non_outlier, 99)
        else:
            lower_bound = np.percentile(orig_flat, 5)
            upper_bound = np.percentile(orig_flat, 95)

        # Theoretical soft clip curve
        center = (lower_bound + upper_bound) / 2
        half_width = (upper_bound - lower_bound) / 2

        def soft_clip_curve(x, c, hw, k):
            """Soft clipping using tanh."""
            normalized = (x - c) / hw
            clipped = np.tanh(normalized / k) * k
            return c + hw * clipped / k

        y_theoretical = soft_clip_curve(x_range, center, half_width, 1/sharpness)

        # Plot identity line (no transformation)
        ax1.plot(x_range, x_range, 'k--', linewidth=1, alpha=0.5, label='Identity (no change)')

        # Plot soft clip curve
        ax1.plot(x_range, y_theoretical, '-', color=self.palette.highlight, linewidth=3,
                label=f'Soft clip (k={sharpness:.1f})')

        # Mark the bounds
        ax1.axvline(lower_bound, color=self.palette.case, linestyle=':', linewidth=2)
        ax1.axvline(upper_bound, color=self.palette.case, linestyle=':', linewidth=2)
        ax1.axhline(lower_bound, color=self.palette.case, linestyle=':', linewidth=2, alpha=0.5)
        ax1.axhline(upper_bound, color=self.palette.case, linestyle=':', linewidth=2, alpha=0.5)

        # Shade compression regions
        ax1.fill_between(x_range, x_range, y_theoretical, where=x_range < lower_bound,
                        alpha=0.3, color=self.palette.ctrl, label='Expansion zone')
        ax1.fill_between(x_range, x_range, y_theoretical, where=x_range > upper_bound,
                        alpha=0.3, color=self.palette.case, label='Compression zone')

        ax1.set_xlabel("Original Value")
        ax1.set_ylabel("Soft-Clipped Value")
        ax1.set_title("Soft Clipping Transformation Curve")
        ax1.legend(loc='upper left', fontsize=8)

        # === Panel 2: Before vs After scatter for outliers ===
        ax2 = axes[1]

        if len(orig_out) > 0:
            # Color by direction of change
            delta = clip_out - orig_out
            pulled_down = delta < 0
            pulled_up = delta > 0
            unchanged = np.abs(delta) < 1e-6

            ax2.scatter(orig_out[pulled_down], clip_out[pulled_down],
                       alpha=0.5, s=20, c=self.palette.case,
                       label=f'Compressed ({pulled_down.sum():,})', edgecolors='none')
            ax2.scatter(orig_out[pulled_up], clip_out[pulled_up],
                       alpha=0.5, s=20, c=self.palette.ctrl,
                       label=f'Expanded ({pulled_up.sum():,})', edgecolors='none')

            # Identity line
            all_vals = np.concatenate([orig_out, clip_out])
            lims = [all_vals.min(), all_vals.max()]
            ax2.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)

            # Add bounds
            ax2.axvline(lower_bound, color='gray', linestyle=':', alpha=0.5)
            ax2.axvline(upper_bound, color='gray', linestyle=':', alpha=0.5)

        ax2.set_xlabel("Original Value")
        ax2.set_ylabel("Soft-Clipped Value")
        ax2.set_title(f"Outlier Values: Before vs After ({len(orig_out):,} values)")
        ax2.legend(loc='upper left', fontsize=8)

        # === Panel 3: Rank preservation verification ===
        ax3 = axes[2]

        if len(orig_out) > 100:
            # Subsample for visualization
            sample_idx = np.random.choice(len(orig_out), min(500, len(orig_out)), replace=False)
            orig_sample = orig_out[sample_idx]
            clip_sample = clip_out[sample_idx]
        else:
            orig_sample = orig_out
            clip_sample = clip_out

        # Compute ranks
        orig_ranks = stats.rankdata(orig_sample)
        clip_ranks = stats.rankdata(clip_sample)

        # Scatter ranks
        ax3.scatter(orig_ranks, clip_ranks, alpha=0.5, s=15,
                   c=self.palette.highlight, edgecolors='none')

        # Perfect preservation line
        max_rank = max(orig_ranks.max(), clip_ranks.max())
        ax3.plot([1, max_rank], [1, max_rank], 'k--', linewidth=2, alpha=0.7,
                label='Perfect preservation')

        # Compute Spearman correlation
        if len(orig_sample) > 2:
            spearman_r, _ = stats.spearmanr(orig_sample, clip_sample)
        else:
            spearman_r = 1.0

        ax3.set_xlabel("Original Rank")
        ax3.set_ylabel("Soft-Clipped Rank")
        ax3.set_title(f"Rank Preservation (Spearman ρ = {spearman_r:.4f})")
        ax3.legend(loc='upper left', fontsize=8)

        # Annotation about rank preservation
        if spearman_r > 0.999:
            rank_msg = "Perfect rank preservation"
            msg_color = self.palette.highlight
        elif spearman_r > 0.99:
            rank_msg = "Near-perfect rank preservation"
            msg_color = self.palette.highlight
        else:
            rank_msg = "Some rank changes (extreme compression)"
            msg_color = self.palette.outlier

        ax3.text(0.5, 0.02, rank_msg, transform=ax3.transAxes,
                ha='center', fontsize=10, color=msg_color, style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle("Soft Clipping: Smooth Compression Preserving Biological Order",
                    fontweight='bold', fontsize=12)
        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Soft Clipping Transformation",
            description=f"Spearman ρ = {spearman_r:.4f}, {len(orig_out):,} outliers transformed",
            figure_type="matplotlib"
        )

    def plot_weighted_contribution(
        self,
        values_x: np.ndarray,
        values_y: np.ndarray,
        weights: np.ndarray,
        feature_name_x: str = "Feature X",
        feature_name_y: str = "Feature Y",
        unweighted_corr: Optional[float] = None,
        weighted_corr: Optional[float] = None,
        figsize: tuple[float, float] = (14, 6)
    ) -> Figure:
        """
        Visualize weighted correlation contribution.

        Narrative: "Instead of removing outliers from correlation calculations,
        we downweighted their influence. Every data point contributes, but
        extreme values have reduced impact."

        Cognitive Task: See which values had reduced influence.

        Perceptual Channels:
        - Position: X-Y coordinates in correlation space
        - Size: Weight (larger = more influence)
        - Color: Weight intensity (darker = more influence)

        Parameters
        ----------
        values_x, values_y : np.ndarray
            Feature values for correlation pair
        weights : np.ndarray
            Sample weights (0-1, lower for outliers)
        feature_name_x, feature_name_y : str
            Feature labels for axes
        unweighted_corr, weighted_corr : float, optional
            Pre-computed correlations for annotation
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # === Panel 1: Weight distribution ===
        ax1 = axes[0]

        ax1.hist(weights, bins=50, color=self.palette.case, edgecolor='white', alpha=0.8)

        # Mark weight thresholds
        ax1.axvline(1.0, color='black', linestyle='-', linewidth=2, label='Full weight')
        ax1.axvline(0.5, color=self.palette.outlier, linestyle='--', linewidth=2,
                   label='Half weight')

        # Count by weight categories
        n_full = (weights > 0.95).sum()
        n_reduced = ((weights <= 0.95) & (weights > 0.5)).sum()
        n_low = (weights <= 0.5).sum()

        ax1.text(0.02, 0.98,
                f"Full weight (>0.95): {n_full:,} ({100*n_full/len(weights):.1f}%)\n"
                f"Reduced (0.5-0.95): {n_reduced:,} ({100*n_reduced/len(weights):.1f}%)\n"
                f"Low weight (≤0.5): {n_low:,} ({100*n_low/len(weights):.1f}%)",
                transform=ax1.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax1.set_xlabel("Weight")
        ax1.set_ylabel("Count")
        ax1.set_title("Sample Weight Distribution")
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_xlim(0, 1.05)

        # === Panel 2: Scatter with size-encoded weights ===
        ax2 = axes[1]

        # Normalize weights for size mapping
        size_min, size_max = 5, 100
        sizes = size_min + (size_max - size_min) * weights

        # Color by weight
        scatter = ax2.scatter(values_x, values_y, s=sizes, c=weights, cmap='viridis',
                             alpha=0.6, edgecolors='white', linewidths=0.3,
                             vmin=0, vmax=1)

        cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
        cbar.set_label('Weight')

        # Add regression lines
        if len(values_x) > 2:
            # Unweighted regression
            z_uw = np.polyfit(values_x, values_y, 1)
            p_uw = np.poly1d(z_uw)
            x_line = np.linspace(values_x.min(), values_x.max(), 100)
            ax2.plot(x_line, p_uw(x_line), '--', color=self.palette.neutral, linewidth=2,
                    label='Unweighted fit')

            # Weighted regression
            z_w = np.polyfit(values_x, values_y, 1, w=weights)
            p_w = np.poly1d(z_w)
            ax2.plot(x_line, p_w(x_line), '-', color=self.palette.highlight, linewidth=2,
                    label='Weighted fit')

        ax2.set_xlabel(get_gene_symbol(feature_name_x))
        ax2.set_ylabel(get_gene_symbol(feature_name_y))
        ax2.set_title("Correlation Space\n(size & color = weight)")
        ax2.legend(loc='best', fontsize=8)

        # === Panel 3: Correlation comparison ===
        ax3 = axes[2]

        # If correlations not provided, compute them
        if unweighted_corr is None and len(values_x) > 2:
            unweighted_corr, _ = stats.pearsonr(values_x, values_y)

        if weighted_corr is None and len(values_x) > 2:
            # Weighted Pearson correlation
            w_sum = weights.sum()
            mean_x = np.average(values_x, weights=weights)
            mean_y = np.average(values_y, weights=weights)
            cov_xy = np.sum(weights * (values_x - mean_x) * (values_y - mean_y)) / w_sum
            var_x = np.sum(weights * (values_x - mean_x)**2) / w_sum
            var_y = np.sum(weights * (values_y - mean_y)**2) / w_sum
            weighted_corr = cov_xy / np.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 0

        # Bar comparison
        correlations = [unweighted_corr or 0, weighted_corr or 0]
        labels = ['Unweighted', 'Weighted']
        colors = [self.palette.neutral, self.palette.highlight]

        bars = ax3.bar(labels, correlations, color=colors, edgecolor='white', linewidth=2)

        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax3.annotate(f'{corr:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax3.set_ylabel("Pearson Correlation")
        ax3.set_title("Effect of Weighting on Correlation")

        # Show the change
        if unweighted_corr and weighted_corr:
            delta = weighted_corr - unweighted_corr
            delta_pct = 100 * delta / (abs(unweighted_corr) + 1e-10)

            if abs(delta) < 0.01:
                change_msg = "Minimal change"
            elif delta > 0:
                change_msg = f"↑ {abs(delta):.3f} ({abs(delta_pct):.1f}% increase)"
            else:
                change_msg = f"↓ {abs(delta):.3f} ({abs(delta_pct):.1f}% decrease)"

            ax3.text(0.5, 0.02, change_msg, transform=ax3.transAxes,
                    ha='center', fontsize=10, style='italic',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Set y-axis limits
        ax3.set_ylim(min(0, min(correlations) - 0.1), max(1, max(correlations) + 0.1))

        fig.suptitle("Weighted Correlation: Downweighting Outlier Influence",
                    fontweight='bold', fontsize=12)
        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Weighted Correlation",
            description=f"r_unweighted={unweighted_corr:.3f}, r_weighted={weighted_corr:.3f}",
            figure_type="matplotlib"
        )

    def plot_adaptive_summary_card(
        self,
        matrix_before: BioMatrix,
        matrix_after: BioMatrix,
        medcouples: Optional[np.ndarray] = None,
        outlier_pvalues: Optional[np.ndarray] = None,
        degrees_of_freedom: Optional[float] = None,
        weights: Optional[np.ndarray] = None,
        figsize: tuple[float, float] = (16, 12)
    ) -> Figure:
        """
        Unified summary card for adaptive outlier handling.

        Combines insights from all adaptive methods into a single dashboard:
        - Top-left: Intervention magnitude (standard)
        - Top-right: Skewness adaptation (medcouple)
        - Bottom-left: Probabilistic scores (Student's t)
        - Bottom-right: Weight distribution (weighted correlation)

        Parameters
        ----------
        matrix_before : BioMatrix
            Data before imputation
        matrix_after : BioMatrix
            Data after imputation
        medcouples : np.ndarray, optional
            Medcouple values per feature
        outlier_pvalues : np.ndarray, optional
            P-values from Student's t scoring
        degrees_of_freedom : float, optional
            Fitted df parameter
        weights : np.ndarray, optional
            Sample weights for weighted correlation
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig)

        # Get outlier mask
        outlier_mask = matrix_before.data != matrix_after.data
        n_outliers = outlier_mask.sum()
        total = matrix_before.data.size
        pct = 100 * n_outliers / total

        # === Panel 1: Intervention magnitude (standard view) ===
        ax1 = fig.add_subplot(gs[0, 0])

        orig = matrix_before.data[outlier_mask]
        imputed = matrix_after.data[outlier_mask]
        delta = imputed - orig

        if len(orig) > 0:
            pulled_down = delta < 0
            pulled_up = delta > 0

            ax1.scatter(orig[pulled_down], imputed[pulled_down],
                       alpha=0.4, s=12, c=self.palette.case,
                       label=f'Down ({pulled_down.sum():,})', edgecolors='none')
            ax1.scatter(orig[pulled_up], imputed[pulled_up],
                       alpha=0.4, s=12, c=self.palette.ctrl,
                       label=f'Up ({pulled_up.sum():,})', edgecolors='none')

            all_vals = np.concatenate([orig, imputed])
            lims = [all_vals.min(), all_vals.max()]
            ax1.plot(lims, lims, 'k-', linewidth=1.5, alpha=0.7)

            if all_vals.max() / (all_vals.min() + 1) > 100:
                ax1.set_xscale('log')
                ax1.set_yscale('log')

        ax1.set_xlabel("Original")
        ax1.set_ylabel("Imputed")
        ax1.set_title("Winsorization Pattern")
        ax1.legend(loc='upper left', fontsize=7)

        # === Panel 2: Skewness adaptation ===
        ax2 = fig.add_subplot(gs[0, 1])

        if medcouples is not None:
            ax2.hist(medcouples, bins=40, color=self.palette.neutral, edgecolor='white', alpha=0.7)
            ax2.axvspan(-1, -0.1, alpha=0.15, color=self.palette.case)
            ax2.axvspan(-0.1, 0.1, alpha=0.15, color=self.palette.neutral)
            ax2.axvspan(0.1, 1, alpha=0.15, color=self.palette.ctrl)
            ax2.axvline(0, color='black', linestyle='-', linewidth=2)
            ax2.set_xlim(-1, 1)

            n_left = (medcouples < -0.1).sum()
            n_right = (medcouples > 0.1).sum()
            ax2.text(0.02, 0.98, f"L:{n_left:,} R:{n_right:,}",
                    transform=ax2.transAxes, va='top', fontsize=9)
        else:
            ax2.text(0.5, 0.5, "Medcouple not computed", ha='center', va='center',
                    transform=ax2.transAxes, fontsize=10, color='gray')

        ax2.set_xlabel("Medcouple")
        ax2.set_ylabel("Features")
        ax2.set_title("Skewness Distribution")

        # === Panel 3: Probabilistic scores ===
        ax3 = fig.add_subplot(gs[1, 0])

        if outlier_pvalues is not None:
            log_pvals = -np.log10(outlier_pvalues.flatten() + 1e-300)
            bins = np.linspace(0, min(30, log_pvals.max()), 60)
            n, bins_edges, patches = ax3.hist(log_pvals, bins=bins, edgecolor='white', linewidth=0.3)

            cmap = plt.cm.YlOrRd
            for i, patch in enumerate(patches):
                bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
                intensity = min(1.0, bin_center / 10)
                patch.set_facecolor(cmap(intensity))

            ax3.axvline(-np.log10(0.05), color='black', linestyle='--', linewidth=2)
            ax3.axvline(-np.log10(0.001), color='black', linestyle='-.', linewidth=2)

            df_text = f"df={degrees_of_freedom:.1f}" if degrees_of_freedom else ""
            ax3.text(0.98, 0.98, df_text, transform=ax3.transAxes,
                    va='top', ha='right', fontsize=9)
        else:
            ax3.text(0.5, 0.5, "P-values not computed", ha='center', va='center',
                    transform=ax3.transAxes, fontsize=10, color='gray')

        ax3.set_xlabel("-log₁₀(p-value)")
        ax3.set_ylabel("Count")
        ax3.set_title("Outlier Confidence")

        # === Panel 4: Weight distribution ===
        ax4 = fig.add_subplot(gs[1, 1])

        if weights is not None:
            ax4.hist(weights.flatten(), bins=50, color=self.palette.highlight,
                    edgecolor='white', alpha=0.8)
            ax4.axvline(1.0, color='black', linestyle='-', linewidth=2)
            ax4.axvline(0.5, color=self.palette.outlier, linestyle='--', linewidth=2)
            ax4.set_xlim(0, 1.05)

            n_full = (weights > 0.95).sum()
            n_low = (weights <= 0.5).sum()
            ax4.text(0.02, 0.98, f"Full:{n_full:,} Low:{n_low:,}",
                    transform=ax4.transAxes, va='top', fontsize=9)
        else:
            ax4.text(0.5, 0.5, "Weights not computed", ha='center', va='center',
                    transform=ax4.transAxes, fontsize=10, color='gray')

        ax4.set_xlabel("Weight")
        ax4.set_ylabel("Count")
        ax4.set_title("Sample Weights")

        fig.suptitle(f"Adaptive Outlier Handling Summary\n"
                    f"{n_outliers:,} outliers ({pct:.2f}%) detected and processed",
                    fontweight='bold', fontsize=13)
        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Adaptive Outlier Summary",
            description=f"{n_outliers:,} outliers ({pct:.2f}%) with adaptive methods",
            figure_type="matplotlib"
        )

    # =========================================================================
    # SEX CLASSIFICATION NARRATIVE
    # =========================================================================

    def plot_sex_discriminative_signal(
        self,
        matrix: BioMatrix,
        sex_labels: np.ndarray,
        top_features: list[str],
        figsize: tuple[float, float] = (12, 5)
    ) -> Figure:
        """
        Show the biological signal that discriminates M/F.

        Split violin plots for top discriminative proteins.
        Reveals: What expression patterns define sex?
        """
        n_features = min(len(top_features), 4)
        fig, axes = plt.subplots(1, n_features, figsize=figsize, sharey=False)
        if n_features == 1:
            axes = [axes]

        for i, feat in enumerate(top_features[:n_features]):
            ax = axes[i]

            # Get expression for this feature
            if feat in matrix.feature_ids:
                idx = np.where(matrix.feature_ids == feat)[0][0]
                values = matrix.data[idx, :]
            else:
                continue

            # Build dataframe for seaborn
            df = pd.DataFrame({
                'expression': values,
                'sex': sex_labels
            })
            df = df[df['sex'].isin(['M', 'F'])]  # Filter valid

            # Split violin
            sns.violinplot(data=df, x='sex', y='expression', ax=ax, hue='sex',
                          palette={'M': self.palette.male, 'F': self.palette.female},
                          inner='quartile', cut=0, legend=False)

            # Stats
            m_vals = df[df['sex'] == 'M']['expression']
            f_vals = df[df['sex'] == 'F']['expression']

            if len(m_vals) > 0 and len(f_vals) > 0:
                fold_change = np.median(f_vals) - np.median(m_vals)
                _, pval = stats.mannwhitneyu(m_vals, f_vals, alternative='two-sided')

                stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                gene_name = get_gene_symbol(feat)
                ax.set_title(f"{gene_name}\nΔ={fold_change:.2f} {stars}", fontsize=10)
            else:
                gene_name = get_gene_symbol(feat)
                ax.set_title(gene_name, fontsize=10)

            ax.set_xlabel("")
            ax.set_ylabel("Expression" if i == 0 else "")

        fig.suptitle("Sex-Discriminative Features", fontweight='bold')
        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Sex Discriminative Signal",
            description=f"Top {n_features} features separating M/F",
            figure_type="matplotlib"
        )

    def plot_sex_confidence_distribution(
        self,
        probabilities: np.ndarray,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        figsize: tuple[float, float] = (10, 6)
    ) -> Figure:
        """
        Show classification confidence distribution.

        X-axis: P(Male) from 0 to 1
        Stacked histogram by ground truth status

        Reveals: Are predictions confident? Where's ambiguity?
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Separate by ground truth status
        has_gt = ~pd.isna(ground_truth)

        # Known males
        known_m = has_gt & (ground_truth == 'M')
        # Known females
        known_f = has_gt & (ground_truth == 'F')
        # Unknown (imputed)
        unknown = ~has_gt

        bins = np.linspace(0, 1, 21)

        # Stacked histogram
        ax.hist([probabilities[known_m], probabilities[known_f], probabilities[unknown]],
               bins=bins, stacked=True,
               color=[self.palette.male, self.palette.female, self.palette.neutral],
               label=[f'Known Male (n={known_m.sum()})',
                      f'Known Female (n={known_f.sum()})',
                      f'Imputed (n={unknown.sum()})'],
               edgecolor='white', linewidth=0.5)

        # Decision boundary
        ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision boundary')

        # Confidence zones
        ax.axvspan(0, 0.2, alpha=0.1, color=self.palette.female, label='High confidence F')
        ax.axvspan(0.8, 1, alpha=0.1, color=self.palette.male, label='High confidence M')
        ax.axvspan(0.4, 0.6, alpha=0.1, color=self.palette.neutral, label='Ambiguous')

        ax.set_xlabel("P(Male)")
        ax.set_ylabel("Count")
        ax.set_title("Sex Classification Confidence")
        ax.legend(loc='upper center', fontsize=8, ncol=2)

        # Add summary stats
        high_conf = ((probabilities < 0.2) | (probabilities > 0.8)).mean()
        ax.text(0.02, 0.98, f"High confidence: {100*high_conf:.0f}%",
               transform=ax.transAxes, va='top', fontsize=10)

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Sex Classification Confidence",
            description=f"{100*high_conf:.0f}% high-confidence predictions",
            figure_type="matplotlib"
        )

    def plot_imputed_sex_placement(
        self,
        matrix: BioMatrix,
        sex_labels: np.ndarray,
        ground_truth: np.ndarray,
        top_features: list[str],
        figsize: tuple[float, float] = (10, 8)
    ) -> Figure:
        """
        Show where imputed samples fall in feature space.

        2D scatter using top 2 discriminative features.
        Ground truth samples as background, imputed highlighted.

        Reveals: Are imputed samples clearly one sex, or borderline?
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Get top 2 features
        feat1, feat2 = top_features[0], top_features[1] if len(top_features) > 1 else top_features[0]

        idx1 = np.where(matrix.feature_ids == feat1)[0]
        idx2 = np.where(matrix.feature_ids == feat2)[0]

        if len(idx1) == 0 or len(idx2) == 0:
            ax.text(0.5, 0.5, "Features not found", ha='center', va='center')
            return Figure(fig=fig, title="Imputed Sex Placement",
                         description="Features not found", figure_type="matplotlib")

        x = matrix.data[idx1[0], :]
        y = matrix.data[idx2[0], :]

        # Separate by status
        has_gt = ~pd.isna(ground_truth)
        known_m = has_gt & (np.array(ground_truth) == 'M')
        known_f = has_gt & (np.array(ground_truth) == 'F')
        imputed = ~has_gt

        # Plot known samples (background)
        ax.scatter(x[known_m], y[known_m], c=self.palette.male, alpha=0.3, s=30,
                  label=f'Known Male (n={known_m.sum()})', edgecolors='none')
        ax.scatter(x[known_f], y[known_f], c=self.palette.female, alpha=0.3, s=30,
                  label=f'Known Female (n={known_f.sum()})', edgecolors='none')

        # Plot imputed samples (foreground, highlighted)
        if imputed.any():
            imputed_predictions = sex_labels[imputed]
            imputed_m = imputed & (sex_labels == 'M')
            imputed_f = imputed & (sex_labels == 'F')

            ax.scatter(x[imputed_m], y[imputed_m], c=self.palette.male, s=100,
                      marker='*', edgecolors='black', linewidths=1,
                      label=f'Imputed → Male (n={imputed_m.sum()})')
            ax.scatter(x[imputed_f], y[imputed_f], c=self.palette.female, s=100,
                      marker='*', edgecolors='black', linewidths=1,
                      label=f'Imputed → Female (n={imputed_f.sum()})')

        # Use gene symbols for axis labels
        gene1 = get_gene_symbol(feat1)
        gene2 = get_gene_symbol(feat2)
        ax.set_xlabel(gene1)
        ax.set_ylabel(gene2)
        ax.set_title("Imputed Samples in Feature Space")
        ax.legend(loc='best', fontsize=9)

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Imputed Sex Placement",
            description=f"{imputed.sum()} imputed samples shown in {gene1} × {gene2} space",
            figure_type="matplotlib"
        )

    def plot_sex_phenotype_interaction(
        self,
        matrix: BioMatrix,
        sex_labels: np.ndarray,
        top_feature: str,
        figsize: tuple[float, float] = (10, 6)
    ) -> Figure:
        """
        Show sex × phenotype interaction for top feature.

        2×2 faceted violin: CASE/CTRL × M/F

        Reveals: Does sex effect differ by disease status?
        """
        if matrix.sample_metadata is None or "phenotype" not in matrix.sample_metadata.columns:
            raise ValueError("phenotype column required")

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # Get feature expression
        if top_feature in matrix.feature_ids:
            idx = np.where(matrix.feature_ids == top_feature)[0][0]
            expression = matrix.data[idx, :]
        else:
            raise ValueError(f"Feature {top_feature} not found")

        phenotype = matrix.sample_metadata["phenotype"].values

        # Build dataframe
        df = pd.DataFrame({
            'expression': expression,
            'sex': sex_labels,
            'phenotype': phenotype
        })
        df = df[df['sex'].isin(['M', 'F'])]

        for i, pheno in enumerate(['CASE', 'CTRL']):
            ax = axes[i]
            subset = df[df['phenotype'] == pheno]

            if len(subset) > 0:
                sns.violinplot(data=subset, x='sex', y='expression', ax=ax, hue='sex',
                              palette={'M': self.palette.male, 'F': self.palette.female},
                              inner='quartile', cut=0, legend=False)

                # Stats
                m_vals = subset[subset['sex'] == 'M']['expression']
                f_vals = subset[subset['sex'] == 'F']['expression']

                if len(m_vals) > 0 and len(f_vals) > 0:
                    _, pval = stats.mannwhitneyu(m_vals, f_vals, alternative='two-sided')
                    stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                    ax.set_title(f"{pheno} (n={len(subset)}) {stars}")
                else:
                    ax.set_title(f"{pheno} (n={len(subset)})")

            ax.set_xlabel("Sex")
            gene_name = get_gene_symbol(top_feature)
            ax.set_ylabel(f"{gene_name} Expression" if i == 0 else "")

        gene_name = get_gene_symbol(top_feature)
        fig.suptitle(f"Sex × Phenotype Interaction: {gene_name}", fontweight='bold')
        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Sex × Phenotype Interaction",
            description=f"Expression of {gene_name} by sex within each phenotype",
            figure_type="matplotlib"
        )

    def plot_sex_summary_card(
        self,
        matrix: BioMatrix,
        sex_labels: np.ndarray,
        ground_truth: np.ndarray,
        top_features: list[str],
        probabilities: Optional[np.ndarray] = None,
        figsize: tuple[float, float] = (14, 10)
    ) -> Figure:
        """
        Single-figure summary of sex classification narrative.

        4-panel layout:
        - Top-left: Discriminative signal (violins)
        - Top-right: Confidence distribution
        - Bottom-left: Imputed placement in feature space
        - Bottom-right: Sex × phenotype interaction
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig)

        # Stats for title
        has_gt = ~pd.isna(ground_truth)
        n_imputed = (~has_gt).sum()
        n_male = (sex_labels == 'M').sum()
        n_female = (sex_labels == 'F').sum()

        # === Panel 1: Discriminative signal ===
        ax1 = fig.add_subplot(gs[0, 0])

        if len(top_features) > 0 and top_features[0] in matrix.feature_ids:
            feat = top_features[0]
            idx = np.where(matrix.feature_ids == feat)[0][0]
            values = matrix.data[idx, :]

            df = pd.DataFrame({'expression': values, 'sex': sex_labels})
            df = df[df['sex'].isin(['M', 'F'])]

            sns.violinplot(data=df, x='sex', y='expression', ax=ax1, hue='sex',
                          palette={'M': self.palette.male, 'F': self.palette.female},
                          inner='quartile', cut=0, legend=False)
            gene_name = get_gene_symbol(feat)
            ax1.set_title(f"Top Feature: {gene_name}")
            ax1.set_xlabel("Sex")
            ax1.set_ylabel("Expression")

        # === Panel 2: Confidence distribution ===
        ax2 = fig.add_subplot(gs[0, 1])

        if probabilities is not None:
            # Use actual probabilities
            bins = np.linspace(0, 1, 21)
            ax2.hist([probabilities[has_gt & (ground_truth == 'M')],
                     probabilities[has_gt & (ground_truth == 'F')],
                     probabilities[~has_gt]],
                    bins=bins, stacked=True,
                    color=[self.palette.male, self.palette.female, self.palette.neutral],
                    label=['Known M', 'Known F', 'Imputed'],
                    edgecolor='white')
            ax2.axvline(0.5, color='black', linestyle='--', linewidth=2)
            ax2.set_xlabel("P(Male)")
            ax2.legend(loc='upper right', fontsize=8)
        else:
            # Show prediction distribution as proxy
            pred_m = (sex_labels == 'M')
            ax2.bar(['Male', 'Female'], [pred_m.sum(), (~pred_m).sum()],
                   color=[self.palette.male, self.palette.female], edgecolor='white')
            ax2.set_ylabel("Count")

        ax2.set_title("Classification Distribution")

        # === Panel 3: Feature space ===
        ax3 = fig.add_subplot(gs[1, 0])

        if len(top_features) >= 2:
            feat1, feat2 = top_features[0], top_features[1]
            idx1 = np.where(matrix.feature_ids == feat1)[0]
            idx2 = np.where(matrix.feature_ids == feat2)[0]

            if len(idx1) > 0 and len(idx2) > 0:
                x = matrix.data[idx1[0], :]
                y = matrix.data[idx2[0], :]

                # Plot by ground truth
                known_m = has_gt & (np.array(ground_truth) == 'M')
                known_f = has_gt & (np.array(ground_truth) == 'F')
                imputed = ~has_gt

                ax3.scatter(x[known_m], y[known_m], c=self.palette.male, alpha=0.4, s=20, label='Known M')
                ax3.scatter(x[known_f], y[known_f], c=self.palette.female, alpha=0.4, s=20, label='Known F')

                if imputed.any():
                    ax3.scatter(x[imputed], y[imputed], c='black', s=60, marker='*',
                               edgecolors='white', label=f'Imputed (n={imputed.sum()})')

                # Use gene symbols for axis labels
                gene1 = get_gene_symbol(feat1)
                gene2 = get_gene_symbol(feat2)
                ax3.set_xlabel(gene1)
                ax3.set_ylabel(gene2)
                ax3.legend(fontsize=8)

        ax3.set_title("Imputed Samples in Feature Space")

        # === Panel 4: Phenotype breakdown ===
        ax4 = fig.add_subplot(gs[1, 1])

        if matrix.sample_metadata is not None and "phenotype" in matrix.sample_metadata.columns:
            phenotype = matrix.sample_metadata["phenotype"].values

            # Count matrix
            counts = {}
            for pheno in ['CASE', 'CTRL']:
                for sex in ['M', 'F']:
                    mask = (phenotype == pheno) & (sex_labels == sex)
                    counts[f"{pheno}_{sex}"] = mask.sum()

            # Grouped bar
            x = np.arange(2)
            width = 0.35

            ax4.bar(x - width/2, [counts['CASE_M'], counts['CTRL_M']], width,
                   label='Male', color=self.palette.male)
            ax4.bar(x + width/2, [counts['CASE_F'], counts['CTRL_F']], width,
                   label='Female', color=self.palette.female)

            ax4.set_xticks(x)
            ax4.set_xticklabels(['CASE', 'CTRL'])
            ax4.set_ylabel("Count")
            ax4.legend()

        ax4.set_title("Sex × Phenotype Distribution")

        fig.suptitle(f"Sex Classification Summary: {n_male} M, {n_female} F ({n_imputed} imputed)",
                    fontweight='bold', fontsize=12)

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Sex Classification Summary",
            description=f"{n_male} male, {n_female} female, {n_imputed} imputed",
            figure_type="matplotlib"
        )
