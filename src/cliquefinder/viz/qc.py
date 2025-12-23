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
