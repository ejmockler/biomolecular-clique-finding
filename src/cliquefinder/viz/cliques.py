"""
Clique visualizations for differential regulatory analysis.

This module provides two complementary views of regulatory cliques:

1. Stratum Heatmap (plot_stratum_heatmap):
   Shows MEMBERSHIP patterns - which genes are in which condition's clique.
   Answers: "What changes across conditions?"

2. Network Graph (plot_clique_network):
   Shows CORRELATION structure - why these genes form a clique.
   Answers: "Why are these genes grouped together?"

Design Philosophy (Perceptual Engineering):
    The Stratum Heatmap Grid builds spatial memory through INVARIANT LAYOUT.
    Users learn "top = healthy, bottom = disease" without conscious effort.

    The Network Graph uses Gestalt grouping (proximity + connection) to reveal
    co-expression structure. Force-directed layout clusters tightly correlated genes.

    Key cognitive operations:
    - Pattern comparison across conditions (spatial)
    - Identifying disease-specific vs stable genes (clustering + annotation)
    - Understanding regulatory mechanisms (network topology)
    - Quantifying membership shifts (marginal annotations)

Stratum Heatmap Narrative:
    "For a given regulator, these are the genes that form coherent regulatory
    cliques in each demographic stratum. Some genes are stable across all
    conditions, some are lost in disease, some are gained."

Network Graph Narrative:
    "These genes form a clique because they are co-expressed. The network shows
    which genes have strong correlations (thick edges) and which are peripheral
    members (fewer connections)."
"""

from __future__ import annotations

from typing import Optional, Literal
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import networkx as nx

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.viz.core import Figure
from cliquefinder.viz.styles import Palette, PALETTES, configure_style, italicize_gene
from cliquefinder.viz.id_mapper import get_gene_symbol


class CliqueVisualizer:
    """
    Visualizations for regulatory clique membership and comparison.

    Designed for comparing clique composition across demographic strata
    (phenotype × sex stratification).
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

    def plot_stratum_heatmap(
        self,
        regulator: str,
        df: pd.DataFrame,
        figsize: tuple[float, float] = (10, 12),
        cluster_genes: bool = True,
        show_sample_counts: bool = True,
        show_margin_annotations: bool = True
    ) -> Figure:
        """
        Create a 2×2 stratum heatmap grid showing gene membership across conditions.

        This visualization builds spatial memory through consistent layout:
        - Top row = CTRL (healthy), Bottom row = CASE (disease)
        - Left column = Female, Right column = Male

        Parameters
        ----------
        regulator : str
            Regulator gene symbol (e.g., "MAPT")
        df : pd.DataFrame
            Stratified cliques DataFrame with columns:
            - regulator: str
            - condition: str (CASE_F, CASE_M, CTRL_F, CTRL_M)
            - n_samples: int
            - clique_genes: str (comma-separated gene symbols)
        figsize : tuple[float, float], default (10, 12)
            Figure size in inches
        cluster_genes : bool, default True
            Whether to hierarchically cluster genes for visual grouping
        show_sample_counts : bool, default True
            Whether to show n_samples in subplot titles
        show_margin_annotations : bool, default True
            Whether to show CTRL-only/CASE-only/Stable gene counts

        Returns
        -------
        Figure
            Figure wrapper containing the heatmap grid

        Examples
        --------
        >>> import pandas as pd
        >>> from cliquefinder.viz import CliqueVisualizer
        >>>
        >>> # Load stratified cliques
        >>> df = pd.read_csv("results/cliques/cliques.csv")
        >>>
        >>> # Create heatmap for MAPT regulator
        >>> viz = CliqueVisualizer(style="paper")
        >>> fig = viz.plot_stratum_heatmap("MAPT", df)
        >>> fig.save("figures/mapt_stratum_heatmap.pdf")
        >>>
        >>> # The heatmap reveals:
        >>> # - Genes present only in CTRL (lost in disease) → top-row-only
        >>> # - Genes present only in CASE (gained in disease) → bottom-row-only
        >>> # - Stable genes (present in both) → full row filled

        Notes
        -----
        Perceptual encoding:
        - Cell fill: Dark = gene present, Light = gene absent
        - Row ordering: Hierarchical clustering groups co-occurring genes
        - Color tint: Orange gradient for CTRL, Blue gradient for CASE
        - Layout invariance: Always [CTRL_F, CTRL_M] top, [CASE_F, CASE_M] bottom
        """
        # Filter data for this regulator
        reg_data = df[df['regulator'] == regulator].copy()

        if len(reg_data) == 0:
            raise ValueError(f"No data found for regulator '{regulator}'")

        # Extract gene lists and build presence/absence matrix
        gene_lists = {}
        sample_counts = {}

        for _, row in reg_data.iterrows():
            condition = row['condition']
            # Skip rows with NaN clique_genes
            if pd.isna(row['clique_genes']):
                gene_lists[condition] = set()
            else:
                genes = [g.strip() for g in str(row['clique_genes']).split(',')]
                gene_lists[condition] = set(genes)
            sample_counts[condition] = row['n_samples']

        # Get union of all genes
        all_genes = sorted(set().union(*gene_lists.values()))
        n_genes = len(all_genes)

        # Build presence/absence matrix
        # Columns in FIXED ORDER: CTRL_F, CTRL_M, CASE_F, CASE_M
        stratum_order = ['CTRL_F', 'CTRL_M', 'CASE_F', 'CASE_M']
        stratum_labels = ['CTRL_F', 'CTRL_M', 'CASE_F', 'CASE_M']

        # Ensure all strata exist (fill missing with empty sets)
        for stratum in stratum_order:
            if stratum not in gene_lists:
                gene_lists[stratum] = set()
                sample_counts[stratum] = 0

        # Build matrix: rows = genes, columns = strata
        matrix = np.zeros((n_genes, 4), dtype=int)
        for i, gene in enumerate(all_genes):
            for j, stratum in enumerate(stratum_order):
                if gene in gene_lists[stratum]:
                    matrix[i, j] = 1

        # Cluster genes if requested
        gene_order = list(range(n_genes))
        if cluster_genes and n_genes > 1:
            # Cluster based on co-occurrence pattern
            # Use Jaccard distance for binary data
            try:
                distances = pdist(matrix, metric='jaccard')
                if not np.all(np.isnan(distances)):
                    linkage = hierarchy.linkage(distances, method='average')
                    dendro = hierarchy.dendrogram(linkage, no_plot=True)
                    gene_order = dendro['leaves']
            except:
                # Fall back to original order if clustering fails
                pass

        # Reorder matrix
        matrix_ordered = matrix[gene_order, :]
        genes_ordered = [all_genes[i] for i in gene_order]

        # Compute gene category annotations
        ctrl_genes = gene_lists['CTRL_F'] | gene_lists['CTRL_M']
        case_genes = gene_lists['CASE_F'] | gene_lists['CASE_M']

        ctrl_only = ctrl_genes - case_genes
        case_only = case_genes - ctrl_genes
        stable = ctrl_genes & case_genes

        # Create figure with GridSpec for fine control
        fig = plt.figure(figsize=figsize)

        if show_margin_annotations:
            # Reserve space for margin annotations
            gs = GridSpec(
                2, 3,
                figure=fig,
                width_ratios=[0.45, 0.45, 0.1],
                height_ratios=[0.5, 0.5],
                hspace=0.15,
                wspace=0.05,
                left=0.25,
                right=0.95,
                top=0.93,
                bottom=0.05
            )
        else:
            gs = GridSpec(
                2, 2,
                figure=fig,
                width_ratios=[0.5, 0.5],
                height_ratios=[0.5, 0.5],
                hspace=0.15,
                wspace=0.05,
                left=0.25,
                right=0.95,
                top=0.93,
                bottom=0.05
            )

        # Define subplots in 2×2 grid
        # Top row = CTRL, Bottom row = CASE
        # Left col = Female, Right col = Male
        positions = {
            'CTRL_F': (0, 0),  # Top-left
            'CTRL_M': (0, 1),  # Top-right
            'CASE_F': (1, 0),  # Bottom-left
            'CASE_M': (1, 1),  # Bottom-right
        }

        axes = {}
        for stratum, (row, col) in positions.items():
            axes[stratum] = fig.add_subplot(gs[row, col])

        # Color schemes: orange tint for CTRL, blue tint for CASE
        colors = {
            'CTRL_F': ['#fff7ed', '#fb923c'],  # Orange: light → dark
            'CTRL_M': ['#fff7ed', '#f97316'],  # Orange: light → dark (slightly darker)
            'CASE_F': ['#eff6ff', '#60a5fa'],  # Blue: light → dark
            'CASE_M': ['#eff6ff', '#2563eb'],  # Blue: light → dark (slightly darker)
        }

        # Plot each stratum
        for stratum_idx, stratum in enumerate(stratum_order):
            ax = axes[stratum]

            # Extract column for this stratum
            col_data = matrix_ordered[:, stratum_idx].reshape(-1, 1)

            # Create custom colormap
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(colors[stratum])

            # Plot heatmap
            im = ax.imshow(
                col_data,
                aspect='auto',
                cmap=cmap,
                interpolation='nearest',
                vmin=0,
                vmax=1
            )

            # Set title with sample count
            phenotype, sex = stratum.split('_')
            sex_label = 'Female' if sex == 'F' else 'Male'

            if show_sample_counts:
                n_samples = sample_counts[stratum]
                title = f"{sex_label}\n(n={n_samples})"
            else:
                title = sex_label

            ax.set_title(title, fontsize=10, fontweight='bold')

            # Remove x-axis
            ax.set_xticks([])
            ax.set_xlabel('')

            # Y-axis: only show gene labels on leftmost column
            if stratum in ['CTRL_F', 'CASE_F']:
                ax.set_yticks(range(n_genes))
                ax.set_yticklabels(
                    [italicize_gene(g) for g in genes_ordered],
                    fontsize=7
                )
                ax.tick_params(axis='y', which='both', length=0)
            else:
                ax.set_yticks([])

            # Add row label (phenotype) on left
            if stratum in ['CTRL_F', 'CASE_F']:
                phenotype_label = 'Healthy' if phenotype == 'CTRL' else 'ALS'
                ax.set_ylabel(
                    phenotype_label,
                    fontsize=11,
                    fontweight='bold',
                    rotation=90,
                    va='center'
                )

            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Add margin annotations if requested
        if show_margin_annotations:
            ax_margin = fig.add_subplot(gs[:, 2])
            ax_margin.set_xlim(0, 1)
            ax_margin.set_ylim(0, n_genes)
            ax_margin.axis('off')

            # Color code genes by category
            for i, gene in enumerate(genes_ordered):
                y_pos = n_genes - i - 0.5

                if gene in ctrl_only:
                    color = '#fb923c'  # Orange
                    marker = '○'
                elif gene in case_only:
                    color = '#2563eb'  # Blue
                    marker = '●'
                elif gene in stable:
                    color = '#059669'  # Green
                    marker = '■'
                else:
                    continue

                ax_margin.text(
                    0.3, y_pos, marker,
                    fontsize=8,
                    color=color,
                    ha='center',
                    va='center'
                )

            # Add legend at bottom
            legend_y = -0.5
            ax_margin.text(0.3, legend_y, '○ CTRL-only', fontsize=7,
                          color='#fb923c', ha='center', va='top')
            ax_margin.text(0.3, legend_y - 1, '● CASE-only', fontsize=7,
                          color='#2563eb', ha='center', va='top')
            ax_margin.text(0.3, legend_y - 2, '■ Stable', fontsize=7,
                          color='#059669', ha='center', va='top')

        # Add main title
        fig.suptitle(
            f"{italicize_gene(regulator)} Regulatory Clique Membership Across Strata",
            fontsize=13,
            fontweight='bold',
            y=0.98
        )

        # Add caption with summary statistics
        caption = (
            f"Gene membership patterns: {len(stable)} stable, "
            f"{len(ctrl_only)} CTRL-only, {len(case_only)} CASE-only"
        )
        fig.text(
            0.5, 0.01,
            caption,
            ha='center',
            fontsize=9,
            style='italic',
            color='#6b7280'
        )

        return Figure(
            fig=fig,
            title=f"{regulator} Stratum Heatmap",
            description=(
                f"Gene membership in {regulator} regulatory cliques across "
                f"4 demographic strata (CTRL/CASE × Female/Male). "
                f"Reveals condition-specific and stable regulatory relationships."
            ),
            figure_type="matplotlib",
            metadata={
                'regulator': regulator,
                'n_genes': n_genes,
                'n_stable': len(stable),
                'n_ctrl_only': len(ctrl_only),
                'n_case_only': len(case_only),
                'strata': stratum_order,
            }
        )

    def plot_clique_network(
        self,
        regulator: str,
        df: pd.DataFrame,
        correlation_data: Optional[np.ndarray | pd.DataFrame | BioMatrix] = None,
        condition_focus: Optional[str] = None,
        correlation_threshold: float = 0.4,
        figsize: tuple[float, float] = (10, 10),
        node_size_scale: float = 300,
        edge_width_scale: float = 3.0,
        layout_iterations: int = 50,
        layout_k: Optional[float] = None
    ) -> Figure:
        """
        Create network graph showing correlation structure within a regulatory clique.

        This visualization reveals WHY genes form a clique by showing their
        co-expression relationships. Force-directed layout clusters tightly
        correlated genes together.

        Parameters
        ----------
        regulator : str
            Regulator gene symbol (e.g., "MAPT")
        df : pd.DataFrame
            Stratified cliques DataFrame with columns:
            - regulator: str
            - condition: str (CASE_F, CASE_M, CTRL_F, CTRL_M)
            - clique_genes: str (comma-separated gene symbols)
        correlation_data : np.ndarray, pd.DataFrame, or BioMatrix, optional
            Gene expression data for computing correlations.
            - If np.ndarray: genes × samples matrix (feature names inferred)
            - If pd.DataFrame: genes (index) × samples (columns)
            - If BioMatrix: uses data and feature_ids
            - If None: shows membership-only network (no edge weights)
        condition_focus : str, optional
            If provided, show only this condition's clique.
            Otherwise shows union across all conditions.
        correlation_threshold : float, default 0.4
            Minimum correlation to draw an edge (absolute value).
        figsize : tuple[float, float], default (10, 10)
            Figure size in inches
        node_size_scale : float, default 300
            Scaling factor for node sizes (based on degree centrality)
        edge_width_scale : float, default 3.0
            Scaling factor for edge widths (based on correlation strength)
        layout_iterations : int, default 50
            Number of iterations for force-directed layout
        layout_k : float, optional
            Optimal distance between nodes in layout. If None, auto-computed.

        Returns
        -------
        Figure
            Figure wrapper containing the network graph

        Examples
        --------
        >>> import pandas as pd
        >>> from cliquefinder.viz import CliqueVisualizer
        >>> from cliquefinder.core.biomatrix import BioMatrix
        >>>
        >>> # Load cliques and expression data
        >>> df = pd.read_csv("results/cliques/cliques.csv")
        >>> expr = pd.read_csv("results/proteomics_imputed.data.csv", index_col=0)
        >>>
        >>> # Create network with correlations
        >>> viz = CliqueVisualizer(style="paper")
        >>> fig = viz.plot_clique_network("MAPT", df, correlation_data=expr)
        >>> fig.save("figures/mapt_clique_network.pdf")
        >>>
        >>> # Network reveals:
        >>> # - Hub genes (large nodes) with many strong correlations
        >>> # - Peripheral genes (small nodes) with fewer connections
        >>> # - Condition-specific genes colored by membership pattern

        Notes
        -----
        Perceptual encoding:
        - Node color: Membership pattern across conditions
          - Emerald (#059669): Present in ALL conditions (stable)
          - Blue (#2563eb): Present only in CASE conditions
          - Orange (#f97316): Present only in CTRL conditions
          - Gray (#6b7280): Present in some conditions
        - Node size: Degree centrality (connectivity)
        - Edge thickness: Correlation strength (absolute value)
        - Edge color: Gray (#9ca3af) with alpha ~ correlation
        - Layout: Force-directed (spring) clusters correlated genes
        """
        # Filter data for this regulator
        reg_data = df[df['regulator'] == regulator].copy()

        if len(reg_data) == 0:
            raise ValueError(f"No data found for regulator '{regulator}'")

        # Extract gene lists for each condition
        gene_lists = {}
        for _, row in reg_data.iterrows():
            condition = row['condition']
            # Skip rows with NaN clique_genes
            if pd.isna(row['clique_genes']):
                gene_lists[condition] = set()
            else:
                genes = [g.strip() for g in str(row['clique_genes']).split(',')]
                gene_lists[condition] = set(genes)

        # Determine which genes to include
        if condition_focus:
            if condition_focus not in gene_lists:
                raise ValueError(
                    f"Condition '{condition_focus}' not found for regulator '{regulator}'. "
                    f"Available: {list(gene_lists.keys())}"
                )
            all_genes = sorted(gene_lists[condition_focus])
            network_title = f"{condition_focus} Condition"
        else:
            # Union across all conditions
            all_genes = sorted(set().union(*gene_lists.values()))
            network_title = "All Conditions"

        n_genes = len(all_genes)

        # Determine gene membership categories
        # Parse conditions into CASE and CTRL sets
        ctrl_genes = set()
        case_genes = set()
        all_conditions = set(gene_lists.keys())

        for condition, genes in gene_lists.items():
            if 'CTRL' in condition:
                ctrl_genes.update(genes)
            elif 'CASE' in condition:
                case_genes.update(genes)

        # Categorize genes
        gene_categories = {}
        for gene in all_genes:
            in_ctrl = gene in ctrl_genes
            in_case = gene in case_genes

            # Check if present in ALL conditions
            in_all = all(gene in gene_lists[cond] for cond in all_conditions)

            if in_all:
                gene_categories[gene] = 'stable'  # Emerald
            elif in_ctrl and not in_case:
                gene_categories[gene] = 'ctrl_only'  # Orange
            elif in_case and not in_ctrl:
                gene_categories[gene] = 'case_only'  # Blue
            else:
                gene_categories[gene] = 'partial'  # Gray

        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes
        for gene in all_genes:
            G.add_node(gene, category=gene_categories[gene])

        # Compute correlations if data provided
        if correlation_data is not None:
            # Parse correlation data
            if isinstance(correlation_data, BioMatrix):
                expr_df = pd.DataFrame(
                    correlation_data.data,
                    index=correlation_data.feature_ids,
                    columns=correlation_data.sample_ids
                )
            elif isinstance(correlation_data, np.ndarray):
                # Assume genes in rows, samples in columns
                # Try to infer gene names (this is a fallback)
                expr_df = pd.DataFrame(correlation_data)
            else:
                expr_df = correlation_data

            # Filter to genes in clique
            available_genes = [g for g in all_genes if g in expr_df.index]
            missing_genes = set(all_genes) - set(available_genes)

            if len(available_genes) < 2:
                warnings.warn(
                    f"Only {len(available_genes)} genes found in expression data. "
                    f"Cannot compute correlations. Showing membership-only network."
                )
                correlation_data = None
            else:
                if missing_genes:
                    warnings.warn(
                        f"{len(missing_genes)} genes not found in expression data: "
                        f"{', '.join(sorted(missing_genes)[:5])}"
                        f"{'...' if len(missing_genes) > 5 else ''}"
                    )

                # Compute pairwise correlations
                expr_subset = expr_df.loc[available_genes]
                corr_matrix = expr_subset.T.corr()

                # Add edges based on correlation threshold
                for i, gene1 in enumerate(available_genes):
                    for j, gene2 in enumerate(available_genes):
                        if i < j:  # Upper triangle only
                            corr = corr_matrix.loc[gene1, gene2]
                            abs_corr = abs(corr)

                            if abs_corr >= correlation_threshold:
                                G.add_edge(
                                    gene1,
                                    gene2,
                                    weight=corr,
                                    abs_weight=abs_corr
                                )

        # If no correlation data or couldn't compute, create fully connected graph
        if correlation_data is None or G.number_of_edges() == 0:
            # Add edges between all nodes with uniform weight
            for i, gene1 in enumerate(all_genes):
                for j, gene2 in enumerate(all_genes):
                    if i < j:
                        G.add_edge(gene1, gene2, weight=0.5, abs_weight=0.5)

        # Compute layout
        if layout_k is None:
            layout_k = 1 / np.sqrt(n_genes)

        pos = nx.spring_layout(
            G,
            k=layout_k,
            iterations=layout_iterations,
            seed=42  # Reproducible layout
        )

        # Compute node sizes based on degree centrality
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        node_sizes = [
            node_size_scale * (1 + 2 * degrees[node] / max_degree)
            for node in G.nodes()
        ]

        # Assign node colors based on category
        color_map = {
            'stable': '#059669',      # Emerald (all conditions)
            'ctrl_only': '#f97316',   # Orange (CTRL only)
            'case_only': '#2563eb',   # Blue (CASE only)
            'partial': '#6b7280'      # Gray (some conditions)
        }
        node_colors = [
            color_map[gene_categories[node]]
            for node in G.nodes()
        ]

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Draw edges
        if G.number_of_edges() > 0:
            edges = G.edges()
            weights = [G[u][v]['abs_weight'] for u, v in edges]

            # Normalize weights for visual encoding
            max_weight = max(weights) if weights else 1
            edge_widths = [edge_width_scale * w / max_weight for w in weights]
            edge_alphas = [0.3 + 0.6 * w / max_weight for w in weights]

            # Draw each edge with custom alpha
            for (u, v), width, alpha in zip(edges, edge_widths, edge_alphas):
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=width,
                    alpha=alpha,
                    edge_color='#9ca3af',  # Gray
                    ax=ax
                )

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='white',
            linewidths=1.5,
            ax=ax
        )

        # Draw labels
        labels = {node: italicize_gene(node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_color='#1a1a1a',
            ax=ax
        )

        # Create legend
        legend_elements = [
            mpatches.Patch(
                facecolor=color_map['stable'],
                edgecolor='white',
                label='All conditions'
            ),
            mpatches.Patch(
                facecolor=color_map['case_only'],
                edgecolor='white',
                label='CASE only'
            ),
            mpatches.Patch(
                facecolor=color_map['ctrl_only'],
                edgecolor='white',
                label='CTRL only'
            ),
            mpatches.Patch(
                facecolor=color_map['partial'],
                edgecolor='white',
                label='Some conditions'
            ),
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper left',
            frameon=True,
            facecolor='white',
            edgecolor='#e5e7eb',
            fontsize=9
        )

        # Title and styling
        title = f"{italicize_gene(regulator)} Regulatory Clique Network\n{network_title}"
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.axis('off')

        # Add caption
        n_edges = G.number_of_edges()
        if correlation_data is not None and n_edges > 0:
            caption = (
                f"{n_genes} genes, {n_edges} correlations ≥ {correlation_threshold:.2f}"
            )
        else:
            caption = f"{n_genes} genes (membership-only, no correlation data)"

        fig.text(
            0.5, 0.02,
            caption,
            ha='center',
            fontsize=9,
            style='italic',
            color='#6b7280'
        )

        plt.tight_layout()

        return Figure(
            fig=fig,
            title=f"{regulator} Clique Network",
            description=(
                f"Correlation structure of {regulator} regulatory clique. "
                f"Node size indicates connectivity, edge thickness shows "
                f"correlation strength, colors show condition membership."
            ),
            figure_type="matplotlib",
            metadata={
                'regulator': regulator,
                'n_genes': n_genes,
                'n_edges': n_edges,
                'correlation_threshold': correlation_threshold,
                'condition_focus': condition_focus,
                'gene_categories': dict(
                    stable=sum(1 for c in gene_categories.values() if c == 'stable'),
                    ctrl_only=sum(1 for c in gene_categories.values() if c == 'ctrl_only'),
                    case_only=sum(1 for c in gene_categories.values() if c == 'case_only'),
                    partial=sum(1 for c in gene_categories.values() if c == 'partial'),
                ),
            }
        )
    def plot_gene_flow(
        self,
        regulator: str,
        df: pd.DataFrame,
        figsize: tuple[float, float] = (14, 10),
        show_sex_split: bool = False,
        max_genes_per_flow: int = 15
    ) -> Figure:
        """
        Create a gene flow diagram showing regulatory rewiring between conditions.

        This visualization reveals CHANGE DETECTION - which genes are gained, lost,
        or stable between healthy (CTRL) and disease (CASE) states.

        Visual Layout:
        - Left column: CTRL (healthy) genes
        - Right column: CASE (disease) genes
        - Flows between columns show gene transitions:
          * Emerald (#059669): GAINED genes (new in disease)
          * Red (#ef4444): LOST genes (removed in disease)
          * Gray (#6b7280): STABLE genes (present in both)

        Parameters
        ----------
        regulator : str
            Regulator gene symbol (e.g., "MAPT")
        df : pd.DataFrame
            Stratified cliques DataFrame with columns:
            - regulator: str
            - condition: str (CASE_F, CASE_M, CTRL_F, CTRL_M)
            - clique_genes: str (comma-separated gene symbols)
        figsize : tuple[float, float], default (14, 10)
            Figure size in inches
        show_sex_split : bool, default False
            If True, show F/M separately with different flow widths.
            If False, aggregate M+F for simpler visualization.
        max_genes_per_flow : int, default 15
            Maximum number of genes to show labels for in each flow category.
            If exceeded, shows "N genes" label instead.

        Returns
        -------
        Figure
            Figure wrapper containing the gene flow diagram

        Examples
        --------
        >>> import pandas as pd
        >>> from cliquefinder.viz import CliqueVisualizer
        >>>
        >>> # Load stratified cliques
        >>> df = pd.read_csv("results/cliques/cliques.csv")
        >>>
        >>> # Create flow diagram for MAPT regulator
        >>> viz = CliqueVisualizer(style="paper")
        >>> fig = viz.plot_gene_flow("MAPT", df)
        >>> fig.save("figures/mapt_gene_flow.pdf")
        >>>
        >>> # The diagram reveals:
        >>> # - Red flows: Genes present in CTRL but lost in CASE (disease removes)
        >>> # - Emerald flows: Genes gained in CASE (disease adds)
        >>> # - Gray flows: Stable genes (present in both conditions)

        Notes
        -----
        Perceptual encoding:
        - Flow color: Encodes change direction (gain/loss/stable)
        - Flow width: Proportional to number of genes in that category
        - Gene labels: Italicized per biology convention
        - Spatial layout: Left-to-right implies temporal progression (healthy→disease)
        """
        # Filter data for this regulator
        reg_data = df[df['regulator'] == regulator].copy()

        if len(reg_data) == 0:
            raise ValueError(f"No data found for regulator '{regulator}'")

        # Extract gene lists
        gene_lists = {}
        for _, row in reg_data.iterrows():
            condition = row['condition']
            # Skip rows with NaN clique_genes
            if pd.isna(row['clique_genes']):
                gene_lists[condition] = set()
            else:
                genes = [g.strip() for g in str(row['clique_genes']).split(',')]
                gene_lists[condition] = set(genes)

        # Aggregate by condition (CTRL vs CASE)
        ctrl_genes = gene_lists.get('CTRL_F', set()) | gene_lists.get('CTRL_M', set())
        case_genes = gene_lists.get('CASE_F', set()) | gene_lists.get('CASE_M', set())

        # Categorize genes by flow type
        stable = ctrl_genes & case_genes
        lost = ctrl_genes - case_genes
        gained = case_genes - ctrl_genes

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Define column positions
        left_x = 0.15
        right_x = 0.85
        column_width = 0.08

        # Define colors for flows
        color_gained = '#059669'  # Emerald
        color_lost = '#ef4444'    # Red
        color_stable = '#6b7280'  # Gray

        # Helper function to draw a flow with Bezier curve
        def draw_flow(y_start, y_end, width, color, alpha=0.3):
            """Draw a curved flow ribbon between two vertical positions."""
            # Control points for Bezier curve
            x0, y0 = left_x + column_width, y_start
            x3, y3 = right_x, y_end
            # Control points at 1/3 and 2/3 distance
            x1, y1 = left_x + column_width + 0.2, y_start
            x2, y2 = right_x - 0.2, y_end

            # Create path vertices for filled area
            n_points = 100
            t = np.linspace(0, 1, n_points)

            # Bezier curve formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
            bezier_x = (1-t)**3 * x0 + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x3
            bezier_y = (1-t)**3 * y0 + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y3

            # Create upper and lower boundaries for the ribbon
            half_width = width / 2
            vertices_top = list(zip(bezier_x, bezier_y + half_width))
            vertices_bottom = list(zip(bezier_x[::-1], (bezier_y - half_width)[::-1]))
            vertices = vertices_top + vertices_bottom

            # Create polygon patch
            from matplotlib.patches import Polygon
            poly = Polygon(vertices, facecolor=color, edgecolor=color,
                          alpha=alpha, linewidth=0.5)
            ax.add_patch(poly)

        # Helper function to draw gene list box
        def draw_gene_box(x, y_center, genes, title, color):
            """Draw a box with gene list at specified position."""
            n_genes = len(genes)

            # Box dimensions
            box_height = max(0.05, n_genes * 0.03)
            box_top = y_center + box_height / 2
            box_bottom = y_center - box_height / 2

            # Draw box outline
            rect = Rectangle(
                (x, box_bottom),
                column_width,
                box_height,
                facecolor='white',
                edgecolor=color,
                linewidth=2,
                zorder=10
            )
            ax.add_patch(rect)

            # Title
            ax.text(
                x + column_width / 2,
                box_top + 0.02,
                f"{title}\n({n_genes} genes)",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color=color
            )

            # Gene labels (limit to max_genes_per_flow)
            if n_genes <= max_genes_per_flow:
                sorted_genes = sorted(genes)
                for i, gene in enumerate(sorted_genes):
                    y_pos = box_top - 0.01 - (i / n_genes) * box_height
                    ax.text(
                        x + column_width / 2,
                        y_pos,
                        italicize_gene(gene),
                        ha='center',
                        va='top',
                        fontsize=7,
                        color='#1a1a1a'
                    )
            else:
                # Too many genes, just show count
                ax.text(
                    x + column_width / 2,
                    y_center,
                    f"{n_genes} genes\n(see legend)",
                    ha='center',
                    va='center',
                    fontsize=8,
                    style='italic',
                    color='#6b7280'
                )

            return box_top, box_bottom, y_center

        # Calculate vertical positions for each gene set
        # Space them evenly with padding
        y_positions = {
            'lost': 0.75,
            'stable': 0.5,
            'gained': 0.25
        }

        # Draw CTRL column boxes
        ctrl_y_positions = {}
        if len(lost) > 0:
            top, bottom, center = draw_gene_box(
                left_x, y_positions['lost'], lost, "CTRL-only", color_lost
            )
            ctrl_y_positions['lost'] = center

        if len(stable) > 0:
            top, bottom, center = draw_gene_box(
                left_x, y_positions['stable'], stable, "Stable", color_stable
            )
            ctrl_y_positions['stable'] = center

        # Draw CASE column boxes
        case_y_positions = {}
        if len(stable) > 0:
            top, bottom, center = draw_gene_box(
                right_x - column_width, y_positions['stable'], stable, "Stable", color_stable
            )
            case_y_positions['stable'] = center

        if len(gained) > 0:
            top, bottom, center = draw_gene_box(
                right_x - column_width, y_positions['gained'], gained, "CASE-only", color_gained
            )
            case_y_positions['gained'] = center

        # Draw flows between columns
        # Flow width proportional to number of genes (with scaling)
        max_genes = max(len(lost), len(stable), len(gained), 1)
        scale_factor = 0.15 / max_genes  # Max width of 0.15

        # Lost flow (CTRL → nowhere, fades out)
        if len(lost) > 0:
            flow_width = len(lost) * scale_factor
            # Draw flow that fades out to the right
            y_start = ctrl_y_positions['lost']
            y_end = y_start - 0.1  # Drops down slightly
            draw_flow(y_start, y_end, flow_width, color_lost, alpha=0.25)

        # Stable flow (CTRL → CASE)
        if len(stable) > 0:
            flow_width = len(stable) * scale_factor
            y_start = ctrl_y_positions['stable']
            y_end = case_y_positions['stable']
            draw_flow(y_start, y_end, flow_width, color_stable, alpha=0.3)

        # Gained flow (nowhere → CASE, fades in)
        if len(gained) > 0:
            flow_width = len(gained) * scale_factor
            y_end = case_y_positions['gained']
            y_start = y_end + 0.1  # Comes from above
            draw_flow(y_start, y_end, flow_width, color_gained, alpha=0.25)

        # Column headers
        ax.text(
            left_x + column_width / 2,
            0.95,
            "CTRL\n(Healthy)",
            ha='center',
            va='top',
            fontsize=12,
            fontweight='bold',
            color='#f97316'  # Orange
        )

        ax.text(
            right_x - column_width / 2,
            0.95,
            "CASE\n(ALS)",
            ha='center',
            va='top',
            fontsize=12,
            fontweight='bold',
            color='#2563eb'  # Blue
        )

        # Set axis limits and turn off axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Title
        fig.suptitle(
            f"{italicize_gene(regulator)} Gene Flow: Regulatory Rewiring from Healthy to ALS",
            fontsize=14,
            fontweight='bold',
            y=0.98
        )

        # Caption with summary statistics
        caption = (
            f"Gene flow analysis: {len(lost)} genes lost in disease, "
            f"{len(gained)} genes gained in disease, {len(stable)} stable genes. "
            f"Flow width proportional to gene count."
        )
        fig.text(
            0.5, 0.02,
            caption,
            ha='center',
            fontsize=9,
            style='italic',
            color='#6b7280'
        )

        # Legend
        legend_x = 0.5
        legend_y = 0.08
        legend_elements = [
            mpatches.Patch(facecolor=color_lost, alpha=0.5, label='Lost in disease'),
            mpatches.Patch(facecolor=color_gained, alpha=0.5, label='Gained in disease'),
            mpatches.Patch(facecolor=color_stable, alpha=0.5, label='Stable (both conditions)')
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
            fontsize=9
        )

        return Figure(
            fig=fig,
            title=f"{regulator} Gene Flow Diagram",
            description=(
                f"Gene flow visualization for {regulator} regulatory cliques showing "
                f"which genes are lost, gained, or stable between healthy (CTRL) and "
                f"disease (CASE) conditions. Reveals regulatory rewiring patterns."
            ),
            figure_type="matplotlib",
            metadata={
                'regulator': regulator,
                'n_lost': len(lost),
                'n_gained': len(gained),
                'n_stable': len(stable),
                'lost_genes': sorted(lost),
                'gained_genes': sorted(gained),
                'stable_genes': sorted(stable),
            }
        )

    def plot_regulator_overview(
        self,
        rewiring_df: pd.DataFrame,
        stratified_df: pd.DataFrame,
        figsize: tuple[float, float] = (14, 10),
        min_rewiring_score: float = 0.0,
        label_top_n: int = 20,
        point_alpha: float = 0.7
    ) -> Figure:
        """
        Create regulator overview scatter plot for identifying interesting regulators.

        This is the ENTRY POINT visualization for the clique-finding analysis.
        Users scan all regulators to identify those with strong rewiring patterns.

        Perceptual encoding:
        - X axis: rewiring_score (positive = gained in CASE, negative = lost)
        - Y axis: max coherence across conditions (quality measure)
        - Point size: total clique size (number of genes)
        - Point color: sex interaction pattern

        Parameters
        ----------
        rewiring_df : pd.DataFrame
            Regulator rewiring statistics with columns:
            - regulator: str
            - comparison: str (e.g., "CASE_F_vs_CTRL_F")
            - rewiring_score: float
            - case_coherence: float
            - ctrl_coherence: float
        stratified_df : pd.DataFrame
            Stratified cliques with columns:
            - regulator: str
            - condition: str (CASE_F, CASE_M, CTRL_F, CTRL_M)
            - clique_genes: str (comma-separated)
            - coherence_ratio: float
        figsize : tuple[float, float], default (14, 10)
            Figure size in inches
        min_rewiring_score : float, default 0.0
            Minimum |rewiring_score| to display (filter weak signals)
        label_top_n : int, default 20
            Number of top regulators to label
        point_alpha : float, default 0.7
            Transparency for scatter points

        Returns
        -------
        Figure
            Figure wrapper containing the overview plot

        Examples
        --------
        >>> import pandas as pd
        >>> from cliquefinder.viz import CliqueVisualizer
        >>>
        >>> # Load data
        >>> rewiring_df = pd.read_csv("results/cliques/regulator_rewiring_stats.csv")
        >>> stratified_df = pd.read_csv("results/cliques/cliques.csv")
        >>>
        >>> # Create overview
        >>> viz = CliqueVisualizer(style="paper")
        >>> fig = viz.plot_regulator_overview(rewiring_df, stratified_df)
        >>> fig.save("figures/regulator_overview.pdf")
        >>>
        >>> # The plot reveals:
        >>> # - Top-right quadrant: High-quality cliques gained in CASE
        >>> # - Top-left quadrant: High-quality cliques lost in CASE
        >>> # - Bottom quadrants: Lower coherence changes
        >>> # - Color indicates sex-specific vs both-sex effects

        Notes
        -----
        Perceptual design for scanning:
        - Quadrant structure enables rapid visual triage
        - Color distinguishes sex-specific patterns without requiring legend lookup
        - Size encodes biological relevance (larger cliques = more genes affected)
        - Labels guide attention to strongest signals
        """
        # Aggregate regulator-level metrics
        # For each regulator, we need:
        # 1. Max rewiring score across comparisons (signed: positive=gained, negative=lost)
        # 2. Max coherence across all conditions
        # 3. Total clique size (union of all genes)
        # 4. Sex interaction pattern (both sexes, male-only, female-only)

        reg_metrics = []

        for regulator in rewiring_df['regulator'].unique():
            # Get rewiring data for this regulator
            reg_rewiring = rewiring_df[rewiring_df['regulator'] == regulator]

            # Get stratified data for this regulator
            reg_strata = stratified_df[stratified_df['regulator'] == regulator]

            if len(reg_strata) == 0:
                continue

            # Compute rewiring scores by sex
            male_comparisons = reg_rewiring[reg_rewiring['comparison'].str.contains('_M_vs_')]
            female_comparisons = reg_rewiring[reg_rewiring['comparison'].str.contains('_F_vs_')]

            # Get max absolute rewiring score, preserving sign
            male_rewiring = 0.0
            female_rewiring = 0.0

            if len(male_comparisons) > 0:
                # Choose the rewiring score with max absolute value
                male_scores = male_comparisons['rewiring_score'].values
                male_rewiring = male_scores[np.argmax(np.abs(male_scores))]

            if len(female_comparisons) > 0:
                female_scores = female_comparisons['rewiring_score'].values
                female_rewiring = female_scores[np.argmax(np.abs(female_scores))]

            # Overall rewiring score: max absolute value across sexes
            if abs(male_rewiring) >= abs(female_rewiring):
                rewiring_score = male_rewiring
            else:
                rewiring_score = female_rewiring

            # Determine sex pattern
            # Thresholds for "significant" rewiring
            rewiring_threshold = 0.25

            male_significant = abs(male_rewiring) >= rewiring_threshold
            female_significant = abs(female_rewiring) >= rewiring_threshold

            if male_significant and female_significant:
                sex_pattern = 'both'
            elif male_significant:
                sex_pattern = 'male'
            elif female_significant:
                sex_pattern = 'female'
            else:
                sex_pattern = 'none'

            # Max coherence across all conditions
            max_coherence = reg_strata['coherence_ratio'].max()

            # Total unique genes across all conditions
            all_genes = set()
            for _, row in reg_strata.iterrows():
                # Skip rows with NaN clique_genes
                if pd.isna(row['clique_genes']):
                    continue
                genes = [g.strip() for g in str(row['clique_genes']).split(',')]
                all_genes.update(genes)

            total_genes = len(all_genes)

            reg_metrics.append({
                'regulator': regulator,
                'rewiring_score': rewiring_score,
                'max_coherence': max_coherence,
                'total_genes': total_genes,
                'sex_pattern': sex_pattern,
                'male_rewiring': male_rewiring,
                'female_rewiring': female_rewiring
            })

        # Convert to DataFrame
        plot_df = pd.DataFrame(reg_metrics)

        # Filter by minimum rewiring score
        plot_df = plot_df[np.abs(plot_df['rewiring_score']) >= min_rewiring_score]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Color mapping for sex patterns
        color_map = {
            'both': '#7c3aed',      # Purple (both sexes)
            'male': '#0d9488',      # Teal (male-specific)
            'female': '#9333ea',    # Violet (female-specific)
            'none': '#9ca3af'       # Gray (no significant rewiring)
        }

        # Size mapping: scale by total genes
        # Use sqrt scaling for better visual discrimination
        size_scale = 100
        sizes = np.sqrt(plot_df['total_genes']) * size_scale

        # Plot each sex pattern separately for legend control
        for pattern, color in color_map.items():
            mask = plot_df['sex_pattern'] == pattern
            subset = plot_df[mask]

            if len(subset) == 0:
                continue

            label_map = {
                'both': 'Both sexes',
                'male': 'Male-specific',
                'female': 'Female-specific',
                'none': 'Weak rewiring'
            }

            ax.scatter(
                subset['rewiring_score'],
                subset['max_coherence'],
                s=np.sqrt(subset['total_genes']) * size_scale,
                c=color,
                alpha=point_alpha,
                edgecolors='white',
                linewidths=0.5,
                label=label_map[pattern],
                zorder=2 if pattern != 'none' else 1
            )

        # Add quadrant reference lines
        ax.axvline(0, color='#d1d5db', linestyle='--', linewidth=1, zorder=0)
        ax.axhline(plot_df['max_coherence'].median(), color='#d1d5db',
                   linestyle='--', linewidth=1, zorder=0)

        # Add quadrant labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        quad_fontsize = 10
        quad_color = '#9ca3af'
        quad_alpha = 0.6

        # Top-right: gained + high coherence
        ax.text(
            xlim[1] * 0.95, ylim[1] * 0.95,
            'High-quality\ngains',
            ha='right', va='top',
            fontsize=quad_fontsize,
            color=quad_color,
            alpha=quad_alpha,
            style='italic'
        )

        # Top-left: lost + high coherence
        ax.text(
            xlim[0] * 0.95, ylim[1] * 0.95,
            'High-quality\nlosses',
            ha='left', va='top',
            fontsize=quad_fontsize,
            color=quad_color,
            alpha=quad_alpha,
            style='italic'
        )

        # Label top regulators
        # Sort by absolute rewiring score
        plot_df_sorted = plot_df.sort_values('rewiring_score',
                                              key=lambda x: np.abs(x),
                                              ascending=False)

        # Use adjustText if available, otherwise simple labels
        try:
            from adjustText import adjust_text
            texts = []

            for _, row in plot_df_sorted.head(label_top_n).iterrows():
                txt = ax.text(
                    row['rewiring_score'],
                    row['max_coherence'],
                    italicize_gene(row['regulator']),
                    fontsize=8,
                    ha='center',
                    va='center'
                )
                texts.append(txt)

            # Adjust text positions to avoid overlap
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle='-', color='#6b7280', lw=0.5, alpha=0.5),
                ax=ax
            )
        except ImportError:
            # Simple labeling without adjustment
            for _, row in plot_df_sorted.head(label_top_n).iterrows():
                ax.text(
                    row['rewiring_score'],
                    row['max_coherence'],
                    italicize_gene(row['regulator']),
                    fontsize=7,
                    ha='left',
                    va='bottom',
                    alpha=0.8
                )

        # Axis labels
        ax.set_xlabel(
            'Rewiring Score\n← Lost in ALS     |     Gained in ALS →',
            fontsize=11,
            fontweight='bold'
        )
        ax.set_ylabel(
            'Max Coherence Ratio\n(regulatory quality)',
            fontsize=11,
            fontweight='bold'
        )

        # Title
        ax.set_title(
            'Regulator Rewiring Overview: Scan for ALS-Associated Changes',
            fontsize=13,
            fontweight='bold',
            pad=15
        )

        # Legend
        # Create custom legend with size reference
        handles, labels = ax.get_legend_handles_labels()

        # Add size reference
        from matplotlib.lines import Line2D
        size_ref_handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#6b7280', markersize=np.sqrt(10) * np.sqrt(size_scale) / 10,
                   markeredgecolor='white', markeredgewidth=0.5, label='10 genes'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#6b7280', markersize=np.sqrt(20) * np.sqrt(size_scale) / 10,
                   markeredgecolor='white', markeredgewidth=0.5, label='20 genes'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#6b7280', markersize=np.sqrt(30) * np.sqrt(size_scale) / 10,
                   markeredgecolor='white', markeredgewidth=0.5, label='30 genes'),
        ]

        # Combine legends
        legend1 = ax.legend(
            handles=handles,
            labels=labels,
            title='Sex Pattern',
            loc='upper left',
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=0.95,
            edgecolor='#e5e7eb'
        )

        ax.add_artist(legend1)

        ax.legend(
            handles=size_ref_handles,
            title='Clique Size',
            loc='lower left',
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=0.95,
            edgecolor='#e5e7eb'
        )

        # Grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add caption
        caption = (
            f"Showing {len(plot_df)} regulators with |rewiring_score| ≥ {min_rewiring_score}. "
            f"Positive rewiring = cliques gained in ALS; negative = lost in ALS. "
            f"Coherence ratio measures regulatory quality."
        )
        fig.text(
            0.5, 0.01,
            caption,
            ha='center',
            fontsize=9,
            style='italic',
            color='#6b7280',
            wrap=True
        )

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Regulator Rewiring Overview",
            description=(
                "Overview scatter plot of all regulators showing rewiring patterns "
                "in ALS disease. X-axis shows rewiring score (gained vs lost cliques), "
                "Y-axis shows regulatory quality (coherence), point size shows clique size, "
                "and color indicates sex-specific vs pan-sex effects."
            ),
            figure_type="matplotlib",
            metadata={
                'n_regulators': len(plot_df),
                'min_rewiring_score': min_rewiring_score,
                'n_both_sexes': len(plot_df[plot_df['sex_pattern'] == 'both']),
                'n_male_specific': len(plot_df[plot_df['sex_pattern'] == 'male']),
                'n_female_specific': len(plot_df[plot_df['sex_pattern'] == 'female']),
                'n_weak': len(plot_df[plot_df['sex_pattern'] == 'none']),
            }
        )
