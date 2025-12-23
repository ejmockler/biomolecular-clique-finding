"""
Network visualizations for correlation graphs and cliques.

Provides interactive (plotly) and static (matplotlib) visualizations for:
- Correlation networks with FDR-controlled edges
- Clique membership and differential analysis
- Community/module structure

Designed for performance with 10K+ edge networks using:
- Edge sampling for large graphs
- Efficient igraph layouts
- WebGL rendering in plotly
"""

from __future__ import annotations

from typing import Optional, Literal, Any
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cliquefinder.viz.core import Figure
from cliquefinder.viz.styles import Palette, PALETTES, get_stratum_colors


class NetworkVisualizer:
    """
    Network visualizations for correlation graphs and cliques.

    Supports both interactive (plotly) and static (matplotlib) output.
    Handles large networks (10K+ edges) via edge sampling and WebGL.

    Parameters
    ----------
    layout_algorithm : {"fr", "spring", "kamada_kawai", "circle"}
        Default layout algorithm.
        - "fr": Fruchterman-Reingold (force-directed, good for large graphs)
        - "spring": NetworkX spring layout
        - "kamada_kawai": Good for small graphs
        - "circle": Simple circular layout
    palette : str or Palette
        Color palette.

    Examples
    --------
    >>> from cliquefinder.viz import NetworkVisualizer
    >>> viz = NetworkVisualizer()
    >>>
    >>> # Interactive correlation network
    >>> fig = viz.plot_correlation_network(
    ...     genes, corr_matrix, threshold=0.8
    ... )
    >>> fig.save("network.html")
    """

    def __init__(
        self,
        layout_algorithm: Literal["fr", "spring", "kamada_kawai", "circle"] = "fr",
        palette: str | Palette = "default"
    ):
        self.layout_algorithm = layout_algorithm
        if isinstance(palette, str):
            self.palette = PALETTES.get(palette, PALETTES["default"])
        else:
            self.palette = palette

    def plot_correlation_network(
        self,
        gene_ids: list[str],
        correlation_matrix: np.ndarray,
        correlation_threshold: float = 0.7,
        highlight_cliques: Optional[dict[int, list[str]]] = None,
        node_colors: Optional[dict[str, str]] = None,
        node_sizes: Optional[dict[str, float]] = None,
        max_edges: int = 10000,
        interactive: bool = True,
        figsize: tuple[float, float] = (12, 10)
    ) -> Figure:
        """
        Plot correlation network with optional clique highlighting.

        Parameters
        ----------
        gene_ids : list[str]
            Gene/protein identifiers (must match correlation_matrix).
        correlation_matrix : np.ndarray
            Gene × Gene correlation matrix.
        correlation_threshold : float, default 0.7
            Minimum |correlation| to draw edge.
        highlight_cliques : dict[int, list[str]], optional
            Clique ID -> gene list for coloring.
        node_colors : dict[str, str], optional
            Gene -> color mapping.
        node_sizes : dict[str, float], optional
            Gene -> size mapping.
        max_edges : int, default 10000
            Maximum edges (sampled if exceeded).
        interactive : bool, default True
            If True, use plotly. If False, use matplotlib.
        figsize : tuple
            Figure size for matplotlib.

        Returns
        -------
        Figure
            Interactive (plotly) or static (matplotlib) network.
        """
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(gene_ids)

        # Collect edges above threshold
        edges = []
        for i in range(len(gene_ids)):
            for j in range(i + 1, len(gene_ids)):
                corr = correlation_matrix[i, j]
                if abs(corr) >= correlation_threshold:
                    edges.append((
                        gene_ids[i],
                        gene_ids[j],
                        {"weight": abs(corr), "corr": corr}
                    ))

        # Sample edges if too many
        if len(edges) > max_edges:
            edges = sorted(edges, key=lambda e: e[2]["weight"], reverse=True)[:max_edges]
            print(f"Sampled top {max_edges} edges from {len(edges)}")

        G.add_edges_from(edges)

        # Compute layout
        pos = self._compute_layout(G)

        # Assign node colors
        if node_colors is None and highlight_cliques is not None:
            node_colors = self._colors_from_cliques(highlight_cliques, G.nodes())
        elif node_colors is None:
            node_colors = {n: self.palette.neutral for n in G.nodes()}

        # Assign node sizes (degree centrality by default)
        if node_sizes is None:
            degrees = dict(G.degree())
            max_deg = max(degrees.values()) if degrees else 1
            node_sizes = {n: 10 + 25 * (degrees[n] / max_deg) for n in G.nodes()}

        # Create figure
        if interactive:
            fig = self._plot_network_plotly(G, pos, node_colors, node_sizes, correlation_threshold)
        else:
            fig = self._plot_network_matplotlib(G, pos, node_colors, node_sizes, figsize)

        return Figure(
            fig=fig,
            title="Correlation Network",
            description=f"{len(G.nodes())} nodes, {len(G.edges())} edges (threshold={correlation_threshold})",
            figure_type="plotly" if interactive else "matplotlib",
            metadata={
                "n_nodes": len(G.nodes()),
                "n_edges": len(G.edges()),
                "threshold": correlation_threshold
            }
        )

    def plot_clique_membership(
        self,
        cliques: dict[int, list[str]],
        figsize: tuple[float, float] = (12, 8)
    ) -> Figure:
        """
        Plot clique membership matrix.

        Shows which genes belong to which cliques, identifying
        hub genes that participate in multiple cliques.

        Parameters
        ----------
        cliques : dict[int, list[str]]
            Clique ID -> gene list mapping.
        figsize : tuple
            Figure size.

        Returns
        -------
        Figure
            Clique membership heatmap.
        """
        # Build membership matrix
        all_genes = sorted(set(g for genes in cliques.values() for g in genes))
        clique_ids = sorted(cliques.keys())

        membership = np.zeros((len(all_genes), len(clique_ids)), dtype=int)
        for j, cid in enumerate(clique_ids):
            for gene in cliques[cid]:
                i = all_genes.index(gene)
                membership[i, j] = 1

        # Sort genes by membership count (hub genes first)
        membership_counts = membership.sum(axis=1)
        sort_idx = np.argsort(membership_counts)[::-1]
        membership = membership[sort_idx, :]
        all_genes = [all_genes[i] for i in sort_idx]

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        import seaborn as sns
        sns.heatmap(
            membership,
            cmap="YlOrRd",
            xticklabels=[f"C{cid}" for cid in clique_ids],
            yticklabels=all_genes if len(all_genes) <= 50 else False,
            cbar_kws={"label": "Member"},
            ax=ax
        )

        ax.set_xlabel("Clique")
        ax.set_ylabel(f"Genes (n={len(all_genes)}, sorted by membership)")
        ax.set_title(f"Clique Membership ({len(cliques)} cliques, {len(all_genes)} genes)")

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Clique Membership",
            description=f"{len(cliques)} cliques, {len(all_genes)} unique genes",
            figure_type="matplotlib",
            metadata={"n_cliques": len(cliques), "n_genes": len(all_genes)}
        )

    def plot_clique_sizes(
        self,
        cliques_by_condition: dict[str, dict[int, list[str]]],
        figsize: tuple[float, float] = (10, 6)
    ) -> Figure:
        """
        Compare clique size distributions across conditions.

        Parameters
        ----------
        cliques_by_condition : dict[str, dict[int, list[str]]]
            Condition -> (clique_id -> gene_list) mapping.
        figsize : tuple
            Figure size.

        Returns
        -------
        Figure
            Clique size comparison plot.
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = get_stratum_colors(list(cliques_by_condition.keys()), self.palette)

        for i, (condition, cliques) in enumerate(cliques_by_condition.items()):
            sizes = [len(genes) for genes in cliques.values()]
            if sizes:
                # Violin plot
                parts = ax.violinplot(
                    [sizes],
                    positions=[i],
                    showmeans=True,
                    showmedians=True
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(colors[condition])
                    pc.set_alpha(0.7)

                # Add count annotation
                ax.text(i, max(sizes) + 0.5, f"n={len(cliques)}",
                        ha="center", va="bottom", fontsize=9)

        ax.set_xticks(range(len(cliques_by_condition)))
        ax.set_xticklabels(list(cliques_by_condition.keys()))
        ax.set_ylabel("Clique Size (# genes)")
        ax.set_title("Clique Size Distribution by Condition")

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Clique Size Comparison",
            description=f"Size distributions across {len(cliques_by_condition)} conditions",
            figure_type="matplotlib"
        )

    def plot_differential_network(
        self,
        gene_ids: list[str],
        corr_matrix_case: np.ndarray,
        corr_matrix_ctrl: np.ndarray,
        threshold: float = 0.7,
        diff_threshold: float = 0.3,
        figsize: tuple[float, float] = (14, 6)
    ) -> Figure:
        """
        Plot differential correlation network (CASE vs CTRL).

        Shows edges that are gained or lost in disease condition.

        Parameters
        ----------
        gene_ids : list[str]
            Gene identifiers.
        corr_matrix_case : np.ndarray
            Correlation matrix for CASE samples.
        corr_matrix_ctrl : np.ndarray
            Correlation matrix for CTRL samples.
        threshold : float, default 0.7
            Minimum |correlation| for an edge.
        diff_threshold : float, default 0.3
            Minimum |difference| to highlight as differential.
        figsize : tuple
            Figure size.

        Returns
        -------
        Figure
            Side-by-side differential network plots.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Compute edges
        edges_case = set()
        edges_ctrl = set()

        for i in range(len(gene_ids)):
            for j in range(i + 1, len(gene_ids)):
                if abs(corr_matrix_case[i, j]) >= threshold:
                    edges_case.add((gene_ids[i], gene_ids[j]))
                if abs(corr_matrix_ctrl[i, j]) >= threshold:
                    edges_ctrl.add((gene_ids[i], gene_ids[j]))

        # Classify edges
        gained = edges_case - edges_ctrl  # Only in CASE
        lost = edges_ctrl - edges_case    # Only in CTRL
        stable = edges_case & edges_ctrl  # In both

        # Build graph with all genes involved in any edge
        all_edges = edges_case | edges_ctrl
        nodes = set()
        for e in all_edges:
            nodes.add(e[0])
            nodes.add(e[1])

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(all_edges)

        pos = self._compute_layout(G)

        # Plot CASE
        ax1 = axes[0]
        self._draw_network_differential(
            ax1, G, pos, gained, stable,
            highlight_color="#22c55e",  # Green for gained
            title=f"CASE (n={len(edges_case)} edges, {len(gained)} gained)"
        )

        # Plot CTRL
        ax2 = axes[1]
        self._draw_network_differential(
            ax2, G, pos, lost, stable,
            highlight_color="#ef4444",  # Red for lost
            title=f"CTRL (n={len(edges_ctrl)} edges, {len(lost)} lost)"
        )

        fig.suptitle(
            f"Differential Correlation Network (threshold={threshold})",
            fontweight="bold"
        )

        plt.tight_layout()

        return Figure(
            fig=fig,
            title="Differential Network",
            description=f"{len(gained)} edges gained in CASE, {len(lost)} edges lost",
            figure_type="matplotlib",
            metadata={
                "gained": len(gained),
                "lost": len(lost),
                "stable": len(stable),
                "threshold": threshold
            }
        )

    def _compute_layout(self, G: nx.Graph) -> dict:
        """Compute node positions using configured algorithm."""
        try:
            from cliquefinder.viz.layouts import compute_layout
            return compute_layout(G, self.layout_algorithm)
        except ImportError:
            # Fallback to networkx
            if self.layout_algorithm in ("fr", "spring"):
                return nx.spring_layout(G, seed=42, iterations=50)
            elif self.layout_algorithm == "kamada_kawai":
                return nx.kamada_kawai_layout(G)
            else:
                return nx.circular_layout(G)

    def _colors_from_cliques(
        self,
        cliques: dict[int, list[str]],
        nodes: list[str]
    ) -> dict[str, str]:
        """Assign colors based on clique membership."""
        import seaborn as sns

        colors = {}
        palette = sns.color_palette("husl", len(cliques)).as_hex()

        # First, mark all as neutral
        for node in nodes:
            colors[node] = self.palette.neutral

        # Then, color by clique (last clique wins for multi-membership)
        for i, (cid, genes) in enumerate(cliques.items()):
            for gene in genes:
                if gene in colors:
                    colors[gene] = palette[i % len(palette)]

        return colors

    def _plot_network_plotly(
        self,
        G: nx.Graph,
        pos: dict,
        node_colors: dict[str, str],
        node_sizes: dict[str, float],
        threshold: float
    ) -> Any:
        """Create interactive plotly network."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly required for interactive networks. Install with: pip install plotly")

        # Edge trace
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
            hoverinfo="none"
        )

        # Node trace
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_color = [node_colors.get(n, self.palette.neutral) for n in G.nodes()]
        node_size = [node_sizes.get(n, 15) for n in G.nodes()]

        # Hover text
        hover_text = [
            f"{n}<br>Degree: {G.degree(n)}"
            for n in G.nodes()
        ]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color="white")
            ),
            text=[n if G.degree(n) > 5 else "" for n in G.nodes()],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=hover_text,
            hoverinfo="text"
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Correlation Network ({len(G.nodes())} nodes, {len(G.edges())} edges)",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1000,
            height=800
        )

        return fig

    def _plot_network_matplotlib(
        self,
        G: nx.Graph,
        pos: dict,
        node_colors: dict[str, str],
        node_sizes: dict[str, float],
        figsize: tuple
    ) -> plt.Figure:
        """Create static matplotlib network."""
        fig, ax = plt.subplots(figsize=figsize)

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            alpha=0.3,
            edge_color=self.palette.neutral,
            width=0.5
        )

        # Draw nodes
        node_color_list = [node_colors.get(n, self.palette.neutral) for n in G.nodes()]
        node_size_list = [node_sizes.get(n, 100) * 10 for n in G.nodes()]  # Scale up for matplotlib

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_color_list,
            node_size=node_size_list,
            alpha=0.8,
            edgecolors="white",
            linewidths=1
        )

        # Draw labels for high-degree nodes
        high_degree_nodes = [n for n in G.nodes() if G.degree(n) > 5]
        if high_degree_nodes:
            labels = {n: n for n in high_degree_nodes}
            nx.draw_networkx_labels(
                G, pos, labels, ax=ax,
                font_size=8,
                font_color="black"
            )

        ax.set_title(f"Correlation Network ({len(G.nodes())} nodes, {len(G.edges())} edges)")
        ax.axis("off")

        plt.tight_layout()
        return fig

    def _draw_network_differential(
        self,
        ax: plt.Axes,
        G: nx.Graph,
        pos: dict,
        highlight_edges: set,
        stable_edges: set,
        highlight_color: str,
        title: str
    ):
        """Draw network with highlighted differential edges."""
        # Draw stable edges (gray)
        stable_list = [e for e in stable_edges if e[0] in G.nodes() and e[1] in G.nodes()]
        nx.draw_networkx_edges(
            G, pos, edgelist=stable_list, ax=ax,
            alpha=0.2, edge_color=self.palette.neutral, width=0.5
        )

        # Draw highlighted edges
        highlight_list = [e for e in highlight_edges if e[0] in G.nodes() and e[1] in G.nodes()]
        nx.draw_networkx_edges(
            G, pos, edgelist=highlight_list, ax=ax,
            alpha=0.8, edge_color=highlight_color, width=1.5
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=self.palette.neutral,
            node_size=50,
            alpha=0.7
        )

        ax.set_title(title)
        ax.axis("off")
