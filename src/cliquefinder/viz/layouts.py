"""
Fast network layout algorithms.

Provides optimized layout computation using igraph (10-100x faster than networkx)
with fallback to networkx for compatibility.

Supported Algorithms
--------------------
- fr (Fruchterman-Reingold): Force-directed, best for large graphs
- spring: NetworkX spring layout (slower)
- kamada_kawai: Energy-based, good for small graphs
- circle: Simple circular layout
"""

from __future__ import annotations

from typing import Literal
import numpy as np
import networkx as nx


def compute_layout(
    G: nx.Graph,
    algorithm: Literal["fr", "spring", "kamada_kawai", "circle"] = "fr",
    seed: int = 42
) -> dict[str, tuple[float, float]]:
    """
    Compute network layout positions using fast algorithms.

    For large networks (>500 nodes), automatically uses igraph which is
    10-100x faster than networkx implementations.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    algorithm : {"fr", "spring", "kamada_kawai", "circle"}
        Layout algorithm:
        - "fr": Fruchterman-Reingold (force-directed, O(n²) but parallelized in igraph)
        - "spring": NetworkX spring layout (slower)
        - "kamada_kawai": Energy-based (good for small graphs)
        - "circle": Simple circular layout
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, tuple[float, float]]
        Node -> (x, y) position mapping.

    Examples
    --------
    >>> import networkx as nx
    >>> from cliquefinder.viz.layouts import compute_layout
    >>>
    >>> G = nx.erdos_renyi_graph(1000, 0.01)
    >>> pos = compute_layout(G, algorithm="fr")
    >>> # pos["node_id"] returns (x, y) coordinates
    """
    np.random.seed(seed)

    if len(G.nodes()) == 0:
        return {}

    # Use igraph for large graphs with FR layout
    if len(G.nodes()) > 500 and algorithm == "fr":
        try:
            return _layout_igraph(G, seed)
        except ImportError:
            pass  # Fall through to networkx

    # NetworkX fallbacks
    if algorithm in ("fr", "spring"):
        return nx.spring_layout(G, seed=seed, iterations=50)
    elif algorithm == "kamada_kawai":
        try:
            return nx.kamada_kawai_layout(G)
        except nx.NetworkXError:
            # Fallback for disconnected graphs
            return nx.spring_layout(G, seed=seed)
    elif algorithm == "circle":
        return nx.circular_layout(G)
    else:
        raise ValueError(f"Unknown layout algorithm: {algorithm}")


def _layout_igraph(G: nx.Graph, seed: int = 42) -> dict[str, tuple[float, float]]:
    """
    Compute layout using igraph (much faster for large graphs).

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Node -> position mapping.
    """
    import igraph as ig

    # Convert networkx to igraph
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Create igraph graph
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(nodes))

    # Add edges
    edges = [
        (node_to_idx[u], node_to_idx[v])
        for u, v in G.edges()
        if u in node_to_idx and v in node_to_idx
    ]
    ig_graph.add_edges(edges)

    # Compute layout
    # FR layout with grid optimization for speed
    layout = ig_graph.layout_fruchterman_reingold(
        niter=500,
        seed=seed,
        grid="auto"  # Use grid acceleration for large graphs
    )

    # Convert back to dict
    return {
        node: (layout[node_to_idx[node]][0], layout[node_to_idx[node]][1])
        for node in nodes
    }


def layout_with_communities(
    G: nx.Graph,
    communities: dict[str, int],
    algorithm: str = "fr",
    seed: int = 42
) -> dict[str, tuple[float, float]]:
    """
    Compute layout that respects community structure.

    Places nodes in same community closer together.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    communities : dict[str, int]
        Node -> community_id mapping.
    algorithm : str
        Base layout algorithm.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, tuple[float, float]]
        Node -> position mapping.
    """
    np.random.seed(seed)

    # First, compute standard layout
    base_pos = compute_layout(G, algorithm, seed)

    if not communities:
        return base_pos

    # Group nodes by community
    community_groups = {}
    for node, comm_id in communities.items():
        if comm_id not in community_groups:
            community_groups[comm_id] = []
        community_groups[comm_id].append(node)

    # Compute community centroids
    centroids = {}
    for comm_id, nodes in community_groups.items():
        positions = [base_pos[n] for n in nodes if n in base_pos]
        if positions:
            centroids[comm_id] = (
                np.mean([p[0] for p in positions]),
                np.mean([p[1] for p in positions])
            )

    # Pull nodes toward their community centroid
    pull_factor = 0.3
    adjusted_pos = {}

    for node, pos in base_pos.items():
        if node in communities:
            comm_id = communities[node]
            if comm_id in centroids:
                cx, cy = centroids[comm_id]
                x, y = pos
                adjusted_pos[node] = (
                    x + pull_factor * (cx - x),
                    y + pull_factor * (cy - y)
                )
            else:
                adjusted_pos[node] = pos
        else:
            adjusted_pos[node] = pos

    return adjusted_pos


def hierarchical_layout(
    G: nx.Graph,
    root: str | None = None,
    seed: int = 42
) -> dict[str, tuple[float, float]]:
    """
    Compute hierarchical layout for directed graphs.

    Useful for regulatory networks (TF -> target relationships).

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph (can be directed).
    root : str, optional
        Root node. If None, uses node with highest degree.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, tuple[float, float]]
        Node -> position mapping.
    """
    if len(G.nodes()) == 0:
        return {}

    # Find root if not specified
    if root is None:
        degrees = dict(G.degree())
        root = max(degrees, key=degrees.get)

    # Try hierarchical layout
    try:
        # BFS from root to get levels
        levels = {root: 0}
        visited = {root}
        queue = [root]

        while queue:
            node = queue.pop(0)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    levels[neighbor] = levels[node] + 1
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Handle disconnected nodes
        for node in G.nodes():
            if node not in levels:
                levels[node] = max(levels.values()) + 1

        # Position by level
        level_counts = {}
        for node, level in levels.items():
            if level not in level_counts:
                level_counts[level] = 0
            level_counts[level] += 1

        pos = {}
        level_current = {level: 0 for level in level_counts}

        for node, level in levels.items():
            count = level_counts[level]
            idx = level_current[level]
            level_current[level] += 1

            x = (idx - count / 2 + 0.5) / max(count, 1)
            y = -level  # Top to bottom

            pos[node] = (x, y)

        return pos

    except Exception:
        # Fallback to spring layout
        return nx.spring_layout(G, seed=seed)
