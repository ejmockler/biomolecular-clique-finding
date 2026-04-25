"""Degree-preserving edge rewiring for network null models.

Maslov-Sneppen double-edge-swap implementation with an empirical
mixing diagnostic (Ray-Pinar-Seshadhri 2012 stopping criterion).
Used as a seed-position null for the gradient discovery framework
(:mod:`cliquefinder.stats.perturbation_gradient`).

Design constraints (see memory/edge_rewiring_design.md v2):

- Global rewiring over the full connected component containing the seed
  (local rewiring biases the null via boundary edge freezing and
  shortcut creation).
- Empirical mixing diagnostic instead of fixed ``nswap = k * |E|`` —
  Milo et al. 2003 explicitly disclaims any general mixing-time bound
  for the switching method on heavy-tailed degree sequences.
- Deterministic under ``SeedSequence.spawn``; per-iteration RNGs are
  independent.
- All operations are pure graph ops (no INDRA client, no statistics);
  the statistical orchestration is :mod:`perturbation_gradient`.

References
----------
- Maslov & Sneppen 2002, Science 296:910 — original null for biological networks.
- Milo et al. 2003, arXiv cond-mat/0312028 — mixing caveats.
- Ray, Pinar & Seshadhri 2012, arXiv 1202.3473 — empirical stopping criterion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixingDiagnostic:
    """Trace of a Maslov-Sneppen mixing run.

    Attributes
    ----------
    swap_counts
        Swap iteration checkpoints (cumulative accepted swaps).
    distances_from_origin
        Hamming distance (edge-set symmetric difference / 2) from the
        original graph at each checkpoint.
    plateau_swaps
        Swap count at which the distance trajectory plateaued (mixing
        declared adequate).  ``None`` if the trajectory never plateaued
        within the allotted budget.
    accepted_fraction
        Accepted-swap / attempted-swap ratio at convergence.
    """

    swap_counts: tuple[int, ...]
    distances_from_origin: tuple[int, ...]
    plateau_swaps: int | None
    accepted_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "swap_counts": list(self.swap_counts),
            "distances_from_origin": list(self.distances_from_origin),
            "plateau_swaps": self.plateau_swaps,
            "accepted_fraction": float(self.accepted_fraction),
        }


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def compute_undirected_degrees(graph: nx.Graph) -> dict[str, int]:
    """Return ``{node: degree}`` for an undirected networkx graph."""
    return dict(graph.degree())


def bfs_distances_from(
    graph: nx.Graph,
    seed: str,
    targets: set[str],
    max_hops: int,
) -> dict[str, int]:
    """BFS from seed, return shortest-path distance to each target, early
    terminating at ``max_hops``.

    Parameters
    ----------
    graph
        Undirected graph.
    seed
        Source node.
    targets
        Nodes whose distances we care about.
    max_hops
        Maximum depth; unreachable-within-max-hops targets are omitted.

    Returns
    -------
    ``{target: distance}`` for each target reachable within ``max_hops``.
    """
    if seed not in graph:
        return {}

    targets_remaining = set(targets) - {seed}
    distances: dict[str, int] = {}
    visited = {seed}
    frontier = {seed}

    for hop in range(1, max_hops + 1):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in graph.neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.add(neighbor)
                if neighbor in targets_remaining:
                    distances[neighbor] = hop
                    targets_remaining.discard(neighbor)

        if not next_frontier or not targets_remaining:
            break
        frontier = next_frontier

    return distances


def _edge_set_distance(edges_a: set[frozenset], edges_b: set[frozenset]) -> int:
    """Edge-set Hamming distance (symmetric difference // 2).

    Each edge disagreement contributes one edge to each side of the
    symmetric difference under a simple-graph model; we halve.
    """
    return len(edges_a ^ edges_b) // 2


def _snapshot_edges(graph: nx.Graph) -> set[frozenset]:
    return {frozenset(e) for e in graph.edges()}


def _plateau_detected(
    distances: list[int],
    tolerance_pct: float = 0.02,
    window: int = 3,
    min_checkpoints: int = 5,
) -> bool:
    """Declare the mixing trajectory plateaued.

    Three conditions ALL must hold:

    1. At least ``min_checkpoints`` checkpoints have been recorded
       (avoids firing during the rising transient).
    2. The last ``window`` distances differ by less than
       ``tolerance_pct`` of their max (local stability).
    3. The minimum of the recent window is at least half of the maximum
       distance observed across the entire trajectory so far (saturation
       check — we're not in the slow-rise phase).
    """
    if len(distances) < max(window, min_checkpoints):
        return False
    recent = distances[-window:]
    span = max(recent) - min(recent)
    local_scale = max(recent)
    global_max = max(distances)
    if local_scale == 0:
        return False
    # Local stability
    if span > tolerance_pct * local_scale:
        return False
    # Saturation: recent minimum is at least 50% of global max
    if min(recent) < 0.5 * global_max:
        return False
    return True


def rewire_maslov_sneppen(
    graph: nx.Graph,
    rng: np.random.Generator,
    target_nswap: int | None = None,
    max_swaps: int = 1_000_000,
    check_every: int = 5000,
    tolerance_pct: float = 0.02,
    plateau_window: int = 3,
    min_plateau_checkpoints: int = 5,
    safety_margin: float = 0.2,
) -> tuple[nx.Graph, MixingDiagnostic]:
    """Rewire a graph via Maslov-Sneppen double-edge swaps.

    Two modes:

    - **Diagnostic mode** (``target_nswap=None``): run until the edge-set
      Hamming distance from the original plateaus, then safety margin,
      then return. Records the full mixing trajectory.
    - **Fixed mode** (``target_nswap=N``): run exactly N accepted swaps,
      no plateau detection, no trajectory recording. Use this for
      subsequent iterations after an initial diagnostic run has
      determined an adequate swap count.

    In diagnostic mode, plateau is declared only after at least
    ``min_plateau_checkpoints`` have been recorded AND the recent window
    is stable AND the minimum distance in the window is at least half
    the max distance seen (saturation check, avoiding false plateau
    during the rising-transient phase).

    Performs accepted-swap counting; rejected attempts counted in
    ``accepted_fraction``.

    Parameters
    ----------
    graph
        Undirected networkx graph. Copied; input is not modified.
    rng
        Numpy random generator for reproducible randomness.
    target_nswap
        If set, run exactly this many accepted swaps. Plateau detection
        disabled. Overrides ``max_swaps`` as the budget.
    max_swaps
        Hard ceiling (diagnostic mode only).
    check_every
        Accepted swaps between distance-from-origin checkpoints.
    tolerance_pct, plateau_window, min_plateau_checkpoints, safety_margin
        Plateau detection parameters (diagnostic mode only).

    Returns
    -------
    (rewired_graph, diagnostic)
    """
    G = graph.copy()
    diagnostic_mode = target_nswap is None
    original_edges = _snapshot_edges(graph) if diagnostic_mode else None

    swap_counts: list[int] = []
    distances: list[int] = []
    accepted_total = 0
    attempted_total = 0
    plateau_swaps: int | None = None
    swaps_after_plateau = 0

    edges_list = list(G.edges())
    n_edges = len(edges_list)
    if n_edges < 4:
        logger.warning(
            "Graph too small for meaningful Maslov-Sneppen (|E|=%d)", n_edges
        )
        return G, MixingDiagnostic(
            swap_counts=(0,), distances_from_origin=(0,),
            plateau_swaps=0, accepted_fraction=1.0,
        )

    budget = target_nswap if target_nswap is not None else max_swaps
    # Hard ceiling on attempts to prevent infinite spin when acceptance
    # rate decays toward 0 (heavy-tail saturation). 100x budget is generous —
    # at acceptance ≥1% the chain still completes; at lower rates the null
    # is degenerate anyway and we'd rather fail fast than hang for hours.
    attempt_ceiling = 100 * budget

    while accepted_total < budget:
        if attempted_total >= attempt_ceiling:
            logger.warning(
                "Maslov-Sneppen aborted: %d attempts produced only %d accepted "
                "swaps (target %d). Acceptance rate %.4f%% — graph is likely "
                "saturated. Returning partial rewire.",
                attempted_total, accepted_total, budget,
                100.0 * accepted_total / max(attempted_total, 1),
            )
            break
        # Pick two distinct edges without replacement
        idx = rng.choice(n_edges, size=2, replace=False)
        i, j = int(idx[0]), int(idx[1])

        e1 = edges_list[i]
        e2 = edges_list[j]
        a, b = e1
        c, d = e2

        # Randomize swap orientation
        if rng.random() < 0.5:
            new1, new2 = (a, d), (c, b)
        else:
            new1, new2 = (a, c), (b, d)

        attempted_total += 1

        # Skip if would create self-loop or duplicate edge
        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue
        if G.has_edge(*new1) or G.has_edge(*new2):
            continue
        if frozenset(new1) == frozenset(new2):
            continue

        # Perform swap
        G.remove_edge(*e1)
        G.remove_edge(*e2)
        G.add_edge(*new1)
        G.add_edge(*new2)
        edges_list[i] = new1
        edges_list[j] = new2

        accepted_total += 1
        if plateau_swaps is not None:
            swaps_after_plateau += 1

        # Diagnostic mode: check plateau
        if diagnostic_mode and accepted_total % check_every == 0:
            current_edges = _snapshot_edges(G)
            dist = _edge_set_distance(original_edges, current_edges)
            swap_counts.append(accepted_total)
            distances.append(dist)

            if plateau_swaps is None and _plateau_detected(
                distances, tolerance_pct, plateau_window,
                min_checkpoints=min_plateau_checkpoints,
            ):
                plateau_swaps = accepted_total
                swaps_after_plateau = 0
                logger.info(
                    "Mixing plateau at %d accepted swaps (d=%d); "
                    "running +%.0f%% safety margin",
                    accepted_total, dist, 100 * safety_margin,
                )

            if plateau_swaps is not None:
                target_extra = int(plateau_swaps * safety_margin)
                if swaps_after_plateau >= target_extra:
                    break

    accepted_fraction = (
        accepted_total / attempted_total if attempted_total > 0 else 0.0
    )
    if diagnostic_mode and plateau_swaps is None:
        logger.warning(
            "Maslov-Sneppen did not detect mixing plateau within %d swaps; "
            "null distribution may be under-mixed",
            max_swaps,
        )

    diagnostic = MixingDiagnostic(
        swap_counts=tuple(swap_counts),
        distances_from_origin=tuple(distances),
        plateau_swaps=plateau_swaps if diagnostic_mode else accepted_total,
        accepted_fraction=accepted_fraction,
    )
    return G, diagnostic


# ---------------------------------------------------------------------------
# Subgraph extraction helper (wraps Cypher results)
# ---------------------------------------------------------------------------


def edges_to_undirected_graph(
    edges: list[tuple[str, str, Any]] | list[tuple[str, str]],
) -> nx.Graph:
    """Build an undirected networkx graph from an edge list.

    Parameters
    ----------
    edges
        Iterable of (source, target) or (source, target, meta) tuples.
        Metadata is ignored in this pure-topology representation.

    Returns
    -------
    Undirected simple graph.  Multi-edges between the same node pair are
    deduplicated; self-loops are preserved (caller may filter).
    """
    G = nx.Graph()
    for e in edges:
        if len(e) >= 2:
            u, v = e[0], e[1]
            if u != v:
                G.add_edge(u, v)
    return G


def seed_component(
    graph: nx.Graph,
    seed: str,
) -> nx.Graph:
    """Return the connected component of ``graph`` containing ``seed``.

    Returns an empty graph if seed is not in the graph.
    """
    if seed not in graph:
        return nx.Graph()
    component_nodes = nx.node_connected_component(graph, seed)
    return graph.subgraph(component_nodes).copy()


# ---------------------------------------------------------------------------
# Pathology diagnostics
# ---------------------------------------------------------------------------


def bimodality_coefficient(values: NDArray[np.float64]) -> tuple[float, float]:
    """Compute Pfister-style bimodality coefficient for a 1D sample.

    Returns ``(coefficient, warning_pseudo_pvalue)``.  Coefficient is

    .. math::
        b = \\frac{g^2 + 1}{k + \\frac{3(n-1)^2}{(n-2)(n-3)}}

    where ``g`` is Fisher's skewness (bias-corrected) and ``k`` is
    Pearson's excess kurtosis (bias-corrected).  Values of ``b > 5/9``
    (~0.555) suggest bimodality; the uniform-distribution coefficient
    is exactly ``5/9``.

    The "pseudo p-value" is a warning-scale value:
    ``max(0, 1 - 2*max(0, b - 5/9))`` — not a proper Monte Carlo
    p-value, but serves as a threshold for the runtime warning in
    :func:`run_rewiring_null`.  Treat ``coefficient > 0.555`` as "check
    mixing" rather than a formal hypothesis test.

    References
    ----------
    Pfister, R., Schwarz, K. A., Janczyk, M., Dale, R., & Freeman, J. B.
    (2013).  Good things peak in pairs: a note on the bimodality
    coefficient.  Frontiers in Psychology, 4:700.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 4:
        return 0.0, 1.0

    mean = np.mean(x)
    sd = np.std(x, ddof=1)
    if sd == 0:
        return 0.0, 1.0

    # Fisher's skewness (bias-corrected, sample)
    m3 = np.mean((x - mean) ** 3)
    g = (m3 / sd**3) * np.sqrt(n * (n - 1)) / (n - 2)

    # Pearson's excess kurtosis (bias-corrected, sample)
    m4 = np.mean((x - mean) ** 4)
    k_raw = m4 / sd**4 - 3
    # Bias correction
    k = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * k_raw + 6)

    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom <= 0:
        return 0.0, 1.0
    b = (g**2 + 1) / denom

    # Pseudo p-value: coefficient above 5/9 threshold maps to low value
    # (triggering the bimodality warning in run_rewiring_null)
    threshold = 5.0 / 9.0
    pseudo_p = max(0.0, 1.0 - 2.0 * max(0.0, b - threshold))
    return float(b), float(pseudo_p)


# Backward-compatible alias (name changed for accuracy: this is not
# Hartigan's dip, it's the bimodality coefficient).
hartigan_dip_test = bimodality_coefficient


def disconnection_rate(
    rewired_graphs_or_components: list[set[str]],
    targets: set[str],
    required_fraction: float = 0.5,
) -> float:
    """Fraction of rewired graphs where the seed's component contains
    fewer than ``required_fraction`` of ``targets``.

    Parameters
    ----------
    rewired_graphs_or_components
        One set per rewiring: the set of node IDs in the seed's connected
        component after rewiring.
    targets
        Set of target nodes we need for shell computation.
    required_fraction
        Threshold; if component covers < this fraction of targets, count
        as disconnected.

    Returns
    -------
    Fraction in [0, 1].
    """
    if not rewired_graphs_or_components or not targets:
        return 0.0

    n_disconnected = 0
    n_targets = len(targets)
    for component in rewired_graphs_or_components:
        covered = len(component & targets) / n_targets
        if covered < required_fraction:
            n_disconnected += 1

    return n_disconnected / len(rewired_graphs_or_components)
