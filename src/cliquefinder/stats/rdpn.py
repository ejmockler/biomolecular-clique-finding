"""Randomized Degree-Preserving Network (RDPN) null model for RWR.

Generates topology-aware null distributions of RWR scores by rewiring
the network while preserving the (in-degree, out-degree) joint sequence.
Produces per-gene z-scores that control for hub bias: high-degree nodes
naturally accumulate RWR probability regardless of disease relevance.

References
----------
- Biran et al. (2019). Controlling false-positive results in network-based
  gene prioritization. Bioinformatics.
- Maslov & Sneppen (2002). Specificity and stability in topology of
  protein networks. Science 296:910-913.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from cliquefinder.stats.network_proximity import compute_rwr_scores

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RDPNNull:
    """Null distribution from degree-preserving network rewirings.

    Attributes
    ----------
    mean_scores : ndarray of shape (n_nodes,)
        Mean RWR score across all rewirings per node.
    std_scores : ndarray of shape (n_nodes,)
        Std of RWR scores across all rewirings per node.
    n_rewirings : int
        Number of rewired networks generated.
    n_successful : int
        Number of rewirings that converged.
    """

    mean_scores: NDArray[np.float64]
    std_scores: NDArray[np.float64]
    n_rewirings: int
    n_successful: int


def degree_preserving_rewire(
    adjacency: sp.csr_matrix,
    n_swaps: int | None = None,
    rng: np.random.Generator | None = None,
) -> sp.csr_matrix:
    """Rewire a directed graph preserving in-degree and out-degree.

    Uses the Maslov-Sneppen edge-swap algorithm: pick two edges (u→v, x→y)
    at random, swap to (u→y, x→v) if neither new edge already exists.

    Parameters
    ----------
    adjacency
        Sparse adjacency matrix (directed, unweighted or weighted).
    n_swaps
        Number of swap attempts. Default: 10 × number of edges.
    rng
        Random number generator for reproducibility.

    Returns
    -------
    Rewired sparse adjacency matrix with same degree sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Extract edge list from sparse matrix (unweighted for topology-only rewiring)
    coo = adjacency.tocoo()
    rows = coo.row.copy()
    cols = coo.col.copy()
    n_edges = len(rows)
    n_nodes = adjacency.shape[0]

    if n_edges < 2:
        return adjacency.copy()

    if n_swaps is None:
        n_swaps = 10 * n_edges

    # Build edge set for O(1) existence checks
    edge_set = set(zip(rows, cols))

    n_successful = 0
    for _ in range(n_swaps):
        # Pick two random edge indices
        e1, e2 = rng.integers(0, n_edges, size=2)
        if e1 == e2:
            continue

        u, v = rows[e1], cols[e1]
        x, y = rows[e2], cols[e2]

        # Skip self-loops and duplicate targets
        if u == y or x == v:
            continue

        # Check new edges don't exist
        if (u, y) in edge_set or (x, v) in edge_set:
            continue

        # Perform swap (topology only — unweighted)
        edge_set.discard((u, v))
        edge_set.discard((x, y))
        edge_set.add((u, y))
        edge_set.add((x, v))

        cols[e1] = y
        cols[e2] = v

        n_successful += 1

    # Return unweighted adjacency — the null controls for topological
    # degree, not weighted degree. Using unweighted ensures the joint
    # (in,out)-degree sequence is exactly preserved.
    data = np.ones(n_edges, dtype=np.float64)
    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_nodes, n_nodes),
    )


def compute_rdpn_null(
    adjacency: sp.csr_matrix,
    seed_index: int,
    n_rewirings: int = 500,
    restart_prob: float = 0.15,
    tol: float = 1e-8,
    max_iter: int = 200,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> RDPNNull:
    """Generate null RWR distribution via degree-preserving rewiring.

    Parameters
    ----------
    adjacency
        Original sparse adjacency matrix.
    seed_index
        Index of the seed node.
    n_rewirings
        Number of rewired networks to generate.
    restart_prob
        RWR restart probability.
    rng
        Random number generator.

    Returns
    -------
    RDPNNull with mean and std of null RWR scores per node.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_nodes = adjacency.shape[0]
    null_scores = np.zeros((n_rewirings, n_nodes), dtype=np.float64)
    n_successful = 0

    for i in range(n_rewirings):
        rewired = degree_preserving_rewire(adjacency, rng=rng)
        scores, delta, n_iter = compute_rwr_scores(
            rewired, seed_index, restart_prob, tol, max_iter,
        )
        null_scores[i] = scores
        if delta < tol:
            n_successful += 1

        if verbose and (i + 1) % 20 == 0:
            logger.info(
                "  RDPN rewiring %d/%d (converged: %d)",
                i + 1, n_rewirings, n_successful,
            )

    return RDPNNull(
        mean_scores=null_scores.mean(axis=0),
        std_scores=null_scores.std(axis=0, ddof=1),
        n_rewirings=n_rewirings,
        n_successful=n_successful,
    )


def compute_rdpn_zscores(
    observed_scores: NDArray[np.float64],
    null: RDPNNull,
) -> NDArray[np.float64]:
    """Compute degree-deconfounded z-scores from RDPN null.

    Parameters
    ----------
    observed_scores
        RWR scores from the real graph, shape (n_nodes,).
    null
        RDPN null distribution from ``compute_rdpn_null()``.

    Returns
    -------
    z-scores, shape (n_nodes,). Zero-variance nodes get z=0.
    """
    z = np.zeros_like(observed_scores)
    nonzero_std = null.std_scores > 1e-10
    z[nonzero_std] = (
        (observed_scores[nonzero_std] - null.mean_scores[nonzero_std])
        / null.std_scores[nonzero_std]
    )
    return z
