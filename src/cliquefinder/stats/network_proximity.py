"""Network proximity tests for knowledge-graph-guided enrichment.

Three parameter-free tests that use continuous network proximity scores
instead of binary gene set membership. Each produces a single p-value,
eliminating the set-size / null-width problem that plagues competitive
enrichment tests on large gene sets.

Tests
-----
1. **Proximity decay**: Shortest path distance from seed gene predicts
   differential expression magnitude (Spearman correlation with
   degree-preserving permutation null).

2. **Reverse causal reasoning**: Starting from ALL differentially expressed
   genes (not a pre-specified set), identify upstream regulators whose
   known targets are concordantly enriched. Tests whether the seed gene
   emerges as a significant regulator without ever being pre-specified.

3. **RWR correlation**: Random Walk with Restart from seed gene on the
   INDRA graph produces proximity scores that correlate with |t-statistics|
   (Spearman correlation with gene-label permutation null).

References
----------
- Guney et al. (2016). Network-based in silico drug efficacy screening.
  Nature Communications 7:10331. doi:10.1038/ncomms10331
- Cowen et al. (2017). Network propagation: a universal amplifier of
  genetic associations. Nature Reviews Genetics 18:551-562.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProximityDecayResult:
    """Result of the network proximity decay test.

    Tests whether genes closer to the seed gene in the INDRA knowledge
    graph show stronger differential expression (higher |t-statistics|).
    """

    seed_gene: str
    spearman_rho: float
    spearman_pvalue: float
    permutation_pvalue: float
    n_permutations: int
    n_genes_reachable: int
    n_genes_unreachable: int
    distance_bins: dict[int, dict[str, float]]
    # {distance: {n_genes, mean_abs_t, median_abs_t, std_abs_t}}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "seed_gene": self.seed_gene,
            "spearman_rho": float(self.spearman_rho),
            "spearman_pvalue": float(self.spearman_pvalue),
            "permutation_pvalue": float(self.permutation_pvalue),
            "n_permutations": self.n_permutations,
            "n_genes_reachable": self.n_genes_reachable,
            "n_genes_unreachable": self.n_genes_unreachable,
            "distance_bins": {
                str(k): {kk: float(vv) for kk, vv in v.items()}
                for k, v in self.distance_bins.items()
            },
        }
        return d


@dataclass(frozen=True)
class ReverseCausalResult:
    """Result of reverse causal reasoning analysis.

    Starting from ALL differentially expressed genes, identifies upstream
    regulators whose INDRA targets are concordantly enriched. Reports
    whether the query gene (e.g. C9orf72) emerges as a significant
    regulator — without ever pre-specifying it as the seed.
    """

    query_gene: str
    query_gene_rank: int  # 1-indexed rank among all regulators
    query_gene_zscore: float
    query_gene_pvalue: float
    n_regulators_tested: int
    n_up_submitted: int
    n_down_submitted: int
    top_regulators: list[dict[str, Any]]  # Top K with scores
    regulator_scores: dict[str, float]  # All regulator z-scores

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_gene": self.query_gene,
            "query_gene_rank": self.query_gene_rank,
            "query_gene_zscore": float(self.query_gene_zscore),
            "query_gene_pvalue": float(self.query_gene_pvalue),
            "n_regulators_tested": self.n_regulators_tested,
            "n_up_submitted": self.n_up_submitted,
            "n_down_submitted": self.n_down_submitted,
            "top_regulators": self.top_regulators,
        }


@dataclass(frozen=True)
class RWRCorrelationResult:
    """Result of Random Walk with Restart correlation test.

    Tests whether RWR proximity to the seed gene on the INDRA graph
    correlates with differential expression magnitude.
    """

    seed_gene: str
    spearman_rho: float
    spearman_pvalue: float
    permutation_pvalue: float
    n_permutations: int
    n_genes: int
    restart_probability: float
    n_graph_nodes: int
    n_graph_edges: int
    convergence_delta: float
    n_iterations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_gene": self.seed_gene,
            "spearman_rho": float(self.spearman_rho),
            "spearman_pvalue": float(self.spearman_pvalue),
            "permutation_pvalue": float(self.permutation_pvalue),
            "n_permutations": self.n_permutations,
            "n_genes": self.n_genes,
            "restart_probability": float(self.restart_probability),
            "n_graph_nodes": self.n_graph_nodes,
            "n_graph_edges": self.n_graph_edges,
            "convergence_delta": float(self.convergence_delta),
            "n_iterations": self.n_iterations,
        }


@dataclass(frozen=True)
class SignedRWRResult:
    """Result of signed RWR propagation on INDRA directed graph.

    Runs RWR separately on activation and repression subgraphs to preserve
    edge semantics. Combined score = act + rep (total influence from seed).
    Signed score = act - rep (net activation direction).
    """

    seed_gene: str
    node_names: tuple[str, ...]
    act_scores: dict[str, float]       # Activation subgraph RWR
    rep_scores: dict[str, float]       # Repression subgraph RWR
    combined_scores: dict[str, float]  # act + rep (genome-wide ranking)
    signed_scores: dict[str, float]    # act - rep (net direction)
    n_act_edges: int
    n_rep_edges: int
    act_convergence: tuple[float, int]  # (delta, n_iter)
    rep_convergence: tuple[float, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_gene": self.seed_gene,
            "n_nodes": len(self.node_names),
            "n_act_edges": self.n_act_edges,
            "n_rep_edges": self.n_rep_edges,
            "act_convergence_delta": self.act_convergence[0],
            "act_n_iterations": self.act_convergence[1],
            "rep_convergence_delta": self.rep_convergence[0],
            "rep_n_iterations": self.rep_convergence[1],
        }


@dataclass(frozen=True)
class NetworkProximityReport:
    """Combined report for all three network proximity tests.

    Designed for integration as Phase 6 in validate_baselines.
    """

    proximity_decay: ProximityDecayResult
    reverse_causal: ReverseCausalResult
    rwr_correlation: RWRCorrelationResult
    bonferroni_alpha: float = 0.05 / 3  # Pre-specified: 3 tests

    @property
    def any_significant(self) -> bool:
        """At least one test passes Bonferroni-corrected threshold."""
        alpha = self.bonferroni_alpha
        return (
            self.proximity_decay.permutation_pvalue < alpha
            or self.reverse_causal.query_gene_pvalue < alpha
            or self.rwr_correlation.permutation_pvalue < alpha
        )

    @property
    def all_significant(self) -> bool:
        alpha = self.bonferroni_alpha
        return (
            self.proximity_decay.permutation_pvalue < alpha
            and self.reverse_causal.query_gene_pvalue < alpha
            and self.rwr_correlation.permutation_pvalue < alpha
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proximity_decay": self.proximity_decay.to_dict(),
            "reverse_causal": self.reverse_causal.to_dict(),
            "rwr_correlation": self.rwr_correlation.to_dict(),
            "bonferroni_alpha": float(self.bonferroni_alpha),
            "any_significant": self.any_significant,
            "all_significant": self.all_significant,
        }


# ---------------------------------------------------------------------------
# Test 1: Proximity decay
# ---------------------------------------------------------------------------


def _build_degree_bins(
    graph_degrees: dict[str, int],
    bin_size: int = 100,
) -> dict[int, list[str]]:
    """Bin genes by degree for degree-preserving permutation.

    Per Guney et al. 2016: genes are grouped into bins of >=100 nodes
    sorted by degree. Random selection within each bin preserves the
    degree distribution.
    """
    sorted_genes = sorted(graph_degrees.keys(), key=lambda g: graph_degrees[g])
    bins: dict[int, list[str]] = {}
    for i, gene in enumerate(sorted_genes):
        bin_id = i // bin_size
        bins.setdefault(bin_id, []).append(gene)
    return bins


def _degree_preserving_sample(
    gene: str,
    degree_bins: dict[int, list[str]],
    gene_to_bin: dict[str, int],
    rng: np.random.Generator,
) -> str:
    """Sample a random gene from the same degree bin."""
    bin_id = gene_to_bin[gene]
    candidates = degree_bins[bin_id]
    return candidates[rng.integers(len(candidates))]


def run_proximity_decay_test(
    distances: dict[str, int],
    abs_t_stats: dict[str, float],
    graph_degrees: dict[str, int],
    seed_gene: str,
    n_permutations: int = 1000,
    seed: int | None = None,
    verbose: bool = True,
) -> ProximityDecayResult:
    """Test whether |t-statistic| decays with graph distance from seed.

    Parameters
    ----------
    distances
        Gene symbol -> shortest path distance from seed. Only reachable
        genes are included.
    abs_t_stats
        Gene symbol -> |t-statistic| from differential expression.
    graph_degrees
        Gene symbol -> degree in INDRA graph (for permutation null).
    seed_gene
        Name of the seed gene (for reporting).
    n_permutations
        Number of degree-preserving permutations for the null.
    seed
        Random seed for reproducibility.

    Returns
    -------
    ProximityDecayResult
    """
    rng = np.random.default_rng(seed)

    # Intersect: genes with both distance and t-statistic
    common = sorted(set(distances) & set(abs_t_stats))
    if len(common) < 10:
        raise ValueError(
            f"Only {len(common)} genes have both distance and t-statistic. "
            "Need at least 10 for a meaningful test."
        )

    dist_arr = np.array([distances[g] for g in common], dtype=np.float64)
    t_arr = np.array([abs_t_stats[g] for g in common], dtype=np.float64)

    # Observed Spearman correlation (negative = decay with distance)
    observed_rho, observed_p = sp_stats.spearmanr(dist_arr, t_arr)

    # Bin genes by degree for degree-preserving permutation
    all_genes_with_degree = set(graph_degrees) & set(abs_t_stats)
    degree_bins = _build_degree_bins(
        {g: graph_degrees[g] for g in all_genes_with_degree}
    )
    gene_to_bin: dict[str, int] = {}
    for bin_id, genes in degree_bins.items():
        for g in genes:
            gene_to_bin[g] = bin_id

    # Permutation null: for each permutation, sample a degree-matched
    # random gene set and recompute the correlation
    null_rhos = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        # Permute the gene-to-distance mapping (keeping t-stats fixed)
        perm_distances = np.empty_like(dist_arr)
        for j, gene in enumerate(common):
            if gene in gene_to_bin:
                random_gene = _degree_preserving_sample(
                    gene, degree_bins, gene_to_bin, rng
                )
                perm_distances[j] = distances.get(
                    random_gene, np.max(dist_arr) + 1
                )
            else:
                perm_distances[j] = dist_arr[j]
        null_rhos[i] = sp_stats.spearmanr(perm_distances, t_arr).statistic

        if verbose and (i + 1) % 200 == 0:
            logger.info(
                "  Proximity decay permutation %d/%d", i + 1, n_permutations
            )

    # One-sided p-value: fraction of null rhos <= observed
    # (we expect negative rho: closer = stronger expression)
    perm_pvalue = (np.sum(null_rhos <= observed_rho) + 1) / (
        n_permutations + 1
    )

    # Distance bin statistics
    n_unreachable = len(set(abs_t_stats) - set(distances))
    unique_dists = sorted(set(int(d) for d in dist_arr))
    distance_bins: dict[int, dict[str, float]] = {}
    for d in unique_dists:
        mask = dist_arr == d
        t_bin = t_arr[mask]
        distance_bins[d] = {
            "n_genes": float(np.sum(mask)),
            "mean_abs_t": float(np.mean(t_bin)),
            "median_abs_t": float(np.median(t_bin)),
            "std_abs_t": float(np.std(t_bin, ddof=1)) if np.sum(mask) > 1 else 0.0,
        }

    return ProximityDecayResult(
        seed_gene=seed_gene,
        spearman_rho=float(observed_rho),
        spearman_pvalue=float(observed_p),
        permutation_pvalue=float(perm_pvalue),
        n_permutations=n_permutations,
        n_genes_reachable=len(common),
        n_genes_unreachable=n_unreachable,
        distance_bins=distance_bins,
    )


# ---------------------------------------------------------------------------
# Test 2: Reverse causal reasoning
# ---------------------------------------------------------------------------


def _compute_signed_enrichment_z(
    regulator_activated: set[str],
    regulator_repressed: set[str],
    up_genes: set[str],
    down_genes: set[str],
    background_size: int,
) -> float:
    """Signed enrichment z-score for a single regulator.

    Concordance: activated targets that are upregulated, or repressed
    targets that are downregulated. Discordance: the opposite.

    Returns z-score: positive means concordant regulation.
    """
    concordant = len(regulator_activated & up_genes) + len(
        regulator_repressed & down_genes
    )
    discordant = len(regulator_activated & down_genes) + len(
        regulator_repressed & up_genes
    )
    total = concordant + discordant

    if total == 0:
        return 0.0

    # Binomial z-score: deviation from 50% concordance
    p_hat = concordant / total
    se = np.sqrt(0.25 / total)  # SE under null p=0.5
    z = (p_hat - 0.5) / se
    return float(z)


def run_reverse_causal_reasoning(
    protein_results: pd.DataFrame,
    query_gene: str,
    feature_to_symbol: dict[str, str],
    env_file: Path | None = None,
    fdr_threshold: float = 0.05,
    min_evidence: int = 2,
    min_targets: int = 3,
    top_k: int = 20,
    verbose: bool = True,
) -> ReverseCausalResult:
    """Reverse causal reasoning: do DE genes point back to query_gene?

    Parameters
    ----------
    protein_results
        DataFrame from run_protein_differential with columns:
        feature_id, t_statistic, log2fc, fdr, is_target.
    query_gene
        Gene symbol to look for in regulator results (e.g. "C9orf72").
    feature_to_symbol
        Mapping from feature_id (UniProt) to gene symbol.
    env_file
        Path to INDRA .env file.
    fdr_threshold
        FDR cutoff for defining DE genes.
    min_evidence
        Minimum INDRA evidence for regulator edges.
    min_targets
        Minimum number of DE targets for a regulator to be tested.
    top_k
        Number of top regulators to report.

    Returns
    -------
    ReverseCausalResult
    """
    from cliquefinder.knowledge.cogex import (
        ALL_REGULATORY_TYPES,
        ACTIVATION_TYPES,
        REPRESSION_TYPES,
        INDRAModuleExtractor,
    )

    # 1. Split DE genes into up and down
    # Compute FDR if not already present (CSV restore may lack it)
    if "fdr" not in protein_results.columns:
        from statsmodels.stats.multitest import multipletests
        pvals = protein_results["p_value"].fillna(1.0).values
        _, fdr_vals, _, _ = multipletests(pvals, method="fdr_bh")
        protein_results = protein_results.copy()
        protein_results["fdr"] = fdr_vals
    sig = protein_results[protein_results["fdr"] < fdr_threshold].copy()
    sig_symbols = set()
    for _, row in sig.iterrows():
        sym = feature_to_symbol.get(row["feature_id"])
        if sym:
            sig_symbols.add(sym)

    up_genes: set[str] = set()
    down_genes: set[str] = set()
    for _, row in sig.iterrows():
        sym = feature_to_symbol.get(row["feature_id"])
        if sym:
            if row["log2fc"] > 0:
                up_genes.add(sym)
            else:
                down_genes.add(sym)

    if verbose:
        logger.info(
            "Reverse causal reasoning: %d up, %d down (FDR < %.3f)",
            len(up_genes),
            len(down_genes),
            fdr_threshold,
        )

    # 2. Query INDRA for all regulators of the DE gene universe
    from cliquefinder.knowledge.cogex import CoGExClient
    cogex_client = CoGExClient(env_file=env_file)
    extractor = INDRAModuleExtractor(client=cogex_client)
    try:
        all_symbols = list(sig_symbols)
        modules = extractor.discover_modules(
            gene_universe=all_symbols,
            min_evidence=min_evidence,
            min_targets=min_targets,
            stmt_types=list(ALL_REGULATORY_TYPES),
        )
    finally:
        cogex_client.close()

    if verbose:
        logger.info("  Found %d regulators with >= %d DE targets", len(modules), min_targets)

    # 3. Score each regulator by signed concordance
    regulator_scores: dict[str, float] = {}
    for module in modules:
        reg_name = module.regulator_name
        activated = {e.target_name for e in module.targets if e.regulation_type == "activation"}
        repressed = {e.target_name for e in module.targets if e.regulation_type == "repression"}

        z = _compute_signed_enrichment_z(
            activated, repressed, up_genes, down_genes, len(sig_symbols)
        )
        regulator_scores[reg_name] = z

    # 4. Rank regulators
    sorted_regs = sorted(
        regulator_scores.items(), key=lambda x: abs(x[1]), reverse=True
    )

    # Find query gene rank
    query_gene_upper = query_gene.upper()
    query_rank = -1
    query_z = 0.0
    for i, (reg, z) in enumerate(sorted_regs):
        if reg.upper() == query_gene_upper:
            query_rank = i + 1
            query_z = z
            break

    # Compute p-value: fraction of regulators with |z| >= |query_z|
    all_z = np.array([abs(z) for _, z in sorted_regs])
    if query_rank > 0 and len(all_z) > 0:
        query_pvalue = float(np.mean(all_z >= abs(query_z)))
    else:
        query_pvalue = 1.0

    # Top K regulators
    top_regs = [
        {"regulator": reg, "zscore": float(z), "rank": i + 1}
        for i, (reg, z) in enumerate(sorted_regs[:top_k])
    ]

    return ReverseCausalResult(
        query_gene=query_gene,
        query_gene_rank=query_rank if query_rank > 0 else len(sorted_regs) + 1,
        query_gene_zscore=float(query_z),
        query_gene_pvalue=query_pvalue,
        n_regulators_tested=len(regulator_scores),
        n_up_submitted=len(up_genes),
        n_down_submitted=len(down_genes),
        top_regulators=top_regs,
        regulator_scores=regulator_scores,
    )


# ---------------------------------------------------------------------------
# Test 3: RWR correlation
# ---------------------------------------------------------------------------


def compute_rwr_scores(
    adjacency: sp.csr_matrix,
    seed_index: int,
    restart_prob: float = 0.15,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> tuple[NDArray[np.float64], float, int]:
    """Random Walk with Restart via power iteration.

    Parameters
    ----------
    adjacency
        Sparse adjacency matrix (n_nodes x n_nodes). Does not need to be
        symmetric — directed edges are respected.
    seed_index
        Index of the seed node in the adjacency matrix.
    restart_prob
        Probability of returning to seed at each step. Default 0.15
        (equivalent to PageRank damping factor 0.85).
    tol
        Convergence tolerance on L1 norm of score change.
    max_iter
        Maximum number of iterations.

    Returns
    -------
    scores : ndarray of shape (n_nodes,)
        Steady-state RWR probability for each node.
    delta : float
        Final convergence delta.
    n_iter : int
        Number of iterations performed.
    """
    n = adjacency.shape[0]

    # Row-normalize adjacency → transition matrix T, then transpose.
    # adjacency[i,j] means edge from i to j. Row sums = out-degree.
    # T[i,j] = A[i,j] / out_degree(i) gives transition probabilities.
    # RWR operates on T^T: p = (1-alpha) * T^T * p + alpha * e_seed.
    row_sums = np.array(adjacency.sum(axis=1)).ravel()
    dangling = row_sums == 0  # Sink nodes (no outgoing edges)
    row_sums[dangling] = 1.0  # Avoid division by zero
    T = adjacency.multiply(1.0 / row_sums[:, np.newaxis])
    W = T.T.tocsr()  # Transpose for left-multiplication

    # Seed vector
    e_seed = np.zeros(n, dtype=np.float64)
    e_seed[seed_index] = 1.0

    # Power iteration: p(t+1) = (1-alpha) * (W * p + dangling_mass/n) + alpha * e_seed
    # Dangling nodes absorb probability; redistribute uniformly (standard PageRank).
    p = e_seed.copy()
    alpha = restart_prob
    has_dangling = dangling.any()
    for iteration in range(max_iter):
        Wp = W.dot(p)
        if has_dangling:
            # Redistribute probability mass from dangling nodes uniformly
            dangling_mass = p[dangling].sum()
            Wp += dangling_mass / n
        p_new = (1 - alpha) * Wp + alpha * e_seed
        delta = np.sum(np.abs(p_new - p))
        p = p_new
        if delta < tol:
            return p, float(delta), iteration + 1

    logger.warning(
        "RWR did not converge after %d iterations (delta=%.2e)", max_iter, delta
    )
    return p, float(delta), max_iter


def run_rwr_correlation_test(
    rwr_scores: dict[str, float],
    abs_t_stats: dict[str, float],
    seed_gene: str,
    n_graph_nodes: int,
    n_graph_edges: int,
    restart_probability: float,
    convergence_delta: float,
    n_rwr_iterations: int,
    n_permutations: int = 1000,
    seed: int | None = None,
    verbose: bool = True,
) -> RWRCorrelationResult:
    """Test whether RWR proximity correlates with |t-statistic|.

    Parameters
    ----------
    rwr_scores
        Gene symbol -> RWR score from seed gene.
    abs_t_stats
        Gene symbol -> |t-statistic| from differential expression.
    seed_gene
        Name of the seed gene.
    n_permutations
        Number of gene-label permutations for the null.
    seed
        Random seed for reproducibility.

    Returns
    -------
    RWRCorrelationResult
    """
    rng = np.random.default_rng(seed)

    # Intersect: genes with both RWR score and t-statistic
    common = sorted(set(rwr_scores) & set(abs_t_stats))
    if len(common) < 10:
        raise ValueError(
            f"Only {len(common)} genes have both RWR score and t-statistic."
        )

    rwr_arr = np.array([rwr_scores[g] for g in common], dtype=np.float64)
    t_arr = np.array([abs_t_stats[g] for g in common], dtype=np.float64)

    # Observed Spearman correlation (positive = closer in graph → higher |t|)
    observed_rho, observed_p = sp_stats.spearmanr(rwr_arr, t_arr)

    # Permutation null: shuffle gene labels on t-statistics
    null_rhos = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm_t = rng.permutation(t_arr)
        null_rhos[i] = sp_stats.spearmanr(rwr_arr, perm_t).statistic

        if verbose and (i + 1) % 200 == 0:
            logger.info(
                "  RWR correlation permutation %d/%d", i + 1, n_permutations
            )

    # One-sided p-value: fraction of null rhos >= observed
    # (positive rho means closer = more differentially expressed)
    perm_pvalue = (np.sum(null_rhos >= observed_rho) + 1) / (
        n_permutations + 1
    )

    return RWRCorrelationResult(
        seed_gene=seed_gene,
        spearman_rho=float(observed_rho),
        spearman_pvalue=float(observed_p),
        permutation_pvalue=float(perm_pvalue),
        n_permutations=n_permutations,
        n_genes=len(common),
        restart_probability=restart_probability,
        n_graph_nodes=n_graph_nodes,
        n_graph_edges=n_graph_edges,
        convergence_delta=convergence_delta,
        n_iterations=n_rwr_iterations,
    )


# ---------------------------------------------------------------------------
# Signed RWR — separate propagation on activation/repression subgraphs
# ---------------------------------------------------------------------------


# Edge type sets imported lazily to avoid circular imports
_ACTIVATION_TYPES: set[str] | None = None
_REPRESSION_TYPES: set[str] | None = None


def _load_edge_types() -> tuple[set[str], set[str]]:
    global _ACTIVATION_TYPES, _REPRESSION_TYPES
    if _ACTIVATION_TYPES is None:
        from cliquefinder.knowledge.cogex import ACTIVATION_TYPES, REPRESSION_TYPES
        _ACTIVATION_TYPES = ACTIVATION_TYPES
        _REPRESSION_TYPES = REPRESSION_TYPES
    return _ACTIVATION_TYPES, _REPRESSION_TYPES


def partition_signed_edges(
    edges: list[tuple[str, str, dict[str, Any]]],
) -> tuple[list[tuple[str, str, dict]], list[tuple[str, str, dict]]]:
    """Split edges by regulatory sign using INDRA statement types.

    Parameters
    ----------
    edges
        List of (source, target, attrs) where attrs contains 'stmt_type'.

    Returns
    -------
    (activation_edges, repression_edges)
        Two edge lists. Edges with types not in either set are excluded.
    """
    act_types, rep_types = _load_edge_types()
    act_edges = []
    rep_edges = []
    for src, tgt, attrs in edges:
        st = attrs.get("stmt_type", "")
        if st in act_types:
            act_edges.append((src, tgt, attrs))
        elif st in rep_types:
            rep_edges.append((src, tgt, attrs))
    return act_edges, rep_edges


def compute_signed_rwr_scores(
    edges: list[tuple[str, str, dict[str, Any]]],
    seed_gene: str,
    restart_prob: float = 0.15,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> SignedRWRResult:
    """Signed RWR via separate subgraph propagation.

    Splits INDRA edges into activation and repression subgraphs, runs
    ``compute_rwr_scores()`` on each independently, and returns combined
    and signed proximity scores.

    Parameters
    ----------
    edges
        List of (source, target, attrs) with 'stmt_type' and 'evidence_count'.
    seed_gene
        Gene symbol of the seed node.
    restart_prob
        Restart probability (0.15 = damping factor 0.85).

    Returns
    -------
    SignedRWRResult
        Contains per-gene activation, repression, combined, and signed scores.
    """
    act_edges, rep_edges = partition_signed_edges(edges)

    # Build graphs and collect the union of all nodes
    all_nodes: set[str] = {seed_gene}
    for src, tgt, _ in edges:
        all_nodes.add(src)
        all_nodes.add(tgt)
    node_list = sorted(all_nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    if seed_gene not in node_to_idx:
        raise ValueError(f"Seed gene '{seed_gene}' not found in edges.")
    seed_idx = node_to_idx[seed_gene]

    def _build_adjacency(edge_list: list) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        for src, tgt, attrs in edge_list:
            if src in node_to_idx and tgt in node_to_idx:
                rows.append(node_to_idx[src])
                cols.append(node_to_idx[tgt])
                data.append(float(attrs.get("evidence_count", 1)))
        if not rows:
            return sp.csr_matrix((n, n), dtype=np.float64)
        return sp.csr_matrix(
            (np.array(data, dtype=np.float64), (np.array(rows), np.array(cols))),
            shape=(n, n),
        )

    act_adj = _build_adjacency(act_edges)
    rep_adj = _build_adjacency(rep_edges)

    # Run RWR on each subgraph
    act_scores_arr, act_delta, act_iter = compute_rwr_scores(
        act_adj, seed_idx, restart_prob, tol, max_iter,
    )
    rep_scores_arr, rep_delta, rep_iter = compute_rwr_scores(
        rep_adj, seed_idx, restart_prob, tol, max_iter,
    )

    # Build per-gene score dicts (raw probabilities)
    act_scores = {node_list[i]: float(act_scores_arr[i]) for i in range(n)}
    rep_scores = {node_list[i]: float(rep_scores_arr[i]) for i in range(n)}
    combined = {g: act_scores[g] + rep_scores[g] for g in node_list}

    # Signed score: z-normalize within each subgraph before subtracting.
    # Raw act - rep is invalid because the two subgraphs have different
    # densities, so raw probabilities are not on a comparable scale.
    def _zscore(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        std = arr.std()
        if std < 1e-12:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / std

    act_z = _zscore(act_scores_arr)
    rep_z = _zscore(rep_scores_arr)
    signed = {node_list[i]: float(act_z[i] - rep_z[i]) for i in range(n)}

    return SignedRWRResult(
        seed_gene=seed_gene,
        node_names=tuple(node_list),
        act_scores=act_scores,
        rep_scores=rep_scores,
        combined_scores=combined,
        signed_scores=signed,
        n_act_edges=len(act_edges),
        n_rep_edges=len(rep_edges),
        act_convergence=(act_delta, act_iter),
        rep_convergence=(rep_delta, rep_iter),
    )


# ---------------------------------------------------------------------------
# Graph extraction (from CoGExClient or MCP)
# ---------------------------------------------------------------------------


def extract_indra_subgraph_edges(
    cogex_client: Any,
    seed_gene_id: str,
    max_hops: int = 4,
    min_evidence: int = 1,
    stmt_types: list[str] | None = None,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Extract INDRA indra_rel edges within max_hops of seed gene.

    Uses BFS expansion from the seed node. Each edge is returned as
    (source_name, target_name, {evidence_count, stmt_type}).

    Parameters
    ----------
    cogex_client
        An active CoGExClient instance. Uses ``_execute_query()`` for
        retry/reconnect handling.
    seed_gene_id
        CURIE identifier for the seed gene (e.g. "hgnc:1667").
    max_hops
        Maximum path length from seed to include in subgraph.
    min_evidence
        Minimum evidence count for edges to include.
    stmt_types
        If provided, filter to these statement types.

    Returns
    -------
    List of (source_name, target_name, edge_attrs) tuples.
    """
    stmt_filter = ""
    if stmt_types:
        stmt_filter = "AND r.stmt_type IN $stmt_types"

    # Two-phase extraction: first find reachable nodes, then collect edges.
    # This avoids the combinatorial explosion of variable-length path +
    # edge matching in a single query on a 92M-node graph.
    query = f"""
    MATCH (seed:BioEntity {{id: $seed_id}})
    CALL apoc.path.subgraphNodes(seed, {{
        relationshipFilter: 'indra_rel',
        maxLevel: {max_hops}
    }}) YIELD node
    WITH collect(node) AS nodes
    UNWIND nodes AS a
    MATCH (a)-[r:indra_rel]->(b)
    WHERE b IN nodes
      AND r.evidence_count >= $min_evidence
      {stmt_filter}
    RETURN DISTINCT
      a.name AS source_name,
      a.id AS source_id,
      b.name AS target_name,
      b.id AS target_id,
      r.evidence_count AS evidence_count,
      r.stmt_type AS stmt_type
    """

    params: dict[str, Any] = {
        "seed_id": seed_gene_id,
        "min_evidence": min_evidence,
    }
    if stmt_types:
        params["stmt_types"] = stmt_types

    rows = cogex_client._execute_query(query, **params)

    edges = []
    for row in rows:
        # query_tx returns List[List[Any]] — positional access only
        # Column order: source_name(0), source_id(1), target_name(2),
        #               target_id(3), evidence_count(4), stmt_type(5)
        edges.append((
            row[0],  # source_name
            row[2],  # target_name
            {
                "evidence_count": row[4],
                "stmt_type": row[5],
                "source_id": row[1],
                "target_id": row[3],
            },
        ))

    return edges


def query_shortest_paths_batched(
    cogex_client: Any,
    seed_gene_name: str,
    target_gene_names: list[str],
    max_hops: int = 8,
    batch_size: int = 500,
    verbose: bool = True,
) -> dict[str, int]:
    """Query Neo4j for shortest path distances from seed to target genes.

    Uses server-side BFS via ``shortestPath`` — orders of magnitude faster
    than extracting the full subgraph and computing BFS locally.

    Parameters
    ----------
    cogex_client
        An active CoGExClient instance.
    seed_gene_name
        Gene symbol of the seed (e.g. "C9orf72").
    target_gene_names
        List of gene symbols to find distances for.
    max_hops
        Maximum path length.
    batch_size
        Number of target genes per query batch.

    Returns
    -------
    Dict of {gene_name: min_distance}. Only reachable genes are included.
    """
    distances: dict[str, int] = {}
    n_batches = (len(target_gene_names) + batch_size - 1) // batch_size

    for i in range(0, len(target_gene_names), batch_size):
        batch = target_gene_names[i:i + batch_size]
        query = f"""
        MATCH (seed:BioEntity {{name: $seed_name}})
        MATCH (target:BioEntity)
        WHERE target.name IN $gene_list
        MATCH path = shortestPath((seed)-[:indra_rel*..{max_hops}]-(target))
        WITH target.name AS gene, length(path) AS distance
        RETURN gene, min(distance) AS min_distance
        """
        rows = cogex_client._execute_query(
            query, seed_name=seed_gene_name, gene_list=batch
        )
        for row in rows:
            gene, dist = row[0], int(row[1])
            if gene not in distances or dist < distances[gene]:
                distances[gene] = dist

        if verbose:
            batch_num = i // batch_size + 1
            logger.info(
                "  Shortest paths batch %d/%d: %d genes found",
                batch_num, n_batches, len(distances),
            )

    return distances


def query_gene_degrees_batched(
    cogex_client: Any,
    gene_names: list[str],
    batch_size: int = 500,
) -> dict[str, int]:
    """Query Neo4j for indra_rel degree of each gene (undirected).

    Parameters
    ----------
    cogex_client
        An active CoGExClient instance.
    gene_names
        Gene symbols to query.
    batch_size
        Genes per batch.

    Returns
    -------
    Dict of {gene_name: degree}.
    """
    degrees: dict[str, int] = {}

    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i:i + batch_size]
        query = """
        MATCH (g:BioEntity)
        WHERE g.name IN $gene_list
        OPTIONAL MATCH (g)-[r:indra_rel]-()
        WITH g.name AS gene, count(r) AS degree
        RETURN gene, degree
        """
        rows = cogex_client._execute_query(query, gene_list=batch)
        for row in rows:
            gene, deg = row[0], int(row[1])
            degrees[gene] = max(degrees.get(gene, 0), deg)

    return degrees


def extract_local_subgraph_edges(
    cogex_client: Any,
    seed_gene_name: str,
    max_hops: int = 2,
    min_evidence: int = 1,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Extract a small local subgraph around seed gene for RWR.

    Unlike ``extract_indra_subgraph_edges`` (which uses APOC and returns
    millions of edges at 4-hop), this uses iterative 1-hop expansion and
    returns a much smaller subgraph suitable for in-memory RWR.

    Parameters
    ----------
    cogex_client
        An active CoGExClient instance.
    seed_gene_name
        Gene symbol (name, not CURIE).
    max_hops
        Maximum hops from seed.
    min_evidence
        Minimum evidence count for edges.

    Returns
    -------
    List of (source_name, target_name, edge_attrs) tuples.
    """
    query = f"""
    MATCH (seed:BioEntity {{name: $seed_name}})
    CALL apoc.path.subgraphNodes(seed, {{
        relationshipFilter: 'indra_rel',
        maxLevel: {max_hops}
    }}) YIELD node
    WITH collect(node) AS nodes
    UNWIND nodes AS a
    MATCH (a)-[r:indra_rel]->(b)
    WHERE b IN nodes
      AND r.evidence_count >= $min_evidence
    RETURN DISTINCT
      a.name AS source_name,
      b.name AS target_name,
      r.evidence_count AS evidence_count,
      r.stmt_type AS stmt_type
    """
    rows = cogex_client._execute_query(
        query, seed_name=seed_gene_name, min_evidence=min_evidence
    )

    edges = []
    for row in rows:
        edges.append((
            row[0],  # source_name
            row[1],  # target_name
            {
                "evidence_count": row[2],
                "stmt_type": row[3],
            },
        ))
    return edges


def build_networkx_graph(
    edges: list[tuple[str, str, dict[str, Any]]],
    weight_key: str = "evidence_count",
) -> tuple[Any, dict[str, int]]:
    """Build a NetworkX DiGraph from extracted edges.

    Parameters
    ----------
    edges
        List of (source, target, attrs) tuples.
    weight_key
        Edge attribute to use as weight. Inverted (1/w) for shortest
        path computation (higher evidence = shorter distance).

    Returns
    -------
    G : nx.DiGraph
    node_to_idx : dict mapping node name -> integer index
    """
    import networkx as nx

    G = nx.DiGraph()
    for src, tgt, attrs in edges:
        w = attrs.get(weight_key, 1)
        # Invert: higher evidence → shorter weighted distance
        inv_w = 1.0 / max(w, 1)
        G.add_edge(src, tgt, weight=inv_w, **attrs)

    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    return G, node_to_idx


def compute_shortest_paths(
    G: Any,
    seed_gene: str,
) -> dict[str, int]:
    """Compute unweighted shortest path from seed to all reachable nodes.

    Returns dict of {gene_name: distance}. Unreachable genes are omitted.
    """
    import networkx as nx

    if seed_gene not in G:
        raise ValueError(f"Seed gene '{seed_gene}' not found in graph.")

    # Undirected shortest path (regulatory links can be traversed either way
    # for proximity; direction matters for RWR but not for distance)
    G_undirected = G.to_undirected()
    lengths = nx.single_source_shortest_path_length(G_undirected, seed_gene)
    # Remove self-distance
    lengths.pop(seed_gene, None)
    return dict(lengths)


def graph_to_sparse_adjacency(
    G: Any,
    nodes: list[str],
) -> sp.csr_matrix:
    """Convert NetworkX DiGraph to sparse adjacency matrix.

    Entries are edge weights (evidence_count).
    """
    import networkx as nx

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rows, cols, data = [], [], []

    for src, tgt, attrs in G.edges(data=True):
        if src in node_to_idx and tgt in node_to_idx:
            i = node_to_idx[src]
            j = node_to_idx[tgt]
            rows.append(i)
            cols.append(j)
            data.append(attrs.get("evidence_count", 1))

    return sp.csr_matrix(
        (np.array(data, dtype=np.float64), (np.array(rows), np.array(cols))),
        shape=(n, n),
    )


def compute_graph_degrees(
    G: Any,
) -> dict[str, int]:
    """Compute undirected degree for each node."""
    G_undirected = G.to_undirected()
    return dict(G_undirected.degree())
