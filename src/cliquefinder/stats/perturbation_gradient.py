"""Perturbation gradient discovery -- measures how differential expression
decays with knowledge graph distance from a seed gene.

Replaces the binary per-arm gate approach (which had 56.5% FPR with ROAST)
with a continuous, aggregate metric.  Instead of testing each intermediary
gene set individually, measures the decay shape across ALL genes at each
hop distance.

The gradient operates on the AGGREGATE of all genes at a hop shell, not
individual arms.  This is fundamentally different from per-arm testing.

Null model: degree-preserving gene-label permutation (Guney et al. 2016).
|t| values are shuffled within degree bins so that hub genes only swap
with other hubs.  This controls for the confound that high-degree genes
are both closer to any seed AND better-studied / more detectable.

Key outputs:
- Hop shell statistics: mean|t|, median|t|, n_genes per distance
- Gradient slope: WLS fit of mean|t| ~ distance with permutation p-value
- Spearman rho: rank correlation between distance and |t| (secondary)
- Active horizon: furthest hop where shell mean|t| exceeds background
  (descriptive, not inferential)
- Edge-quality stratification: gradient broken out by edge quality tier
  (exploratory, Bonferroni-corrected)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HopShellStats:
    """Statistics for all measurable genes at a given hop distance."""

    hop: int
    n_genes: int
    mean_abs_t: float
    median_abs_t: float
    std_abs_t: float
    genes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop": self.hop,
            "n_genes": self.n_genes,
            "mean_abs_t": float(self.mean_abs_t),
            "median_abs_t": float(self.median_abs_t),
            "std_abs_t": float(self.std_abs_t),
        }


@dataclass(frozen=True)
class GradientResult:
    """Result of perturbation gradient analysis.

    Tests whether differential expression decays with graph distance
    from the seed gene, using degree-preserving gene-label permutation.
    """

    seed_gene: str
    shells: tuple[HopShellStats, ...]

    # WLS slope of mean|t| ~ hop distance (negative = decay)
    slope: float
    slope_pvalue: float

    # Spearman rank correlation between hop distance and |t| (secondary).
    # spearman_pvalue is the PERMUTATION p-value (computed from the
    # degree-preserving null), not the asymptotic Spearman p-value.
    spearman_rho: float
    spearman_pvalue: float

    # Furthest hop where shell mean|t| > non-shell background (descriptive)
    active_horizon: int
    background_mean_abs_t: float

    n_permutations: int
    n_genes_total: int

    # Exploratory, Bonferroni-corrected across tiers
    stratified: dict[str, "GradientResult"] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "seed_gene": self.seed_gene,
            "shells": [s.to_dict() for s in self.shells],
            "slope": float(self.slope),
            "slope_pvalue": float(self.slope_pvalue),
            "spearman_rho": float(self.spearman_rho),
            "spearman_pvalue": float(self.spearman_pvalue),
            "active_horizon": self.active_horizon,
            "background_mean_abs_t": float(self.background_mean_abs_t),
            "n_permutations": self.n_permutations,
            "n_genes_total": self.n_genes_total,
        }
        if self.stratified:
            d["stratified"] = {k: v.to_dict() for k, v in self.stratified.items()}
        return d


# ---------------------------------------------------------------------------
# BFS hop shell computation
# ---------------------------------------------------------------------------


def compute_hop_shells(
    adjacency: dict[str, list[str]],
    seed: str,
    max_hops: int = 6,
) -> dict[int, set[str]]:
    """BFS from seed, returning genes grouped by shortest-path distance.

    Parameters
    ----------
    adjacency
        Gene symbol -> list of neighbor gene symbols.
    seed
        Seed gene symbol.
    max_hops
        Maximum BFS depth.

    Returns
    -------
    ``{hop_distance: set_of_genes}`` -- hop 1 = direct neighbors, etc.
    """
    visited = {seed}
    frontier = {seed}
    shells: dict[int, set[str]] = {}

    for hop in range(1, max_hops + 1):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)
        if not next_frontier:
            break
        shells[hop] = next_frontier
        frontier = next_frontier

    return shells


# ---------------------------------------------------------------------------
# Shell statistics and gradient metrics
# ---------------------------------------------------------------------------


def _compute_shell_stats(
    genes: set[str],
    abs_t_stats: dict[str, float],
    hop: int,
) -> HopShellStats:
    """Aggregate statistics for a single hop shell."""
    measurable = sorted(g for g in genes if g in abs_t_stats)
    if not measurable:
        return HopShellStats(
            hop=hop, n_genes=0,
            mean_abs_t=float("nan"), median_abs_t=float("nan"),
            std_abs_t=float("nan"), genes=(),
        )

    t_vals = np.array([abs_t_stats[g] for g in measurable])
    return HopShellStats(
        hop=hop,
        n_genes=len(measurable),
        mean_abs_t=float(np.mean(t_vals)),
        median_abs_t=float(np.median(t_vals)),
        std_abs_t=float(np.std(t_vals, ddof=1)) if len(t_vals) > 1 else 0.0,
        genes=tuple(measurable),
    )


def _gradient_slope(shells: list[HopShellStats]) -> float:
    """WLS slope of mean|t| ~ hop distance, weighted by n_genes.

    Under CLT with approximately independent genes, var(shell mean)
    scales as 1/n, so inverse-variance weights are proportional to n.
    The permutation p-value is valid regardless of weight choice (same
    statistic applied to observed and null), so this affects sensitivity
    but not validity.

    Returns slope (negative = perturbation decays with distance).
    """
    valid = [(s.hop, s.mean_abs_t, s.n_genes) for s in shells if s.n_genes > 0]
    if len(valid) < 2:
        return 0.0

    x = np.array([v[0] for v in valid], dtype=np.float64)
    y = np.array([v[1] for v in valid], dtype=np.float64)
    w = np.array([v[2] for v in valid], dtype=np.float64)

    xw = x * w
    yw = y * w
    sum_w = np.sum(w)
    sum_xw = np.sum(xw)
    sum_yw = np.sum(yw)
    sum_xyw = np.sum(x * yw)
    sum_x2w = np.sum(x * xw)

    denom = sum_w * sum_x2w - sum_xw**2
    if abs(denom) < 1e-15:
        return 0.0

    return float((sum_w * sum_xyw - sum_xw * sum_yw) / denom)


def _active_horizon(
    shells: list[HopShellStats],
    background_mean: float,
) -> int:
    """Furthest hop where shell mean|t| exceeds non-shell background.

    This is a descriptive metric, not an inferential criterion.
    """
    horizon = 0
    for s in shells:
        if s.n_genes > 0 and s.mean_abs_t > background_mean:
            horizon = s.hop
    return horizon


# ---------------------------------------------------------------------------
# Degree-preserving permutation
# ---------------------------------------------------------------------------


def _compute_graph_degrees(
    adjacency: dict[str, list[str]],
    all_genes: list[str],
) -> dict[str, int]:
    """Compute node degree from adjacency for all genes.

    Genes not in the adjacency get degree 0 (isolated nodes).
    """
    degrees: dict[str, int] = {}
    for g in all_genes:
        degrees[g] = len(adjacency.get(g, []))
    return degrees


def _build_degree_bins(
    graph_degrees: dict[str, int],
    bin_size: int = 100,
) -> tuple[dict[int, list[str]], dict[str, int]]:
    """Bin genes by degree for degree-preserving permutation.

    Per Guney et al. 2016: genes are grouped into bins of >= bin_size
    nodes sorted by degree.

    Returns
    -------
    degree_bins
        ``{bin_id: [gene, ...]}``
    gene_to_bin
        ``{gene: bin_id}``
    """
    sorted_genes = sorted(graph_degrees.keys(), key=lambda g: graph_degrees[g])
    bins: dict[int, list[str]] = {}
    gene_to_bin: dict[str, int] = {}
    for i, gene in enumerate(sorted_genes):
        bin_id = i // bin_size
        bins.setdefault(bin_id, []).append(gene)
        gene_to_bin[gene] = bin_id
    return bins, gene_to_bin


def _degree_preserving_permute(
    all_genes: list[str],
    all_t_values: NDArray[np.float64],
    degree_bins: dict[int, list[str]],
    gene_to_bin: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Permute |t| values within degree bins.

    Each gene's |t| is swapped only with genes of similar degree,
    controlling for the confound between hub status and |t| magnitude.
    """
    gene_to_t = dict(zip(all_genes, all_t_values))
    perm_t: dict[str, float] = {}

    for bin_id, bin_genes in degree_bins.items():
        bin_t = np.array([gene_to_t.get(g, 0.0) for g in bin_genes])
        shuffled = rng.permutation(bin_t)
        for g, t in zip(bin_genes, shuffled):
            perm_t[g] = float(t)

    return perm_t


# ---------------------------------------------------------------------------
# Main gradient test
# ---------------------------------------------------------------------------


def run_gradient_test(
    adjacency: dict[str, list[str]],
    abs_t_stats: dict[str, float],
    seed: str,
    max_hops: int = 6,
    n_permutations: int = 1000,
    rng_seed: int | None = None,
    edge_quality: dict[str, str] | None = None,
    verbose: bool = True,
    precomputed_shells: dict[int, set[str]] | None = None,
    graph_degrees: dict[str, int] | None = None,
) -> GradientResult:
    """Test whether perturbation decays with graph distance from seed.

    Degree-preserving gene-label permutation null: |t| values are
    shuffled within degree bins (Guney et al. 2016) so that hub genes
    only swap with other hubs.  This controls for the confound that
    high-degree genes are systematically closer to any seed.

    Parameters
    ----------
    adjacency
        Gene -> list of neighbors.  BFS traverses outward from seed.
        Used for shell construction (when ``precomputed_shells`` is None)
        and degree computation (when ``graph_degrees`` is None).
    abs_t_stats
        Gene symbol -> |t-statistic| from differential expression.
    seed
        Seed gene symbol.
    max_hops
        Maximum BFS depth.
    n_permutations
        Number of degree-preserving permutations for null distribution.
    rng_seed
        Random seed for reproducibility.
    edge_quality
        Gene -> quality tier for stratification (e.g. ``"multi_source"``,
        ``"single_curated"``, ``"single_text_mined"``).
        Results are Bonferroni-corrected across tiers (exploratory).
    verbose
        Log progress.
    precomputed_shells
        Optional ``{hop: set_of_genes}`` to bypass BFS.  Use when shells
        are computed externally (e.g. via Neo4j ``shortestPath`` over the
        full INDRA graph, allowing unmeasured intermediaries).  When
        provided, ``adjacency`` may be empty.
    graph_degrees
        Optional ``{gene: degree}`` to bypass local-adjacency degree
        counting.  Use when degrees come from the full graph (not the
        measured-only induced subgraph).  When provided, the
        degree-preserving null permutes within bins of these degrees
        instead of locally-derived ones.

    Returns
    -------
    GradientResult
    """
    rng = np.random.default_rng(rng_seed)

    # 1. Hop shells: prefer precomputed; otherwise BFS adjacency.
    if precomputed_shells is not None:
        shells_sets = {
            h: gs for h, gs in precomputed_shells.items() if 1 <= h <= max_hops
        }
    else:
        shells_sets = compute_hop_shells(adjacency, seed, max_hops)
    if not shells_sets:
        raise ValueError(f"No neighbors found for seed '{seed}' in adjacency.")

    # 2. Observed shell statistics
    observed_shells: list[HopShellStats] = []
    all_shell_genes: set[str] = set()
    for hop in sorted(shells_sets):
        stats = _compute_shell_stats(shells_sets[hop], abs_t_stats, hop)
        observed_shells.append(stats)
        all_shell_genes.update(stats.genes)

    n_total = len(all_shell_genes)
    if n_total < 10:
        raise ValueError(
            f"Only {n_total} measurable genes in graph neighborhood. "
            "Need at least 10."
        )

    # 3. Observed gradient metrics
    observed_slope = _gradient_slope(observed_shells)

    gene_distances: dict[str, int] = {}
    for hop, genes in shells_sets.items():
        for g in genes:
            if g in abs_t_stats:
                gene_distances[g] = hop
    common = sorted(gene_distances)
    dist_arr = np.array([gene_distances[g] for g in common], dtype=np.float64)
    t_arr = np.array([abs_t_stats[g] for g in common], dtype=np.float64)
    observed_rho = float(sp_stats.spearmanr(dist_arr, t_arr).statistic)

    # Background: mean|t| of genes NOT in any shell (avoids circularity)
    non_shell_t = [v for g, v in abs_t_stats.items() if g not in all_shell_genes]
    if non_shell_t:
        bg_mean = float(np.mean(non_shell_t))
    else:
        bg_mean = float(np.mean(list(abs_t_stats.values())))
    observed_horizon = _active_horizon(observed_shells, bg_mean)

    # 4. Build degree bins for degree-preserving permutation
    all_genes = sorted(abs_t_stats.keys())
    all_t_values = np.array([abs_t_stats[g] for g in all_genes])
    if graph_degrees is None:
        graph_degrees = _compute_graph_degrees(adjacency, all_genes)
    else:
        # Fill missing degrees with 0 so every measured gene gets binned
        graph_degrees = {g: graph_degrees.get(g, 0) for g in all_genes}
    degree_bins, gene_to_bin = _build_degree_bins(graph_degrees)

    # 5. Permutation null: permute |t| within degree bins
    null_slopes = np.empty(n_permutations, dtype=np.float64)
    null_rhos = np.empty(n_permutations, dtype=np.float64)

    for perm_i in range(n_permutations):
        perm_t_map = _degree_preserving_permute(
            all_genes, all_t_values, degree_bins, gene_to_bin, rng,
        )

        perm_shells = [
            _compute_shell_stats(shells_sets[hop], perm_t_map, hop)
            for hop in sorted(shells_sets)
        ]
        null_slopes[perm_i] = _gradient_slope(perm_shells)

        perm_t_arr = np.array([perm_t_map[g] for g in common])
        null_rhos[perm_i] = sp_stats.spearmanr(dist_arr, perm_t_arr).statistic

        if verbose and (perm_i + 1) % 200 == 0:
            logger.info(
                "  Gradient permutation %d/%d", perm_i + 1, n_permutations
            )

    # One-sided: negative slope/rho = perturbation decays with distance
    slope_pvalue = float(
        (np.sum(null_slopes <= observed_slope) + 1) / (n_permutations + 1)
    )
    rho_pvalue = float(
        (np.sum(null_rhos <= observed_rho) + 1) / (n_permutations + 1)
    )

    # 6. Edge-quality stratification (exploratory, Bonferroni-corrected)
    stratified = None
    if edge_quality:
        stratified = _run_stratified(
            shells_sets, abs_t_stats, edge_quality, seed,
            all_t_values, all_genes, degree_bins, gene_to_bin,
            n_permutations, rng, bg_mean,
        )

    return GradientResult(
        seed_gene=seed,
        shells=tuple(observed_shells),
        slope=observed_slope,
        slope_pvalue=slope_pvalue,
        spearman_rho=observed_rho,
        spearman_pvalue=rho_pvalue,
        active_horizon=observed_horizon,
        background_mean_abs_t=bg_mean,
        n_permutations=n_permutations,
        n_genes_total=n_total,
        stratified=stratified,
    )


# ---------------------------------------------------------------------------
# Edge-quality stratification (exploratory)
# ---------------------------------------------------------------------------


def _run_stratified(
    shells_sets: dict[int, set[str]],
    abs_t_stats: dict[str, float],
    edge_quality: dict[str, str],
    seed: str,
    all_t_values: NDArray[np.float64],
    all_genes: list[str],
    degree_bins: dict[int, list[str]],
    gene_to_bin: dict[str, int],
    n_permutations: int,
    rng: np.random.Generator,
    bg_mean: float,
) -> dict[str, GradientResult] | None:
    """Run gradient analysis per edge-quality tier (Bonferroni-corrected)."""
    tiers = sorted(set(edge_quality.values()))
    raw_results: dict[str, tuple[float, float, list, set, dict]] = {}

    for tier in tiers:
        tier_shells: dict[int, set[str]] = {}
        for hop, genes in shells_sets.items():
            tier_genes = {g for g in genes if edge_quality.get(g) == tier}
            if tier_genes:
                tier_shells[hop] = tier_genes

        if not tier_shells:
            continue

        tier_observed = [
            _compute_shell_stats(tier_shells[hop], abs_t_stats, hop)
            for hop in sorted(tier_shells)
        ]
        tier_gene_set: set[str] = set()
        for s in tier_observed:
            tier_gene_set.update(s.genes)

        if len(tier_gene_set) < 5:
            continue

        tier_slope = _gradient_slope(tier_observed)

        tier_dists: dict[str, int] = {}
        for hop, genes in tier_shells.items():
            for g in genes:
                if g in abs_t_stats:
                    tier_dists[g] = hop
        tier_common = sorted(tier_dists)
        tier_dist_arr = np.array(
            [tier_dists[g] for g in tier_common], dtype=np.float64,
        )
        tier_t_arr = np.array(
            [abs_t_stats[g] for g in tier_common], dtype=np.float64,
        )
        tier_rho = (
            float(sp_stats.spearmanr(tier_dist_arr, tier_t_arr).statistic)
            if len(tier_common) >= 3
            else 0.0
        )

        # Degree-preserving permutation null for this tier
        null_slopes = np.empty(n_permutations, dtype=np.float64)
        null_rhos = np.empty(n_permutations, dtype=np.float64)
        for perm_i in range(n_permutations):
            perm_t_map = _degree_preserving_permute(
                all_genes, all_t_values, degree_bins, gene_to_bin, rng,
            )

            perm_shells = [
                _compute_shell_stats(tier_shells[hop], perm_t_map, hop)
                for hop in sorted(tier_shells)
            ]
            null_slopes[perm_i] = _gradient_slope(perm_shells)

            if len(tier_common) >= 3:
                perm_t_arr = np.array([perm_t_map[g] for g in tier_common])
                null_rhos[perm_i] = sp_stats.spearmanr(
                    tier_dist_arr, perm_t_arr,
                ).statistic
            else:
                null_rhos[perm_i] = 0.0

        slope_p = float(
            (np.sum(null_slopes <= tier_slope) + 1) / (n_permutations + 1)
        )
        rho_p = float(
            (np.sum(null_rhos <= tier_rho) + 1) / (n_permutations + 1)
        )

        raw_results[tier] = (
            tier_slope, tier_rho, tier_observed, tier_gene_set,
            {"slope_p": slope_p, "rho_p": rho_p},
        )

    if not raw_results:
        return None

    # Bonferroni correction across tiers
    n_tiers = len(raw_results)
    results: dict[str, GradientResult] = {}
    for tier, (slope, rho, observed, gene_set, pvals) in raw_results.items():
        results[tier] = GradientResult(
            seed_gene=seed,
            shells=tuple(observed),
            slope=slope,
            slope_pvalue=min(pvals["slope_p"] * n_tiers, 1.0),
            spearman_rho=rho,
            spearman_pvalue=min(pvals["rho_p"] * n_tiers, 1.0),
            active_horizon=_active_horizon(observed, bg_mean),
            background_mean_abs_t=bg_mean,
            n_permutations=n_permutations,
            n_genes_total=len(gene_set),
        )

    return results


# ---------------------------------------------------------------------------
# Edge-rewiring null (Wave 20)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewiringNullResult:
    """Result of a degree-preserving edge-rewiring null test.

    Certifies (when p < alpha): the observed gradient slope is not
    reproducible when the graph is randomly rewired while preserving
    the degree sequence.

    Does NOT certify seed-uniqueness (matched-seed control is a
    separate null, Wave 21).
    """

    # Observed and null
    observed_slope: float
    observed_coverage: float  # fraction of in-graph targets reachable from seed within max_hops
    null_slopes: NDArray[np.float64]  # valid iterations only (no imputed zeros)
    null_slopes_imputed: NDArray[np.float64]  # failed iters → 0.0 (what the p-value sees)
    pvalue: float

    # Iteration accounting
    n_rewires_requested: int
    n_rewires_ok: int
    n_rewires_failed: int

    # Graph provenance
    seed: str
    subgraph_n_nodes: int
    subgraph_n_edges: int
    seed_component_too_small: bool  # component has fewer than 30 nodes

    # Rewiring provenance
    rng_seed: int
    max_hops: int
    target_nswap: int  # nswap used for iter ≥ 1 (from iter-0 plateau)
    plateau_nswap: int  # nswap at which iter-0 plateau fired (0 if no plateau)
    iter0_plateau_reached: bool  # False → mixing may be inadequate
    accepted_fraction_iter0: float

    # Pathology diagnostics
    bimodality_coefficient: float
    bimodality_pseudo_pvalue: float
    disconnection_rate: float
    bimodality_warning: bool
    disconnection_warning: bool
    mixing_warning: bool  # True when iter-0 plateau not reached

    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        arr = self.null_slopes
        # Deciles + full array for audit (floats are cheap, ~8KB at N=999)
        deciles = (
            [float(q) for q in np.quantile(arr, np.linspace(0, 1, 11))]
            if len(arr) else None
        )
        return {
            "observed_slope": float(self.observed_slope),
            "observed_coverage": float(self.observed_coverage),
            "null_slopes": [float(s) for s in arr],  # valid iterations
            "null_slopes_imputed_preview": [
                float(s) for s in self.null_slopes_imputed[:min(50, len(self.null_slopes_imputed))]
            ],  # first 50 imputed values for audit
            "null_slopes_summary": {
                "min": float(np.min(arr)) if len(arr) else None,
                "max": float(np.max(arr)) if len(arr) else None,
                "median": float(np.median(arr)) if len(arr) else None,
                "mean": float(np.mean(arr)) if len(arr) else None,
                "std": float(np.std(arr)) if len(arr) > 1 else None,
                "deciles": deciles,
            },
            "pvalue": float(self.pvalue),
            "pvalue_formula": (
                "(sum(null_slopes_imputed <= observed_slope) + 1) / "
                "(n_rewires_requested + 1); failed iterations imputed as 0.0"
            ),
            "n_rewires_requested": self.n_rewires_requested,
            "n_rewires_ok": self.n_rewires_ok,
            "n_rewires_failed": self.n_rewires_failed,
            "seed": self.seed,
            "subgraph_n_nodes": self.subgraph_n_nodes,
            "subgraph_n_edges": self.subgraph_n_edges,
            "seed_component_too_small": bool(self.seed_component_too_small),
            "rng_seed": self.rng_seed,
            "max_hops": self.max_hops,
            "target_nswap": self.target_nswap,
            "plateau_nswap": self.plateau_nswap,
            "iter0_plateau_reached": bool(self.iter0_plateau_reached),
            "accepted_fraction_iter0": float(self.accepted_fraction_iter0),
            "bimodality_coefficient": float(self.bimodality_coefficient),
            "bimodality_pseudo_pvalue": float(self.bimodality_pseudo_pvalue),
            "disconnection_rate": float(self.disconnection_rate),
            "bimodality_warning": bool(self.bimodality_warning),
            "disconnection_warning": bool(self.disconnection_warning),
            "mixing_warning": bool(self.mixing_warning),
            "elapsed_seconds": float(self.elapsed_seconds),
        }


def _slope_and_coverage_from_rewired(
    rewired_graph,
    seed: str,
    abs_t_stats: dict[str, float],
    max_hops: int,
    aliases: dict[str, list[str]] | None = None,
) -> tuple[float | None, float]:
    """Compute the gradient slope on a rewired graph using a single BFS.

    Two modes, distinguished by whether ``aliases`` is provided:

    - **HGNC-keyed (aliases=None)**: ``abs_t_stats`` keyed by graph node
      identifiers (typically HGNC symbols).  BFS distances directly
      drive shell membership.  Used by synthetic tests and any flow
      where the measurement unit and the graph node are the same.
    - **Protein-keyed (aliases provided)**: ``abs_t_stats`` keyed by
      UniProt feature_id.  ``aliases[feature_id] = [hgnc, ...]`` maps
      each protein to its graph-side HGNC aliases.  After BFS gives
      HGNC distances, we aggregate to per-protein min-distance and
      build shells keyed by feature_id.  This is the correct
      semantics when the proteomics measures one |t| per protein and
      multiple HGNC aliases share that |t|.

    Coverage is computed at the *measurement* level: in HGNC mode it is
    the fraction of HGNC keys reachable; in protein mode it is the
    fraction of measured proteins with at least one alias reachable.

    Returns
    -------
    (slope, coverage)
        ``slope`` is None when fewer than 2 non-empty shells or fewer
        than 10 total measurable units.  ``coverage`` is in [0, 1].
    """
    from .graph_rewiring import bfs_distances_from

    rewired_nodes = set(rewired_graph.nodes())

    if aliases is None:
        # HGNC-keyed mode: keys of abs_t_stats are graph nodes
        in_graph_targets = set(abs_t_stats.keys()) & rewired_nodes
        if not in_graph_targets:
            return None, 0.0

        distances = bfs_distances_from(
            rewired_graph, seed, in_graph_targets, max_hops=max_hops,
        )
        coverage = len(distances) / len(in_graph_targets) if in_graph_targets else 0.0

        if not distances:
            return None, coverage

        shells_sets: dict[int, set[str]] = {}
        for gene, d in distances.items():
            shells_sets.setdefault(d, set()).add(gene)
    else:
        # Protein-keyed mode: keys of abs_t_stats are feature_ids;
        # aliases maps each feature_id to its HGNC alias list
        all_alias_targets: set[str] = set()
        for fid in abs_t_stats:
            for hgnc in aliases.get(fid, []):
                if hgnc in rewired_nodes:
                    all_alias_targets.add(hgnc)

        if not all_alias_targets:
            return None, 0.0

        hgnc_distances = bfs_distances_from(
            rewired_graph, seed, all_alias_targets, max_hops=max_hops,
        )
        # Aggregate: per-protein min-distance over its aliases
        prot_distances: dict[str, int] = {}
        for fid in abs_t_stats:
            best: int | None = None
            for hgnc in aliases.get(fid, []):
                d = hgnc_distances.get(hgnc)
                if d is not None and (best is None or d < best):
                    best = d
            if best is not None:
                prot_distances[fid] = best

        # Coverage at measurement level: fraction of measured proteins
        # whose alias set has any reachable HGNC node
        n_proteins_with_aliases = sum(
            1 for fid in abs_t_stats
            if any(h in rewired_nodes for h in aliases.get(fid, []))
        )
        coverage = (
            len(prot_distances) / n_proteins_with_aliases
            if n_proteins_with_aliases else 0.0
        )

        if not prot_distances:
            return None, coverage

        shells_sets = {}
        for fid, d in prot_distances.items():
            shells_sets.setdefault(d, set()).add(fid)

    shells = [
        _compute_shell_stats(shells_sets[h], abs_t_stats, h)
        for h in sorted(shells_sets)
    ]
    non_empty = [s for s in shells if s.n_genes > 0]
    total_genes = sum(s.n_genes for s in non_empty)

    if len(non_empty) < 2 or total_genes < 10:
        return None, coverage

    return _gradient_slope(shells), coverage


def run_rewiring_null(
    graph,
    seed: str,
    abs_t_stats: dict[str, float],
    observed_slope: float,
    observed_coverage: float | None = None,
    n_rewires: int = 999,
    max_hops: int = 3,
    rng_seed: int = 42,
    max_swaps_iter0: int = 500_000,
    check_every: int = 5000,
    swap_multiplier: float = 1.5,
    verbose: bool = True,
    n_jobs: int = 1,
    aliases: dict[str, list[str]] | None = None,
) -> RewiringNullResult:
    """Run the degree-preserving edge-rewiring null test.

    Iteration 0 runs in diagnostic mode to determine the number of
    swaps required for mixing.  Iterations 1..N-1 run that fixed
    swap budget with no diagnostic overhead.

    **Selection-bias handling:** Iterations that produce no valid slope
    (rewiring disconnected seed from targets, or BFS returned <2 shells)
    are treated as null_slope=0.  This is conservative under a decay
    hypothesis (observed slope is negative, so substituting 0 cannot
    make the p-value smaller than the alternative of excluding the
    iteration).  Both the raw ``null_slopes`` array and the
    failure count are reported in the result for auditability.

    **Disconnection rate** uses the observed coverage as baseline:
    a rewiring is counted as "disconnected" if its coverage drops
    below 80% of the observed coverage.  This is adaptive to graph
    scale and observed BFS-reachability.

    **Protein-level vs HGNC-level keys.**  When ``aliases`` is None,
    ``abs_t_stats`` is keyed by graph-node identifiers (typical for
    synthetic tests).  When ``aliases`` is provided, ``abs_t_stats`` is
    keyed by UniProt feature_id and ``aliases[fid] = [hgnc, ...]``
    maps each protein to its graph-side aliases.  Per-rewire BFS gives
    HGNC distances; we aggregate min-over-aliases to get protein
    distances before computing shell stats — so each protein
    measurement contributes one observation regardless of alias count.

    Parameters
    ----------
    graph
        networkx undirected Graph — the full connected component
        containing seed.  Copied per iteration; input not mutated.
    seed
        Seed gene symbol (must be present in graph).
    abs_t_stats
        Measurement-keyed |t|.  Keys are graph nodes when ``aliases`` is
        None; UniProt feature ids when ``aliases`` is provided.
    observed_slope
        The gradient slope on the un-rewired graph.
    observed_coverage
        Fraction of measured units reachable from seed within max_hops
        on the observed graph.  If None, computed here.
    n_rewires
        Number of permutations (p-floor = 1/(N+1)).  Default 999.
    max_hops
        BFS depth for shell construction per rewiring.
    rng_seed
        Base random seed; per-iteration seeds spawn deterministically.
    max_swaps_iter0
        Hard ceiling on iter-0 diagnostic mixing swaps.
    check_every
        Iter-0 plateau check frequency.
    swap_multiplier
        Iter-≥1 target_nswap = multiplier * iter-0 plateau count.
        Default 1.5 gives a safety margin.
    verbose
        Log progress.
    n_jobs
        Reserved; currently single-threaded.
    aliases
        Optional ``{feature_id: [hgnc_alias, ...]}`` map.  When
        provided, ``abs_t_stats`` is interpreted as protein-level and
        per-iteration BFS distances are aggregated min-over-aliases
        before shell construction.  When None, the function operates
        on graph-node-keyed ``abs_t_stats`` directly (legacy / synthetic
        path).

    Returns
    -------
    RewiringNullResult
    """
    import time
    from .graph_rewiring import (
        bimodality_coefficient,
        bfs_distances_from,
        rewire_maslov_sneppen,
    )

    if n_jobs != 1:
        raise NotImplementedError(
            f"n_jobs={n_jobs} is reserved but not yet implemented; "
            f"pass n_jobs=1 for now."
        )

    if seed not in graph:
        raise ValueError(
            f"Seed '{seed}' not in graph (|V|={graph.number_of_nodes()}, "
            f"|E|={graph.number_of_edges()}). Check that the subgraph "
            f"extraction captured the seed and its component."
        )

    # Hard-stop for pathologically small graphs — Maslov-Sneppen on <30 nodes
    # produces a near-identity null and the p-value is meaningless.
    if graph.number_of_nodes() < 30:
        raise ValueError(
            f"Seed component too small ({graph.number_of_nodes()} nodes) for "
            f"meaningful rewiring null. Raise max_hops or seed a different gene."
        )

    n_edges = graph.number_of_edges()
    n_nodes = graph.number_of_nodes()
    logger.info(
        "Rewiring null: seed=%s, |V|=%d, |E|=%d, N=%d, max_hops=%d",
        seed, n_nodes, n_edges, n_rewires, max_hops,
    )

    # Observed coverage: fraction of measured units reachable within max_hops.
    # In HGNC mode, "unit" = HGNC symbol; in protein mode, "unit" = UniProt
    # feature_id whose alias set contains at least one reachable HGNC node.
    graph_nodes = set(graph.nodes())
    if aliases is None:
        in_graph_units = set(abs_t_stats.keys()) & graph_nodes
        n_in_graph_units = len(in_graph_units)
        if observed_coverage is None:
            observed_distances = bfs_distances_from(
                graph, seed, in_graph_units, max_hops=max_hops,
            )
            observed_coverage = (
                len(observed_distances) / n_in_graph_units if n_in_graph_units else 0.0
            )
    else:
        # Protein mode: union of all aliases as BFS targets, then aggregate
        all_alias_targets: set[str] = set()
        proteins_with_aliases: set[str] = set()
        for fid in abs_t_stats:
            alias_in_graph = [h for h in aliases.get(fid, []) if h in graph_nodes]
            if alias_in_graph:
                proteins_with_aliases.add(fid)
                all_alias_targets.update(alias_in_graph)
        n_in_graph_units = len(proteins_with_aliases)
        if observed_coverage is None:
            hgnc_dists = bfs_distances_from(
                graph, seed, all_alias_targets, max_hops=max_hops,
            )
            n_reachable = sum(
                1 for fid in proteins_with_aliases
                if any(h in hgnc_dists for h in aliases.get(fid, []))
            )
            observed_coverage = (
                n_reachable / n_in_graph_units if n_in_graph_units else 0.0
            )
    coverage_floor = 0.8 * observed_coverage
    logger.info(
        "Observed coverage: %.1f%% of %d in-graph %s; "
        "disconnection threshold: %.1f%%",
        100 * observed_coverage, n_in_graph_units,
        "proteins" if aliases is not None else "targets",
        100 * coverage_floor,
    )

    seed_seq = np.random.SeedSequence(rng_seed)
    child_seqs = seed_seq.spawn(n_rewires)

    # Iteration 0: diagnostic mode to determine nswap target
    t0 = time.time()
    rng_0 = np.random.default_rng(child_seqs[0])
    rewired_0, diag_0 = rewire_maslov_sneppen(
        graph, rng_0,
        target_nswap=None,  # diagnostic mode
        max_swaps=max_swaps_iter0,
        check_every=check_every,
    )
    plateau_nswap = diag_0.plateau_swaps or max_swaps_iter0
    target_nswap = int(plateau_nswap * swap_multiplier)
    logger.info(
        "Iter 0: plateau at %d swaps; target for iters ≥1 = %d swaps",
        plateau_nswap, target_nswap,
    )

    slope_0, coverage_0 = _slope_and_coverage_from_rewired(
        rewired_0, seed, abs_t_stats, max_hops, aliases=aliases,
    )
    null_slopes_raw: list[float | None] = [slope_0]
    n_disconnected = 0
    if coverage_0 < coverage_floor:
        n_disconnected += 1

    # Iterations 1..N-1: fixed-swap mode, no diagnostic
    for i in range(1, n_rewires):
        rng_i = np.random.default_rng(child_seqs[i])
        rewired, _ = rewire_maslov_sneppen(
            graph, rng_i,
            target_nswap=target_nswap,
        )
        slope, coverage = _slope_and_coverage_from_rewired(
            rewired, seed, abs_t_stats, max_hops, aliases=aliases,
        )
        null_slopes_raw.append(slope)
        if coverage < coverage_floor:
            n_disconnected += 1

        if verbose and (i + 1) % max(1, n_rewires // 10) == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_rewires - i - 1)
            n_ok = sum(1 for s in null_slopes_raw if s is not None)
            logger.info(
                "  rewiring null %d/%d (%.0f%%)  elapsed=%.0fs  eta=%.0fs  "
                "slope_ok=%d  disc=%d",
                i + 1, n_rewires, 100.0 * (i + 1) / n_rewires,
                elapsed, eta, n_ok, n_disconnected,
            )

    elapsed = time.time() - t0

    # Selection-bias handling: treat failed iterations as null_slope=0.
    # Under the decay hypothesis, observed_slope < 0 so substituting 0
    # cannot make the p-value smaller than the alternative of exclusion.
    null_slopes_imputed = np.array(
        [s if s is not None else 0.0 for s in null_slopes_raw],
        dtype=np.float64,
    )
    null_slopes_valid = np.array(
        [s for s in null_slopes_raw if s is not None],
        dtype=np.float64,
    )
    n_ok = len(null_slopes_valid)
    n_failed = n_rewires - n_ok

    # One-sided p-value with conservative failure imputation
    pvalue = float(
        (np.sum(null_slopes_imputed <= observed_slope) + 1) / (n_rewires + 1)
    )

    # Pathology diagnostics on the VALID (non-imputed) null slopes
    if n_ok >= 4:
        bc, bc_pseudo_p = bimodality_coefficient(null_slopes_valid)
    else:
        bc, bc_pseudo_p = 0.0, 1.0
    disc_rate = n_disconnected / n_rewires if n_rewires else 0.0

    # Warning thresholds
    bimodality_warning = bc > 5.0 / 9.0
    disconnection_warning = disc_rate > 0.05

    if bimodality_warning:
        logger.warning(
            "Null-slope distribution shows bimodality (BC=%.3f > 0.555); "
            "mixing may be inadequate — interpret p=%.3f with caution.",
            bc, pvalue,
        )
    if disconnection_warning:
        logger.warning(
            "%.1f%% of rewirings had coverage < 80%% of observed; "
            "failed iterations imputed as null_slope=0 (conservative).",
            100 * disc_rate,
        )

    iter0_plateau_reached = diag_0.plateau_swaps is not None
    mixing_warning = not iter0_plateau_reached
    seed_component_too_small = n_nodes < 30

    if mixing_warning:
        logger.warning(
            "Iter-0 mixing plateau was NOT reached within %d swaps; "
            "target_nswap=%d may reflect inadequate mixing. "
            "Interpret p=%.3f with caution.",
            max_swaps_iter0, target_nswap, pvalue,
        )

    return RewiringNullResult(
        observed_slope=observed_slope,
        observed_coverage=float(observed_coverage),
        null_slopes=null_slopes_valid,
        null_slopes_imputed=null_slopes_imputed,
        pvalue=pvalue,
        n_rewires_requested=n_rewires,
        n_rewires_ok=n_ok,
        n_rewires_failed=n_failed,
        seed=seed,
        subgraph_n_nodes=n_nodes,
        subgraph_n_edges=n_edges,
        seed_component_too_small=seed_component_too_small,
        rng_seed=rng_seed,
        max_hops=max_hops,
        target_nswap=target_nswap,
        plateau_nswap=plateau_nswap,
        iter0_plateau_reached=iter0_plateau_reached,
        accepted_fraction_iter0=float(diag_0.accepted_fraction),
        bimodality_coefficient=float(bc),
        bimodality_pseudo_pvalue=float(bc_pseudo_p),
        disconnection_rate=float(disc_rate),
        bimodality_warning=bimodality_warning,
        disconnection_warning=disconnection_warning,
        mixing_warning=mixing_warning,
        elapsed_seconds=float(elapsed),
    )
