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
