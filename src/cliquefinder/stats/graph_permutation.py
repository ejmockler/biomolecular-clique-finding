"""
Node-label permutation test on the INDRA knowledge graph.

Tests whether the INDRA graph structure itself generates false enrichment
signals, or whether the specific gene identities matter.  The methodology
(suggested by Karen Sachs):

1. Start with the INDRA regulatory subgraph (regulators -> targets).
2. Test the real target gene set via ROAST -- this is the real p-value.
3. For each permutation, pick a random eligible regulator, take its
   neighborhood, and map targets through a random permutation of
   *measurable* gene labels (XVI-2: restricted to resolvable genes).
4. Resolve the permuted targets against proteomics data and test via ROAST.
5. Repeat N times.  Build a null distribution and ask: does the real
   p-value stand out?

XVI-1: Instead of shuffling all labels and discarding when the query
lands on a non-regulator, we sample directly from regulators.  This is
statistically equivalent but wastes zero iterations.

XVI-2: Restrict the permutation space to genes that are both in the graph
AND resolvable (symbol_to_feature AND engine.gene_to_idx).  This prevents
set size contraction: every permuted symbol is guaranteed to resolve.

Warning convention:
    warnings.warn() -- user-facing (convergence, sparse graph)
    logger.warning() -- operator-facing (fallback, missing data)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class GraphPermutationResult:
    """Result of graph node-label permutation null test.

    Attributes:
        target_pvalue: p-value of the actual target gene set.
        target_set_id: Identifier for the target gene set.
        target_set_size: Number of genes in the target set.
        control_pvalues: Array of p-values from permuted graphs (NaN filtered).
        fpr: False positive rate (fraction of controls with p < alpha).
        alpha: Significance threshold used for FPR.
        target_percentile: Percentile rank of target p-value among controls
            (0 = most significant, 100 = least).
        median_control_pvalue: Median p-value across valid permutations.
        mean_control_pvalue: Mean p-value across valid permutations.
        n_permutations: Total number of permutations requested.
        n_valid_permutations: Number of permutations that produced a p-value.
        n_eligible_regulators: Number of regulators with >= 2 resolvable targets.
        n_excluded_regulators: Number of regulators excluded (< 2 resolvable targets).
        median_control_set_size: Median number of genes in permuted sets.
        graph_stats: Summary statistics of the input graph.
    """

    target_pvalue: float
    target_set_id: str
    target_set_size: int
    control_pvalues: NDArray[np.float64]
    fpr: float
    alpha: float
    target_percentile: float
    median_control_pvalue: float
    mean_control_pvalue: float
    n_permutations: int
    n_valid_permutations: int
    n_eligible_regulators: int
    n_excluded_regulators: int
    median_control_set_size: int
    graph_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = {
            "target_set_id": self.target_set_id,
            "target_set_size": self.target_set_size,
            "target_pvalue": self.target_pvalue,
            "fpr": self.fpr,
            "alpha": self.alpha,
            "target_percentile": self.target_percentile,
            "median_control_pvalue": self.median_control_pvalue,
            "mean_control_pvalue": self.mean_control_pvalue,
            "n_permutations": self.n_permutations,
            "n_valid_permutations": self.n_valid_permutations,
            "n_eligible_regulators": self.n_eligible_regulators,
            "n_excluded_regulators": self.n_excluded_regulators,
            "median_control_set_size": self.median_control_set_size,
            "graph_stats": self.graph_stats,
        }

        if len(self.control_pvalues) > 0:
            d["control_pvalue_quantiles"] = {
                "q05": float(np.percentile(self.control_pvalues, 5)),
                "q25": float(np.percentile(self.control_pvalues, 25)),
                "q50": float(np.percentile(self.control_pvalues, 50)),
                "q75": float(np.percentile(self.control_pvalues, 75)),
                "q95": float(np.percentile(self.control_pvalues, 95)),
            }

        return d


def _compute_graph_stats(
    adjacency: dict[str, list[str]],
    n_measurable_nodes: int = 0,
) -> dict:
    """Compute summary statistics of the adjacency graph."""
    all_targets = set()
    total_edges = 0
    for targets in adjacency.values():
        all_targets.update(targets)
        total_edges += len(targets)

    all_nodes = set(adjacency.keys()) | all_targets
    n_regulators = len(adjacency)
    n_nodes = len(all_nodes)
    mean_degree = total_edges / n_regulators if n_regulators > 0 else 0.0

    stats = {
        "n_nodes": n_nodes,
        "n_edges": total_edges,
        "n_regulators": n_regulators,
        "mean_degree": round(mean_degree, 2),
    }
    if n_measurable_nodes > 0:
        stats["n_measurable_nodes"] = n_measurable_nodes

    return stats


def run_graph_permutation_null(
    engine,  # RotationTestEngine -- already fitted
    target_gene_ids: list[str],
    target_set_id: str,
    adjacency: dict[str, list[str]],
    symbol_to_feature: dict[str, str],
    n_permutations: int = 100,
    alpha: float = 0.05,
    seed: int | None = None,
    verbose: bool = True,
) -> GraphPermutationResult:
    """
    Run node-label permutation test on the INDRA knowledge graph.

    Samples random regulators, maps their targets through a permutation of
    measurable gene labels, resolves against proteomics data, and tests via
    ROAST.  Repeats N times to build a null distribution.

    Args:
        engine: RotationTestEngine that has already been fitted (fit() called).
        target_gene_ids: Feature IDs (e.g. UniProt) of the real target set.
        target_set_id: Identifier for the target set (e.g. "C9ORF72_targets").
        adjacency: Regulator-centric adjacency dict from the INDRA subgraph.
            Keys are regulator gene symbols, values are lists of target
            gene symbols.
        symbol_to_feature: Mapping from gene symbol to feature ID in the
            proteomics data.
        n_permutations: Number of label shuffles (default: 100).
        alpha: Significance threshold for FPR calculation (default: 0.05).
        seed: Random seed for reproducibility.
        verbose: Print progress updates.

    Returns:
        GraphPermutationResult with null distribution and calibration stats.

    Raises:
        ValueError: If adjacency is empty.
        RuntimeError: If engine is not fitted, or if no eligible regulators,
            or if all permutations fail.
    """
    if not adjacency:
        raise ValueError("Adjacency dict is empty -- no graph to permute.")

    if not engine._fitted:
        raise RuntimeError(
            "RotationTestEngine must be fitted before running graph "
            "permutation test. Call engine.fit() first."
        )

    # Collect all unique gene names in the graph
    all_gene_names_set: set[str] = set()
    for regulator, targets in adjacency.items():
        all_gene_names_set.add(regulator)
        all_gene_names_set.update(targets)

    # XVI-2: Compute the resolvable subset -- genes in the graph that can
    # be resolved to features measured in the proteomics data.
    resolvable = {
        g for g in all_gene_names_set
        if g in symbol_to_feature
        and symbol_to_feature[g] in engine.gene_to_idx
    }

    graph_stats = _compute_graph_stats(adjacency, n_measurable_nodes=len(resolvable))

    # XVI-1: Pre-compute eligible regulators and their resolvable target lists.
    # Only regulators with >= 2 resolvable targets are eligible (ROAST needs >= 2).
    eligible_regulators: list[str] = []
    regulator_resolvable_targets: dict[str, list[str]] = {}
    n_total_regulators = len(adjacency)

    for reg, targets in adjacency.items():
        resolvable_targets = [t for t in targets if t in resolvable]
        if len(resolvable_targets) >= 2:
            eligible_regulators.append(reg)
            regulator_resolvable_targets[reg] = resolvable_targets

    n_eligible = len(eligible_regulators)
    n_excluded = n_total_regulators - n_eligible

    if n_eligible == 0:
        raise RuntimeError(
            f"No eligible regulators: all {n_total_regulators} regulators "
            f"have < 2 resolvable targets. Cannot build a null distribution. "
            f"(n_resolvable={len(resolvable)}, "
            f"n_nodes={graph_stats['n_nodes']}, "
            f"n_regulators={n_total_regulators})"
        )

    # Find target genes in the measured universe
    target_in_data = [g for g in target_gene_ids if g in engine.gene_to_idx]
    target_size = len(target_in_data)

    if target_size == 0:
        raise ValueError("No target genes found in the measured gene universe.")

    # Run ROAST on actual target set
    target_result = engine.test_gene_set(
        gene_set=target_in_data,
        gene_set_id=target_set_id,
    )
    target_pvalue = float(target_result.p_values.get("msq", {}).get("mixed", 1.0))

    # XVI-2: Build the sorted resolvable array for permutation
    resolvable_sorted = sorted(resolvable)
    n_resolvable = len(resolvable_sorted)

    if verbose:
        print(
            f"Graph permutation test: {n_permutations} permutations, "
            f"{graph_stats['n_nodes']} nodes, {graph_stats['n_edges']} edges"
        )
        print(f"  Target set p-value: {target_pvalue:.4f}")
        print(
            f"  Eligible regulators: {n_eligible}/{n_total_regulators} "
            f"(excluded {n_excluded} with < 2 resolvable targets)"
        )
        print(f"  Resolvable genes: {n_resolvable}/{len(all_gene_names_set)}")

    rng = np.random.default_rng(seed)

    # Run permutations
    control_pvalues = np.full(n_permutations, np.nan)
    set_sizes: list[int] = []

    for i in range(n_permutations):
        if verbose and (i + 1) % 25 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}...")

        # XVI-1: Sample a random eligible regulator
        reg = eligible_regulators[rng.integers(n_eligible)]
        reg_targets = regulator_resolvable_targets[reg]

        # XVI-2: Permute only resolvable gene names
        permuted_resolvable = rng.permutation(resolvable_sorted)
        forward_map = {
            resolvable_sorted[j]: permuted_resolvable[j]
            for j in range(n_resolvable)
        }

        # Map targets through the permutation
        fake_target_symbols = [forward_map[t] for t in reg_targets]

        # Resolve to feature IDs -- guaranteed to succeed by construction
        fake_feature_ids = [
            symbol_to_feature[sym]
            for sym in fake_target_symbols
            if sym in symbol_to_feature
            and symbol_to_feature[sym] in engine.gene_to_idx
        ]

        set_sizes.append(len(fake_feature_ids))

        if len(fake_feature_ids) < 2:
            # Should not happen by construction, but guard anyway
            continue

        try:
            perm_result = engine.test_gene_set(
                gene_set=fake_feature_ids,
                gene_set_id=f"graph_perm_{i}",
            )
            control_pvalues[i] = float(
                perm_result.p_values.get("msq", {}).get("mixed", 1.0)
            )
        except Exception as e:
            if i == 0:
                logger.warning("Graph permutation 0 failed: %s", e)
            control_pvalues[i] = np.nan

    # Filter NaN
    valid_controls = control_pvalues[~np.isnan(control_pvalues)]
    n_valid = len(valid_controls)

    if n_valid == 0:
        raise RuntimeError(
            f"All {n_permutations} graph permutations produced failed "
            f"ROAST results. (n_eligible_regulators={n_eligible}, "
            f"n_resolvable={n_resolvable})."
        )

    # Compute statistics
    fpr = float(np.sum(valid_controls < alpha)) / n_valid
    target_percentile = float(np.sum(valid_controls <= target_pvalue)) / n_valid * 100
    median_control = float(np.median(valid_controls))
    mean_control = float(np.mean(valid_controls))
    median_set_size = int(np.median(set_sizes)) if set_sizes else 0

    if verbose:
        print(f"  Valid permutations: {n_valid}/{n_permutations}")
        print(f"  FPR (p < {alpha}): {fpr:.3f}")
        print(f"  Target percentile: {target_percentile:.1f}%")
        print(f"  Median control p-value: {median_control:.4f}")
        print(f"  Median control set size: {median_set_size}")

    return GraphPermutationResult(
        target_pvalue=target_pvalue,
        target_set_id=target_set_id,
        target_set_size=target_size,
        control_pvalues=valid_controls,
        fpr=fpr,
        alpha=alpha,
        target_percentile=target_percentile,
        median_control_pvalue=median_control,
        mean_control_pvalue=mean_control,
        n_permutations=n_permutations,
        n_valid_permutations=n_valid,
        n_eligible_regulators=n_eligible,
        n_excluded_regulators=n_excluded,
        median_control_set_size=median_set_size,
        graph_stats=graph_stats,
    )
