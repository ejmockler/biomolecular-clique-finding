"""
Bootstrap stability assessment for regulatory cliques.

Provides:
- Bootstrap resampling of samples
- Clique stability scoring (fraction of bootstraps where clique appears)
- Consensus cliques (stable across bootstrap iterations)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, FrozenSet
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


@dataclass
class StableClique:
    """A clique with bootstrap stability score."""
    genes: FrozenSet[str]
    stability: float  # Fraction of bootstraps where clique appears (0-1)
    mean_correlation: float  # Mean pairwise correlation in full data
    bootstrap_correlations: List[float] = field(default_factory=list)  # Correlations per bootstrap

    @property
    def is_stable(self) -> bool:
        """Clique is stable if appears in >= 80% of bootstraps."""
        return self.stability >= 0.8

    def __hash__(self):
        return hash(self.genes)


def bootstrap_clique_stability(
    matrix: 'BioMatrix',
    genes: Set[str],
    condition: str,
    min_correlation: float = 0.7,
    min_clique_size: int = 3,
    n_bootstrap: int = 100,
    n_jobs: int = 4,
    random_state: Optional[int] = None
) -> List[StableClique]:
    """
    Assess clique stability via bootstrap resampling.

    Args:
        matrix: BioMatrix with expression data
        genes: Set of genes to analyze (e.g., regulator targets)
        condition: Condition string for stratification
        min_correlation: Minimum correlation for clique edges
        min_clique_size: Minimum genes in a clique
        n_bootstrap: Number of bootstrap iterations
        n_jobs: Parallel workers
        random_state: Random seed for reproducibility

    Returns:
        List of StableClique objects with stability scores
    """
    from cliquefinder.knowledge.clique_validator import CliqueValidator

    if random_state is not None:
        np.random.seed(random_state)

    # Get condition mask
    validator = CliqueValidator(matrix, stratify_by=None)
    condition_mask = validator._compute_condition_mask_internal(condition)
    condition_indices = np.where(condition_mask)[0]
    n_samples = len(condition_indices)

    logger.info(f"Bootstrap stability: {n_bootstrap} iterations, {n_samples} samples, {len(genes)} genes")

    # Track clique frequencies across bootstraps
    clique_counts: Dict[FrozenSet[str], int] = {}
    clique_correlations: Dict[FrozenSet[str], List[float]] = {}

    def run_bootstrap(b: int) -> List[FrozenSet[str]]:
        """Single bootstrap iteration."""
        # Resample with replacement
        boot_indices = np.random.choice(condition_indices, size=n_samples, replace=True)

        # Create bootstrap matrix (only need the subset)
        boot_data = matrix.data[:, boot_indices]

        # Compute correlation matrix for this bootstrap
        gene_indices = [i for i, g in enumerate(matrix.feature_ids) if g in genes]
        if len(gene_indices) < min_clique_size:
            return []

        gene_data = boot_data[gene_indices, :]
        gene_names = [matrix.feature_ids[i] for i in gene_indices]

        # Compute correlations
        corr_matrix = np.corrcoef(gene_data)

        # Build graph and find cliques
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(gene_names)

        for i, g1 in enumerate(gene_names):
            for j, g2 in enumerate(gene_names):
                if i < j and abs(corr_matrix[i, j]) >= min_correlation:
                    G.add_edge(g1, g2, weight=corr_matrix[i, j])

        # Find cliques (use greedy for speed)
        cliques = []
        for clique_nodes in nx.find_cliques(G):
            if len(clique_nodes) >= min_clique_size:
                cliques.append(frozenset(clique_nodes))
                if len(cliques) >= 1000:  # Limit per bootstrap
                    break

        return cliques

    # Run bootstraps in parallel
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        all_cliques = list(executor.map(run_bootstrap, range(n_bootstrap)))

    # Aggregate clique frequencies
    for boot_cliques in all_cliques:
        for clique in boot_cliques:
            clique_counts[clique] = clique_counts.get(clique, 0) + 1

    # Compute full-data correlations for stable cliques
    full_condition_data = matrix.data[:, condition_mask]
    gene_indices = {g: i for i, g in enumerate(matrix.feature_ids) if g in genes}

    stable_cliques = []
    for clique, count in clique_counts.items():
        stability = count / n_bootstrap

        # Compute mean correlation in full data
        clique_genes = list(clique)
        if all(g in gene_indices for g in clique_genes):
            indices = [gene_indices[g] for g in clique_genes]
            clique_data = full_condition_data[indices, :]
            corr = np.corrcoef(clique_data)
            # Mean of upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(len(clique_genes), k=1)
            mean_corr = np.mean(corr[triu_indices])
        else:
            mean_corr = 0.0

        stable_cliques.append(StableClique(
            genes=clique,
            stability=stability,
            mean_correlation=mean_corr
        ))

    # Sort by stability (highest first)
    stable_cliques.sort(key=lambda c: (-c.stability, -len(c.genes)))

    logger.info(f"Found {len(stable_cliques)} cliques, {sum(1 for c in stable_cliques if c.is_stable)} stable (>=80%)")

    return stable_cliques


def filter_stable_cliques(
    cliques: List[StableClique],
    min_stability: float = 0.8
) -> List[StableClique]:
    """Filter to only stable cliques."""
    return [c for c in cliques if c.stability >= min_stability]


def summarize_stability(cliques: List[StableClique]) -> pd.DataFrame:
    """Create summary DataFrame of clique stability."""
    return pd.DataFrame([
        {
            'genes': ','.join(sorted(c.genes)),
            'size': len(c.genes),
            'stability': c.stability,
            'mean_correlation': c.mean_correlation,
            'is_stable': c.is_stable
        }
        for c in cliques
    ])
