#!/usr/bin/env python3
"""
Regulatory Coherence Analysis Module

A statistically rigorous, computationally tractable approach to identifying
coherent regulatory modules from TF->target relationships and expression data.

Design Philosophy (informed by computational biology best practices):
1. REPLACE cliques with community detection (Louvain/Leiden) - O(n log n)
2. USE soft thresholding instead of hard cutoffs
3. SEPARATE positive and negative correlations (distinct biological meaning)
4. BOOTSTRAP correlations to quantify uncertainty
5. PERMUTATION null to establish statistical significance
6. FILTER for expressed genes to reduce noise
7. CORRECT for multiple testing

Key Classes:
- CoherenceAnalyzer: Main orchestrator for regulatory coherence analysis
- CommunityResult: A detected gene community with statistics
- StratifiedCoherenceResult: Per-condition coherence analysis
- DifferentialCoherenceResult: Cross-condition comparisons

Author: Refactored based on brutalist critique of naive clique approach
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

# Suppress FutureWarnings from community detection libraries
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import community as community_louvain  # python-louvain package
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

try:
    import leidenalg
    import igraph as ig
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False

logger = logging.getLogger(__name__)


class CommunityMethod(Enum):
    """Available community detection methods."""
    LOUVAIN = "louvain"  # O(n log n), well-tested
    LEIDEN = "leiden"     # Improved Louvain, better guarantees
    HIERARCHICAL = "hierarchical"  # Agglomerative clustering
    GREEDY = "greedy"     # Greedy modularity optimization


class CorrelationSign(Enum):
    """Sign of correlation - biologically distinct."""
    POSITIVE = "positive"  # Co-activation
    NEGATIVE = "negative"  # Antagonistic regulation


@dataclass
class GeneCorrelationChange:
    """Statistical comparison of gene-gene correlation between two conditions."""
    gene_i: str
    gene_j: str
    corr_a: float  # Correlation in condition A
    corr_b: float  # Correlation in condition B
    z_diff: float  # Z-score of difference (Fisher z-transform)
    p_value: float  # Two-tailed p-value
    fdr_adjusted_p: Optional[float] = None  # FDR-corrected p-value

    @property
    def corr_delta(self) -> float:
        """Absolute change in correlation."""
        return self.corr_b - self.corr_a

    @property
    def is_significant(self) -> bool:
        """Check if difference is statistically significant after FDR correction."""
        if self.fdr_adjusted_p is not None:
            return self.fdr_adjusted_p < 0.05
        return self.p_value < 0.05


@dataclass
class CommunityResult:
    """A detected gene community with associated statistics."""
    community_id: int
    genes: Set[str]

    # Statistics within community
    mean_correlation: float
    min_correlation: float
    max_correlation: float
    density: float  # Fraction of possible edges present

    # Significance
    modularity_contribution: float
    bootstrap_stability: Optional[float] = None  # Fraction of bootstraps containing this community
    # RC-VIII-4: This is a GRAPH-LEVEL permutation p-value (whole-graph
    # modularity vs null), NOT a per-community p-value.  All communities of
    # the same sign share the same value.  Use ``is_significant`` for
    # per-community significance, which also requires density/size thresholds.
    permutation_pvalue: Optional[float] = None

    # Metadata
    regulator_name: Optional[str] = None
    condition: Optional[str] = None
    correlation_sign: CorrelationSign = CorrelationSign.POSITIVE

    @property
    def size(self) -> int:
        return len(self.genes)

    @property
    def is_significant(self) -> bool:
        """Check if community passes significance thresholds.

        RC-VIII-4: ``permutation_pvalue`` is a graph-level quantity (whole-graph
        modularity vs null), shared by ALL communities of the same sign.  A
        small noisy 3-gene community should not be deemed significant solely
        because the overall graph modularity is significant.  Therefore we
        require BOTH graph-level significance AND per-community quality
        (density >= 0.5, size >= 3).
        """
        graph_sig = (
            self.permutation_pvalue is not None
            and self.permutation_pvalue < 0.05
        )
        community_quality = self.size >= 3 and self.density >= 0.5
        if self.permutation_pvalue is not None:
            return graph_sig and community_quality
        # Fallback when no permutation test was run
        return community_quality


@dataclass
class StratifiedCoherenceResult:
    """Coherence analysis for a single condition."""
    condition: str
    n_samples: int

    # Communities found
    positive_communities: List[CommunityResult]
    negative_communities: List[CommunityResult]

    # Global statistics
    modularity_positive: float
    modularity_negative: float

    # Gene coverage
    genes_in_positive: Set[str]
    genes_in_negative: Set[str]
    genes_unclustered: Set[str]

    @property
    def total_communities(self) -> int:
        return len(self.positive_communities) + len(self.negative_communities)

    @property
    def largest_positive_community(self) -> Optional[CommunityResult]:
        if not self.positive_communities:
            return None
        return max(self.positive_communities, key=lambda c: c.size)

    @property
    def largest_negative_community(self) -> Optional[CommunityResult]:
        if not self.negative_communities:
            return None
        return max(self.negative_communities, key=lambda c: c.size)


@dataclass
class DifferentialCoherenceResult:
    """Compare coherence between two conditions using Fisher Z-transformation."""
    condition_a: str
    condition_b: str
    n_samples_a: int
    n_samples_b: int

    # Communities unique to each condition
    communities_only_a: List[CommunityResult]
    communities_only_b: List[CommunityResult]
    communities_shared: List[Tuple[CommunityResult, CommunityResult]]  # (a, b) pairs

    # Gene-level correlation changes
    significant_changes: List[GeneCorrelationChange]  # FDR < 0.05 and |Δr| >= threshold
    all_changes: List[GeneCorrelationChange]  # All tested gene pairs

    # Differential statistics
    jaccard_similarity: float  # Gene overlap between conditions
    n_significant_increases: int  # Correlations stronger in B
    n_significant_decreases: int  # Correlations weaker in B

    # Significance of differential coherence
    permutation_pvalue: Optional[float] = None

    @property
    def n_significant_changes(self) -> int:
        """Total number of significant correlation changes."""
        return len(self.significant_changes)

    @property
    def mean_correlation_delta(self) -> float:
        """Mean absolute correlation change across significant pairs."""
        if not self.significant_changes:
            return 0.0
        return float(np.mean([abs(c.corr_delta) for c in self.significant_changes]))


@dataclass
class PermutationResult:
    """Results from permutation-based significance testing."""
    observed_modularity: float
    null_modularities: np.ndarray
    p_value: float
    z_score: float
    n_permutations: int

    @property
    def is_significant(self) -> bool:
        """Check if observed modularity is significantly higher than null."""
        return self.p_value < 0.05

    @property
    def null_mean(self) -> float:
        """Mean of null distribution."""
        return float(np.mean(self.null_modularities))

    @property
    def null_std(self) -> float:
        """Standard deviation of null distribution."""
        return float(np.std(self.null_modularities))


@dataclass
class CoherenceConfig:
    """
    Configuration for coherence analysis.

    Magic Number Documentation:
    ---------------------------
    soft_threshold_power (default: 6.0):
        WGCNA-style soft thresholding power. This is the standard WGCNA default
        and should be tuned via the scale-free topology criterion for optimal
        network properties.

        Why 6.0?
        - WGCNA uses soft thresholding: weight = |correlation|^β
        - Optimal β satisfies scale-free topology (power-law degree distribution)
        - Typical range: β ∈ [4, 20] for biological networks
        - β = 6 is a starting point; optimize using pickSoftThreshold() or similar

        References:
            Zhang, B., & Horvath, S. (2005). A general framework for weighted
            gene co-expression network analysis. Statistical Applications in
            Genetics and Molecular Biology, 4(1), Article17.

    min_expression_percentile (default: 20.0):
        Filters genes below 20th percentile of mean expression. Commonly used
        default to remove lowly expressed noise. Validate for your data:
        - Bulk RNA-seq: 10-25th percentile typical
        - Single-cell: 30-50th percentile (higher due to sparsity)
        - High-quality: 5-10th percentile (less aggressive)

    min_variance_percentile (default: 10.0):
        Filters genes below 10th percentile of variance. Removes genes with
        minimal variation. Validate for your data:
        - Highly variable data: 20-30th percentile
        - Stable expression: 5-10th percentile
        - Note: Too aggressive may remove housekeeping genes
    """
    # Community detection
    method: CommunityMethod = CommunityMethod.LOUVAIN
    resolution: float = 1.0  # Louvain/Leiden resolution parameter

    # Soft thresholding (see docstring for rationale)
    soft_threshold_power: float = 6.0  # WGCNA default, tune via scale-free topology
    min_edge_weight: float = 0.1  # Minimum weight to keep edge

    # Filtering (see docstring for rationale)
    min_expression_percentile: float = 20.0  # Filter low-expression genes
    min_variance_percentile: float = 10.0    # Filter low-variance genes

    # Statistical rigor
    n_bootstrap: int = 100  # Bootstrap iterations for stability
    n_permutations: int = 100  # Permutations for null distribution
    fdr_threshold: float = 0.05  # Benjamini-Hochberg FDR

    # Sample requirements
    min_samples_per_condition: int = 20  # Minimum samples for correlation

    # Community filtering
    min_community_size: int = 3
    min_community_density: float = 0.3


def fisher_z_test(r1: float, r2: float, n1: int, n2: int) -> Tuple[float, float]:
    """
    Compare two correlations using Fisher z-transformation.

    The Fisher z-transformation stabilizes the variance of correlation coefficients,
    making them suitable for statistical comparison. This is the standard method
    for testing whether two correlations are significantly different.

    Mathematical formulation:
        z = arctanh(r) = 0.5 * ln((1+r)/(1-r))  # Fisher transform
        SE(z1 - z2) = sqrt(1/(n1-3) + 1/(n2-3))  # Standard error
        Z = (z1 - z2) / SE  # Test statistic (follows standard normal)
        p = 2 * (1 - Φ(|Z|))  # Two-tailed p-value

    Args:
        r1: Correlation coefficient in condition 1 (range: [-1, 1])
        r2: Correlation coefficient in condition 2 (range: [-1, 1])
        n1: Sample size in condition 1 (must be >= 4)
        n2: Sample size in condition 2 (must be >= 4)

    Returns:
        (z_score, p_value): Test statistic and two-tailed p-value

    References:
        - Fisher, R. A. (1915). Frequency distribution of the values of the correlation
          coefficient in samples from an indefinitely large population. Biometrika.
        - Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003).
          Applied multiple regression/correlation analysis for the behavioral sciences.
    """
    # Clip correlations to avoid arctanh(±1) = ±∞
    # Biologically, r=±1 implies perfect linear relationship, which is rare
    r1_clipped = np.clip(r1, -0.999, 0.999)
    r2_clipped = np.clip(r2, -0.999, 0.999)

    # Fisher z-transformation: z = arctanh(r) = 0.5 * ln((1+r)/(1-r))
    z1 = np.arctanh(r1_clipped)
    z2 = np.arctanh(r2_clipped)

    # Standard error of z1 - z2
    # Degrees of freedom: n - 3 (corrects for estimation of mean and variance)
    if n1 < 4 or n2 < 4:
        # Too few samples for reliable inference
        return 0.0, 1.0

    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))

    # Test statistic: follows standard normal under null hypothesis
    z_score = (z1 - z2) / se if se > 0 else 0.0

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return float(z_score), float(p_value)


def _community_modularity(G: 'nx.Graph', community_genes: Set[str]) -> float:
    """Compute modularity contribution of a single community.

    Uses a 2-partition (community vs rest) to measure how much this
    community contributes to the overall modularity score.

    Args:
        G: The full graph
        community_genes: Set of genes in this community

    Returns:
        Modularity of the 2-partition {community, rest}
    """
    comm_set = set(community_genes)
    rest_set = set(G.nodes()) - comm_set
    if not rest_set:
        return 0.0
    try:
        return nx_community.modularity(G, [comm_set, rest_set], weight='weight')
    except (ZeroDivisionError, nx.NetworkXError):
        return 0.0


class CoherenceAnalyzer:
    """
    Statistically rigorous regulatory coherence analyzer.

    Replaces naive clique enumeration with:
    1. Soft-thresholded correlation networks
    2. Community detection (Louvain/Leiden)
    3. Sign-separated analysis (positive vs negative correlations)
    4. Bootstrap stability assessment
    5. Permutation-based significance testing

    Usage:
        analyzer = CoherenceAnalyzer(expression_matrix, config)
        result = analyzer.analyze_coherence(
            genes=indra_targets,
            condition="CASE_Male",
            regulator_name="TP53"
        )
    """

    def __init__(
        self,
        matrix: 'BioMatrix',
        stratify_by: Optional[List[str]] = None,
        config: Optional[CoherenceConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize coherence analyzer.

        Args:
            matrix: BioMatrix with expression data
            stratify_by: Columns to stratify by (e.g., ['phenotype', 'Sex'])
            config: Analysis configuration
            seed: Random seed for reproducibility (uses np.random.default_rng)
        """
        self.matrix = matrix
        self.stratify_by = stratify_by or []
        self.config = config or CoherenceConfig()
        self._rng = np.random.default_rng(seed)

        # Precompute sample groups
        self._sample_groups = self._compute_sample_groups()

        # Validate dependencies
        self._validate_dependencies()

    def _get_community_seed(self) -> int:
        """Get a deterministic seed for community detection from self._rng."""
        return int(self._rng.integers(0, 2**31))

    def _validate_dependencies(self):
        """Check for required community detection libraries."""
        if self.config.method == CommunityMethod.LOUVAIN and not HAS_LOUVAIN:
            if HAS_NETWORKX:
                logger.warning("python-louvain not installed, falling back to NetworkX Louvain")
            else:
                raise ImportError("Install python-louvain or networkx for Louvain")

        if self.config.method == CommunityMethod.LEIDEN and not HAS_LEIDEN:
            raise ImportError("Install leidenalg and igraph for Leiden algorithm")

    def _compute_sample_groups(self) -> Dict[str, np.ndarray]:
        """Precompute sample indices for each condition."""
        groups = {}

        if not self.stratify_by:
            groups['all'] = np.arange(self.matrix.n_samples)
            return groups

        metadata = self.matrix.sample_metadata
        for group_key, group_df in metadata.groupby(self.stratify_by, observed=True):
            if isinstance(group_key, tuple):
                condition = '||'.join(str(v) for v in group_key)
            else:
                condition = str(group_key)

            # Get sample indices
            sample_mask = self.matrix.sample_ids.isin(group_df.index)
            groups[condition] = np.where(sample_mask)[0]

        return groups

    def get_available_conditions(self) -> List[str]:
        """Get list of conditions with sufficient samples."""
        valid = []
        for condition, indices in self._sample_groups.items():
            if len(indices) >= self.config.min_samples_per_condition:
                valid.append(condition)
            else:
                logger.debug(
                    f"Condition '{condition}' has {len(indices)} samples "
                    f"(< {self.config.min_samples_per_condition}), skipping"
                )
        return sorted(valid)

    def filter_genes(
        self,
        genes: Set[str],
        condition: str
    ) -> Set[str]:
        """
        Filter genes for expression and variance.

        Addresses brutalist critique: "60,664 genes includes noise"
        """
        sample_idx = self._sample_groups.get(condition)
        if sample_idx is None:
            raise ValueError(f"Unknown condition: {condition}")

        # Get genes present in matrix
        available_genes = set(self.matrix.feature_ids) & genes
        if not available_genes:
            return set()

        gene_list = sorted(available_genes)
        gene_idx = [self.matrix.feature_ids.get_loc(g) for g in gene_list]

        # Extract expression data for this condition
        expr_data = self.matrix.data[np.ix_(gene_idx, sample_idx)]

        # Filter by expression level
        mean_expr = np.mean(expr_data, axis=1)
        expr_threshold = np.percentile(mean_expr, self.config.min_expression_percentile)
        expr_mask = mean_expr >= expr_threshold

        # Filter by variance
        var_expr = np.var(expr_data, axis=1)
        var_threshold = np.percentile(var_expr[var_expr > 0], self.config.min_variance_percentile)
        var_mask = var_expr >= var_threshold

        # Combined filter
        keep_mask = expr_mask & var_mask
        filtered_genes = {gene_list[i] for i in range(len(gene_list)) if keep_mask[i]}

        n_removed = len(available_genes) - len(filtered_genes)
        if n_removed > 0:
            logger.info(
                f"Filtered {n_removed} low-expression/variance genes "
                f"({len(filtered_genes)} remaining)"
            )

        return filtered_genes

    def compute_correlation_matrix(
        self,
        genes: List[str],
        condition: str,
        method: str = 'pearson'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute correlation matrix with proper handling.

        Returns:
            (correlation_matrix, gene_list) - NaN-safe correlation matrix
        """
        sample_idx = self._sample_groups.get(condition)
        if sample_idx is None:
            raise ValueError(f"Unknown condition: {condition}")

        # Get gene indices (only genes in matrix)
        valid_genes = [g for g in genes if g in self.matrix.feature_ids]
        if len(valid_genes) < 2:
            return np.array([[]]), []

        gene_idx = [self.matrix.feature_ids.get_loc(g) for g in valid_genes]

        # Extract expression matrix (genes × samples)
        expr_data = self.matrix.data[np.ix_(gene_idx, sample_idx)]

        # Compute correlation
        if method == 'pearson':
            corr_matrix = np.corrcoef(expr_data)
        elif method == 'spearman':
            corr_matrix, _ = stats.spearmanr(expr_data.T)
            if corr_matrix.ndim == 0:
                # spearmanr returns scalar for 2 variables — reconstruct 2×2 matrix
                rho = float(corr_matrix)
                corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
        else:
            raise ValueError(f"Unknown method: {method}")

        # Handle NaN (genes with zero variance)
        np.fill_diagonal(corr_matrix, 1.0)
        # DATA-III-1 (Audit III): Log NaN correlations from zero-variance genes.
        nan_mask = np.isnan(corr_matrix)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logger.debug(
                "compute_correlation_matrix: %d NaN correlations from "
                "zero-variance genes — zeroed.", n_nan,
            )
            corr_matrix[nan_mask] = 0.0

        return corr_matrix, valid_genes

    def soft_threshold(
        self,
        corr_matrix: np.ndarray,
        power: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply WGCNA-style soft thresholding.

        Addresses brutalist critique: "Hard threshold is insane"

        Instead of: edge if |r| >= 0.7
        We use: weight = |r|^power

        This preserves the continuous nature of correlation while
        emphasizing strong correlations.
        """
        if power is None:
            power = self.config.soft_threshold_power

        # Soft threshold: weight = |r|^power
        weights = np.abs(corr_matrix) ** power

        # Apply minimum edge weight filter
        weights[weights < self.config.min_edge_weight] = 0.0

        # Zero diagonal
        np.fill_diagonal(weights, 0.0)

        return weights

    def build_signed_graphs(
        self,
        corr_matrix: np.ndarray,
        genes: List[str]
    ) -> Tuple[nx.Graph, nx.Graph]:
        """
        Build separate graphs for positive and negative correlations.

        Addresses brutalist critique: "abs(correlation) loses sign"

        Returns:
            (positive_graph, negative_graph) - soft-thresholded graphs
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX required for graph construction")

        n = len(genes)

        # Soft threshold
        weights = self.soft_threshold(corr_matrix)

        # Positive correlation graph
        G_pos = nx.Graph()
        G_pos.add_nodes_from(genes)

        # Negative correlation graph
        G_neg = nx.Graph()
        G_neg.add_nodes_from(genes)

        # Vectorize: find non-zero upper-triangle entries
        rows, cols = np.where(np.triu(weights, k=1) > 0)
        for r, c in zip(rows, cols):
            w = float(weights[r, c])
            if corr_matrix[r, c] > 0:
                G_pos.add_edge(genes[r], genes[c], weight=w)
            else:
                G_neg.add_edge(genes[r], genes[c], weight=w)

        return G_pos, G_neg

    def detect_communities(
        self,
        G: nx.Graph,
        method: Optional[CommunityMethod] = None
    ) -> Dict[str, int]:
        """
        Detect communities using specified method.

        Returns:
            Dict mapping gene -> community_id
        """
        if len(G.nodes()) == 0:
            return {}

        method = method or self.config.method

        if method == CommunityMethod.LOUVAIN:
            return self._louvain_communities(G)
        elif method == CommunityMethod.LEIDEN:
            return self._leiden_communities(G)
        elif method == CommunityMethod.HIERARCHICAL:
            return self._hierarchical_communities(G)
        elif method == CommunityMethod.GREEDY:
            return self._greedy_communities(G)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _louvain_communities(self, G: nx.Graph) -> Dict[str, int]:
        """Louvain community detection."""
        if len(G.edges()) == 0:
            # No edges - each node is its own community
            return {node: i for i, node in enumerate(G.nodes())}

        if HAS_LOUVAIN:
            return community_louvain.best_partition(
                G,
                weight='weight',
                resolution=self.config.resolution,
                random_state=self._get_community_seed()
            )
        else:
            # Fall back to NetworkX implementation
            communities = nx_community.louvain_communities(
                G,
                weight='weight',
                resolution=self.config.resolution,
                seed=self._get_community_seed()
            )
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
            return partition

    def _leiden_communities(self, G: nx.Graph) -> Dict[str, int]:
        """Leiden community detection (improved Louvain)."""
        if not HAS_LEIDEN:
            raise ImportError("leidenalg and igraph required")

        if len(G.edges()) == 0:
            return {node: i for i, node in enumerate(G.nodes())}

        # Convert NetworkX to igraph
        node_list = list(G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}

        edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
        weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]

        ig_graph = ig.Graph(n=len(node_list), edges=edges)
        ig_graph.es['weight'] = weights

        # Run Leiden
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=self.config.resolution,
            seed=self._get_community_seed()
        )

        return {node_list[i]: partition.membership[i] for i in range(len(node_list))}

    def _hierarchical_communities(self, G: nx.Graph) -> Dict[str, int]:
        """Hierarchical clustering on correlation distance."""
        if len(G.nodes()) < 2:
            return {node: 0 for node in G.nodes()}

        node_list = list(G.nodes())
        n = len(node_list)

        # Build distance matrix (1 - weight)
        dist_matrix = np.ones((n, n))
        node_idx = {n: i for i, n in enumerate(node_list)}

        for u, v, data in G.edges(data=True):
            i, j = node_idx[u], node_idx[v]
            dist = 1.0 - data.get('weight', 0.0)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

        np.fill_diagonal(dist_matrix, 0.0)

        # Hierarchical clustering
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method='average')

        # Cut tree to get clusters
        labels = fcluster(Z, t=0.5, criterion='distance')

        return {node_list[i]: int(labels[i]) for i in range(n)}

    def _greedy_communities(self, G: nx.Graph) -> Dict[str, int]:
        """Greedy modularity optimization."""
        if len(G.edges()) == 0:
            return {node: i for i, node in enumerate(G.nodes())}

        communities = nx_community.greedy_modularity_communities(G, weight='weight')
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition

    def compute_community_stats(
        self,
        community_genes: Set[str],
        corr_matrix: np.ndarray,
        gene_list: List[str],
        G: nx.Graph
    ) -> Dict[str, float]:
        """Compute statistics for a community."""
        if len(community_genes) < 2:
            return {
                'mean_correlation': 1.0,
                'min_correlation': 1.0,
                'max_correlation': 1.0,
                'density': 1.0
            }

        # RC-VIII-2: O(1) dict lookup instead of O(n) list.index()
        gene_to_idx = {g: i for i, g in enumerate(gene_list)}
        indices = [gene_to_idx[g] for g in community_genes if g in gene_to_idx]

        if len(indices) < 2:
            return {
                'mean_correlation': 1.0,
                'min_correlation': 1.0,
                'max_correlation': 1.0,
                'density': 1.0
            }

        # Extract submatrix
        sub_corr = corr_matrix[np.ix_(indices, indices)]

        # Get upper triangle (excluding diagonal)
        upper_idx = np.triu_indices(len(indices), k=1)
        correlations = sub_corr[upper_idx]

        # Compute edge density in graph
        subgraph = G.subgraph(community_genes)
        n_possible = len(community_genes) * (len(community_genes) - 1) // 2
        n_actual = subgraph.number_of_edges()
        density = n_actual / n_possible if n_possible > 0 else 0.0

        return {
            'mean_correlation': float(np.mean(np.abs(correlations))),
            'min_correlation': float(np.min(np.abs(correlations))),
            'max_correlation': float(np.max(np.abs(correlations))),
            'density': density
        }

    def _compute_corr(
        self,
        expr_data: np.ndarray,
        method: str = 'pearson',
    ) -> tuple[np.ndarray, int]:
        """Compute correlation matrix using the specified method.

        RC-VII-5: Shared helper so bootstrap/permutation use the same
        correlation measure as ``compute_correlation_matrix``.

        RC-VIII-1: Also handles NaN (from zero-variance genes) and fills
        diagonal to 1.0 internally, so callers don't have to.

        Returns:
            (corr_matrix, n_nan_off_diag): Cleaned correlation matrix and
            the count of NaN entries that were zeroed (excluding diagonal).
        """
        if method == 'spearman':
            corr, _ = stats.spearmanr(expr_data.T)
            if corr.ndim == 0:
                rho = float(corr)
                corr = np.array([[1.0, rho], [rho, 1.0]])
        else:
            corr = np.corrcoef(expr_data)

        # Fill diagonal first, then count off-diagonal NaN
        np.fill_diagonal(corr, 1.0)
        nan_mask = np.isnan(corr)
        n_nan_off_diag = int(nan_mask.sum())  # diagonal is already 1.0
        if n_nan_off_diag > 0:
            corr[nan_mask] = 0.0
        return corr, n_nan_off_diag

    def bootstrap_stability(
        self,
        genes: Set[str],
        condition: str,
        n_bootstrap: Optional[int] = None,
        correlation_sign: Optional[CorrelationSign] = None,
        method: str = 'pearson',
    ) -> Dict[int, float]:
        """
        Assess community stability via bootstrap resampling.

        Addresses brutalist critique: "Bootstrap your correlations"

        Returns:
            Dict mapping community_id -> stability (fraction of bootstraps)
        """
        n_bootstrap = n_bootstrap or self.config.n_bootstrap
        sample_idx = self._sample_groups[condition]
        n_samples = len(sample_idx)

        # Get gene indices
        gene_list = sorted(g for g in genes if g in self.matrix.feature_ids)
        gene_idx = [self.matrix.feature_ids.get_loc(g) for g in gene_list]

        # Original communities
        corr_original, _ = self.compute_correlation_matrix(gene_list, condition)
        G_pos, G_neg = self.build_signed_graphs(corr_original, gene_list)
        if correlation_sign == CorrelationSign.NEGATIVE:
            original_partition = self.detect_communities(G_neg)
        else:
            original_partition = self.detect_communities(G_pos)

        # Invert to get community -> genes
        original_communities = {}
        for gene, comm_id in original_partition.items():
            if comm_id not in original_communities:
                original_communities[comm_id] = set()
            original_communities[comm_id].add(gene)

        # Bootstrap
        stability_counts = {cid: 0 for cid in original_communities}
        nan_sub_counts: list[int] = []

        for b in range(n_bootstrap):
            # Resample samples with replacement
            boot_sample_idx = self._rng.choice(sample_idx, size=n_samples, replace=True)

            # RC-VII-5 + RC-VIII-1: _compute_corr now handles NaN and diagonal.
            expr_data = self.matrix.data[np.ix_(gene_idx, boot_sample_idx)]
            boot_corr, n_nan = self._compute_corr(expr_data, method=method)
            if n_nan > 0:
                logger.debug(
                    "Bootstrap iter %d: %d NaN correlations from zero-variance genes",
                    b, n_nan,
                )
                nan_sub_counts.append(n_nan)

            # Detect communities
            G_boot_pos, G_boot_neg = self.build_signed_graphs(boot_corr, gene_list)
            if correlation_sign == CorrelationSign.NEGATIVE:
                boot_partition = self.detect_communities(G_boot_neg)
            else:
                boot_partition = self.detect_communities(G_boot_pos)

            # Check which original communities are recovered
            boot_communities = {}
            for gene, comm_id in boot_partition.items():
                if comm_id not in boot_communities:
                    boot_communities[comm_id] = set()
                boot_communities[comm_id].add(gene)

            # RC-VII-1: Match communities with exclusion tracking to prevent
            # multiple original communities from matching the same bootstrap community.
            matched_boot_ids: set = set()
            for orig_cid, orig_genes in original_communities.items():
                best_jaccard = 0.0
                best_boot_id = None
                for boot_id, boot_genes in boot_communities.items():
                    if boot_id in matched_boot_ids:
                        continue
                    intersection = len(orig_genes & boot_genes)
                    union = len(orig_genes | boot_genes)
                    jaccard = intersection / union if union > 0 else 0.0
                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
                        best_boot_id = boot_id

                # Count as "recovered" if Jaccard > 0.5
                if best_jaccard > 0.5 and best_boot_id is not None:
                    stability_counts[orig_cid] += 1
                    matched_boot_ids.add(best_boot_id)

        if nan_sub_counts:
            logger.warning(
                "Bootstrap stability: %d/%d iterations had NaN correlations "
                "(total %d substitutions). This may indicate zero-variance genes "
                "in bootstrap resamples.",
                len(nan_sub_counts), n_bootstrap, sum(nan_sub_counts),
            )

        # Convert to proportions
        stability = {cid: count / n_bootstrap for cid, count in stability_counts.items()}
        return stability

    def permutation_null(
        self,
        genes: Set[str],
        condition: str,
        observed_modularity: float,
        n_permutations: Optional[int] = None,
        correlation_sign: Optional[CorrelationSign] = None,
        method: str = 'pearson',
    ) -> float:
        """
        Compute permutation p-value for observed modularity.

        Args:
            genes: Set of genes to analyze
            condition: Condition to analyze
            observed_modularity: The observed modularity to test against
            n_permutations: Number of permutations (defaults to config)
            correlation_sign: Which graph to test. NEGATIVE uses the negative
                correlation graph; default/POSITIVE uses the positive graph.

        Returns:
            p-value: Probability of observing modularity >= observed under null
        """
        n_permutations = n_permutations or self.config.n_permutations
        sample_idx = self._sample_groups[condition]
        n_samples = len(sample_idx)

        gene_list = sorted(g for g in genes if g in self.matrix.feature_ids)
        gene_idx = [self.matrix.feature_ids.get_loc(g) for g in gene_list]

        null_modularities = []
        perm_nan_counts: list[int] = []

        for p in range(n_permutations):
            # RC-VII-7: Per-gene independent shuffling to destroy inter-gene
            # correlation while preserving each gene's marginal distribution.
            # Column permutation is correlation-invariant (np.corrcoef(X[:, perm])
            # == np.corrcoef(X)), so the old approach was a no-op.
            expr_data = self.matrix.data[np.ix_(gene_idx, sample_idx)].copy()
            for row in range(expr_data.shape[0]):
                self._rng.shuffle(expr_data[row])

            # RC-VII-5 + RC-VIII-1: _compute_corr now handles NaN and diagonal.
            perm_corr, n_nan = self._compute_corr(expr_data, method=method)
            if n_nan > 0:
                logger.debug(
                    "Permutation iter %d: %d NaN correlations from zero-variance genes",
                    p, n_nan,
                )
                perm_nan_counts.append(n_nan)

            # Build graph and detect communities
            G_perm_pos, G_perm_neg = self.build_signed_graphs(perm_corr, gene_list)
            if correlation_sign == CorrelationSign.NEGATIVE:
                G_test = G_perm_neg
            else:
                G_test = G_perm_pos
            partition = self.detect_communities(G_test)

            # Compute modularity
            if G_test.number_of_edges() > 0:
                communities = {}
                for gene, cid in partition.items():
                    if cid not in communities:
                        communities[cid] = set()
                    communities[cid].add(gene)

                mod = nx_community.modularity(G_test, communities.values(), weight='weight')
                null_modularities.append(mod)
            else:
                null_modularities.append(0.0)

        if perm_nan_counts:
            logger.warning(
                "Permutation null: %d/%d iterations had NaN correlations "
                "(total %d substitutions). This may indicate zero-variance genes "
                "in permuted resamples.",
                len(perm_nan_counts), n_permutations, sum(perm_nan_counts),
            )

        # Compute p-value using Phipson-Smyth correction: (b+1)/(B+1)
        # Ensures p-value is never exactly 0.0 and has correct coverage
        null_modularities = np.array(null_modularities)
        b = int(np.sum(null_modularities >= observed_modularity))
        p_value = (b + 1) / (len(null_modularities) + 1)

        return p_value

    def analyze_coherence(
        self,
        genes: Set[str],
        condition: str,
        regulator_name: Optional[str] = None,
        compute_bootstrap: bool = True,
        compute_permutation: bool = True
    ) -> StratifiedCoherenceResult:
        """
        Main analysis method: find coherent communities in gene set.

        Args:
            genes: Set of genes to analyze (e.g., INDRA targets)
            condition: Condition to analyze (e.g., "CASE_Male")
            regulator_name: Upstream regulator name for annotation
            compute_bootstrap: Whether to compute bootstrap stability
            compute_permutation: Whether to compute permutation p-value

        Returns:
            StratifiedCoherenceResult with detected communities
        """
        sample_idx = self._sample_groups.get(condition)
        if sample_idx is None:
            raise ValueError(f"Unknown condition: {condition}")

        n_samples = len(sample_idx)
        if n_samples < self.config.min_samples_per_condition:
            raise ValueError(
                f"Condition {condition} has {n_samples} samples "
                f"(< {self.config.min_samples_per_condition} required)"
            )

        logger.info(f"Analyzing coherence for {len(genes)} genes in {condition} (n={n_samples})")

        # Step 1: Filter genes
        filtered_genes = self.filter_genes(genes, condition)
        if len(filtered_genes) < 3:
            logger.warning(f"Too few genes after filtering: {len(filtered_genes)}")
            return StratifiedCoherenceResult(
                condition=condition,
                n_samples=n_samples,
                positive_communities=[],
                negative_communities=[],
                modularity_positive=0.0,
                modularity_negative=0.0,
                genes_in_positive=set(),
                genes_in_negative=set(),
                genes_unclustered=genes
            )

        # Step 2: Compute correlations
        corr_matrix, gene_list = self.compute_correlation_matrix(list(filtered_genes), condition)

        # Step 3: Build signed graphs
        G_pos, G_neg = self.build_signed_graphs(corr_matrix, gene_list)

        logger.info(
            f"Built graphs: positive={G_pos.number_of_edges()} edges, "
            f"negative={G_neg.number_of_edges()} edges"
        )

        # Step 4: Detect communities
        partition_pos = self.detect_communities(G_pos)
        partition_neg = self.detect_communities(G_neg)

        # Step 5: Compute modularity
        if G_pos.number_of_edges() > 0:
            comm_pos = {}
            for gene, cid in partition_pos.items():
                if cid not in comm_pos:
                    comm_pos[cid] = set()
                comm_pos[cid].add(gene)
            modularity_pos = nx_community.modularity(G_pos, comm_pos.values(), weight='weight')
        else:
            comm_pos = {}
            modularity_pos = 0.0

        if G_neg.number_of_edges() > 0:
            comm_neg = {}
            for gene, cid in partition_neg.items():
                if cid not in comm_neg:
                    comm_neg[cid] = set()
                comm_neg[cid].add(gene)
            modularity_neg = nx_community.modularity(G_neg, comm_neg.values(), weight='weight')
        else:
            comm_neg = {}
            modularity_neg = 0.0

        # Step 6: Bootstrap stability (if requested)
        bootstrap_stability_pos = {}
        bootstrap_stability_neg = {}
        if compute_bootstrap and self.config.n_bootstrap > 0:
            logger.info("Computing bootstrap stability...")
            bootstrap_stability_pos = self.bootstrap_stability(
                filtered_genes, condition, correlation_sign=CorrelationSign.POSITIVE
            )
            # KG-IV-2 (Audit IV): Also compute stability for negative communities
            if comm_neg:
                bootstrap_stability_neg = self.bootstrap_stability(
                    filtered_genes, condition, correlation_sign=CorrelationSign.NEGATIVE
                )

        # Step 6b: Permutation null (if requested)
        perm_pvalue_pos = None
        perm_pvalue_neg = None
        if compute_permutation and self.config.n_permutations > 0:
            logger.info("Computing permutation null for positive graph...")
            perm_pvalue_pos = self.permutation_null(
                filtered_genes, condition, modularity_pos,
                correlation_sign=CorrelationSign.POSITIVE
            )
            if comm_neg:
                logger.info("Computing permutation null for negative graph...")
                perm_pvalue_neg = self.permutation_null(
                    filtered_genes, condition, modularity_neg,
                    correlation_sign=CorrelationSign.NEGATIVE
                )

        # Step 7: Build CommunityResult objects
        positive_communities = []
        for cid, comm_genes in comm_pos.items():
            if len(comm_genes) < self.config.min_community_size:
                continue

            stats = self.compute_community_stats(comm_genes, corr_matrix, gene_list, G_pos)
            if stats['density'] < self.config.min_community_density:
                continue

            positive_communities.append(CommunityResult(
                community_id=cid,
                genes=comm_genes,
                mean_correlation=stats['mean_correlation'],
                min_correlation=stats['min_correlation'],
                max_correlation=stats['max_correlation'],
                density=stats['density'],
                modularity_contribution=_community_modularity(G_pos, comm_genes),
                bootstrap_stability=bootstrap_stability_pos.get(cid),
                permutation_pvalue=perm_pvalue_pos,
                regulator_name=regulator_name,
                condition=condition,
                correlation_sign=CorrelationSign.POSITIVE
            ))

        negative_communities = []
        for cid, comm_genes in comm_neg.items():
            if len(comm_genes) < self.config.min_community_size:
                continue

            stats = self.compute_community_stats(comm_genes, corr_matrix, gene_list, G_neg)
            if stats['density'] < self.config.min_community_density:
                continue

            negative_communities.append(CommunityResult(
                community_id=cid,
                genes=comm_genes,
                mean_correlation=stats['mean_correlation'],
                min_correlation=stats['min_correlation'],
                max_correlation=stats['max_correlation'],
                density=stats['density'],
                modularity_contribution=_community_modularity(G_neg, comm_genes),
                bootstrap_stability=bootstrap_stability_neg.get(cid),
                permutation_pvalue=perm_pvalue_neg,
                regulator_name=regulator_name,
                condition=condition,
                correlation_sign=CorrelationSign.NEGATIVE
            ))

        # Compute gene coverage
        genes_in_pos = set().union(*[c.genes for c in positive_communities]) if positive_communities else set()
        genes_in_neg = set().union(*[c.genes for c in negative_communities]) if negative_communities else set()
        genes_unclustered = filtered_genes - genes_in_pos - genes_in_neg

        logger.info(
            f"Found {len(positive_communities)} positive communities, "
            f"{len(negative_communities)} negative communities"
        )

        return StratifiedCoherenceResult(
            condition=condition,
            n_samples=n_samples,
            positive_communities=sorted(positive_communities, key=lambda c: -c.size),
            negative_communities=sorted(negative_communities, key=lambda c: -c.size),
            modularity_positive=modularity_pos,
            modularity_negative=modularity_neg,
            genes_in_positive=genes_in_pos,
            genes_in_negative=genes_in_neg,
            genes_unclustered=genes_unclustered
        )

    def analyze_all_conditions(
        self,
        genes: Set[str],
        regulator_name: Optional[str] = None,
        compute_bootstrap: bool = True,
        compute_permutation: bool = False
    ) -> Dict[str, StratifiedCoherenceResult]:
        """
        Analyze coherence across all valid conditions.

        Returns:
            Dict mapping condition -> StratifiedCoherenceResult
        """
        results = {}
        conditions = self.get_available_conditions()

        logger.info(f"Analyzing {len(conditions)} conditions: {conditions}")

        for condition in conditions:
            try:
                result = self.analyze_coherence(
                    genes=genes,
                    condition=condition,
                    regulator_name=regulator_name,
                    compute_bootstrap=compute_bootstrap,
                    compute_permutation=compute_permutation
                )
                results[condition] = result
            except (ValueError, RuntimeError) as e:
                logger.error(f"Error analyzing {condition}: {e}")
                continue

        return results

    def compare_conditions(
        self,
        result_a: StratifiedCoherenceResult,
        result_b: StratifiedCoherenceResult,
        genes: Set[str],
        fdr_threshold: float = 0.05,
        min_corr_change: float = 0.3
    ) -> DifferentialCoherenceResult:
        """
        Compare coherence between two conditions using Fisher Z-transformation.

        This method identifies gene pairs with significantly different correlations
        between conditions, enabling discovery of disease-specific regulatory rewiring.

        Steps:
        1. Compute correlation matrices for both conditions
        2. Apply Fisher z-test to all gene pairs
        3. FDR correction (Benjamini-Hochberg)
        4. Filter for significant changes (FDR < threshold AND |Δr| >= min_corr_change)
        5. Match communities between conditions

        Args:
            result_a: Coherence results for condition A
            result_b: Coherence results for condition B
            genes: Set of genes to analyze
            fdr_threshold: FDR threshold for significance (default: 0.05)
            min_corr_change: Minimum |Δr| for biological relevance (default: 0.3)

        Returns:
            DifferentialCoherenceResult with statistical comparison
        """
        logger.info(
            f"Comparing conditions: {result_a.condition} (n={result_a.n_samples}) "
            f"vs {result_b.condition} (n={result_b.n_samples})"
        )

        # Step 1: Get correlation matrices for both conditions
        # Find common genes that passed filtering in both conditions
        genes_a = result_a.genes_in_positive | result_a.genes_in_negative | result_a.genes_unclustered
        genes_b = result_b.genes_in_positive | result_b.genes_in_negative | result_b.genes_unclustered
        common_genes = genes_a & genes_b & genes

        if len(common_genes) < 3:
            logger.warning(f"Too few common genes: {len(common_genes)}")
            return DifferentialCoherenceResult(
                condition_a=result_a.condition,
                condition_b=result_b.condition,
                n_samples_a=result_a.n_samples,
                n_samples_b=result_b.n_samples,
                communities_only_a=[],
                communities_only_b=[],
                communities_shared=[],
                significant_changes=[],
                all_changes=[],
                jaccard_similarity=0.0,
                n_significant_increases=0,
                n_significant_decreases=0
            )

        gene_list = sorted(common_genes)
        n_genes = len(gene_list)
        logger.info(f"Analyzing {n_genes} common genes ({n_genes*(n_genes-1)//2} pairs)")

        # Compute correlation matrices
        corr_a, _ = self.compute_correlation_matrix(gene_list, result_a.condition)
        corr_b, _ = self.compute_correlation_matrix(gene_list, result_b.condition)

        # Step 2: Apply Fisher z-test to all gene pairs
        all_changes = []
        z_scores = []
        p_values = []

        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                r_a = corr_a[i, j]
                r_b = corr_b[i, j]

                # Fisher z-test
                z_diff, p_val = fisher_z_test(
                    r_a, r_b,
                    result_a.n_samples,
                    result_b.n_samples
                )

                change = GeneCorrelationChange(
                    gene_i=gene_list[i],
                    gene_j=gene_list[j],
                    corr_a=r_a,
                    corr_b=r_b,
                    z_diff=z_diff,
                    p_value=p_val
                )
                all_changes.append(change)
                p_values.append(p_val)

        # Step 3: FDR correction (Benjamini-Hochberg)
        if len(p_values) > 0:
            reject, pvals_corrected, _, _ = multipletests(
                p_values,
                alpha=fdr_threshold,
                method='fdr_bh'
            )

            # Attach FDR-corrected p-values
            for change, fdr_p in zip(all_changes, pvals_corrected):
                change.fdr_adjusted_p = float(fdr_p)

        # Step 4: Identify significant changes
        # Require BOTH statistical significance (FDR < threshold) AND effect size (|Δr| >= min_corr_change)
        significant_changes = [
            c for c in all_changes
            if c.fdr_adjusted_p is not None
            and c.fdr_adjusted_p < fdr_threshold
            and abs(c.corr_delta) >= min_corr_change
        ]

        # Count increases vs decreases
        n_increases = sum(1 for c in significant_changes if c.corr_delta > 0)
        n_decreases = sum(1 for c in significant_changes if c.corr_delta < 0)

        logger.info(
            f"Found {len(significant_changes)} significant correlation changes: "
            f"{n_increases} increases, {n_decreases} decreases"
        )

        # Step 5: Match communities between conditions
        communities_only_a, communities_only_b, communities_shared = self._match_communities(
            result_a,
            result_b,
            jaccard_threshold=0.3
        )

        # Compute Jaccard similarity
        genes_a_clustered = result_a.genes_in_positive | result_a.genes_in_negative
        genes_b_clustered = result_b.genes_in_positive | result_b.genes_in_negative
        intersection = len(genes_a_clustered & genes_b_clustered)
        union = len(genes_a_clustered | genes_b_clustered)
        jaccard = intersection / union if union > 0 else 0.0

        return DifferentialCoherenceResult(
            condition_a=result_a.condition,
            condition_b=result_b.condition,
            n_samples_a=result_a.n_samples,
            n_samples_b=result_b.n_samples,
            communities_only_a=communities_only_a,
            communities_only_b=communities_only_b,
            communities_shared=communities_shared,
            significant_changes=significant_changes,
            all_changes=all_changes,
            jaccard_similarity=jaccard,
            n_significant_increases=n_increases,
            n_significant_decreases=n_decreases
        )

    def _match_communities(
        self,
        result_a: StratifiedCoherenceResult,
        result_b: StratifiedCoherenceResult,
        jaccard_threshold: float = 0.3
    ) -> Tuple[List[CommunityResult], List[CommunityResult], List[Tuple[CommunityResult, CommunityResult]]]:
        """
        Match communities between two conditions using Jaccard similarity.

        Args:
            result_a: Coherence results for condition A
            result_b: Coherence results for condition B
            jaccard_threshold: Minimum Jaccard similarity to consider communities matched

        Returns:
            (gained, lost, matched_pairs):
            - gained: Communities in B not in A (disease-gained communities)
            - lost: Communities in A not in B (disease-lost communities)
            - matched: Pairs of similar communities (stable across conditions)
        """
        communities_only_a = []
        communities_only_b = []
        communities_shared = []

        # RC-VIII-5: Match positive-to-positive and negative-to-negative
        # separately. Cross-sign matching conflates co-activation with
        # anti-correlation, which are biologically distinct phenomena.
        for comms_a, comms_b in [
            (result_a.positive_communities, result_b.positive_communities),
            (result_a.negative_communities, result_b.negative_communities),
        ]:
            matched_b = set()
            for comm_a in comms_a:
                best_match = None
                best_jaccard = 0.0
                for i, comm_b in enumerate(comms_b):
                    if i in matched_b:
                        continue
                    inter = len(comm_a.genes & comm_b.genes)
                    uni = len(comm_a.genes | comm_b.genes)
                    j = inter / uni if uni > 0 else 0.0
                    if j > best_jaccard and j >= jaccard_threshold:
                        best_jaccard = j
                        best_match = (i, comm_b)

                if best_match:
                    matched_b.add(best_match[0])
                    communities_shared.append((comm_a, best_match[1]))
                else:
                    communities_only_a.append(comm_a)

            for i, comm_b in enumerate(comms_b):
                if i not in matched_b:
                    communities_only_b.append(comm_b)

        return communities_only_a, communities_only_b, communities_shared


def print_coherence_report(
    results: Dict[str, StratifiedCoherenceResult],
    regulator_name: str = "Regulator"
) -> str:
    """Generate a human-readable report of coherence analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"REGULATORY COHERENCE ANALYSIS: {regulator_name}")
    lines.append("=" * 70)

    for condition, result in results.items():
        lines.append(f"\n## Condition: {condition} (n={result.n_samples})")
        lines.append("-" * 50)

        lines.append(f"Positive correlation communities: {len(result.positive_communities)}")
        lines.append(f"  Modularity: {result.modularity_positive:.3f}")
        lines.append(f"  Genes covered: {len(result.genes_in_positive)}")

        if result.positive_communities:
            largest = result.largest_positive_community
            lines.append(f"  Largest: {largest.size} genes, density={largest.density:.2f}")
            if largest.bootstrap_stability is not None:
                lines.append(f"           bootstrap stability={largest.bootstrap_stability:.2f}")

        lines.append(f"\nNegative correlation communities: {len(result.negative_communities)}")
        lines.append(f"  Modularity: {result.modularity_negative:.3f}")
        lines.append(f"  Genes covered: {len(result.genes_in_negative)}")

        if result.negative_communities:
            largest = result.largest_negative_community
            lines.append(f"  Largest: {largest.size} genes, density={largest.density:.2f}")

        lines.append(f"\nUnclustered genes: {len(result.genes_unclustered)}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
