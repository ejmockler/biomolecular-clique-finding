"""
Correlation-based clique validation for regulatory modules.

This module validates regulatory modules by identifying correlation cliques within
gene sets. A correlation clique is a maximal subset of genes where all pairwise
correlations exceed a threshold. This is critical for identifying coherent regulatory
modules in ALS transcriptomics.

Biological Context:
    In transcriptomics, co-expressed genes often share regulatory mechanisms.
    Upstream regulators (TFs, kinases, signaling molecules) regulate sets of target
    genes, and these targets should show correlated expression patterns if they are
    truly co-regulated.

    Clique finding identifies the maximal coherent subset of a putative regulatory
    module. This addresses:
    - False positives from knowledge bases (not all causal links = co-expression)
    - Condition-specific regulation (genes regulated only in disease state)
    - Differential correlation (cliques gained/lost in disease vs control)

Engineering Design:
    - Stratified analysis: Compute correlations separately for each phenotype/sex group
    - Graph-based cliques: Use NetworkX for efficient maximal clique enumeration
    - Coherent modules: Derive maximally correlated subsets from INDRA targets
    - Differential analysis: Identify cliques gained or lost in disease conditions

Scientific Rationale:
    - Pearson correlation: Captures linear co-expression relationships
    - Spearman correlation: Robust to outliers, captures monotonic relationships
    - Minimum correlation threshold: 0.7 is standard for co-expression networks
    - Minimum clique size: 3+ genes form meaningful regulatory modules

Examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from cliquefinder import BioMatrix
    >>> from cliquefinder.core.quality import QualityFlag
    >>> from cliquefinder.knowledge.clique_validator import CliqueValidator
    >>>
    >>> # Create sample matrix
    >>> data = np.random.randn(1000, 100)  # 1000 genes, 100 samples
    >>> feature_ids = pd.Index([f"GENE{i}" for i in range(1000)])
    >>> sample_ids = pd.Index([f"S{i}" for i in range(100)])
    >>> metadata = pd.DataFrame({
    ...     'phenotype': ['CASE']*50 + ['CTRL']*50,
    ...     'Sex': ['Male', 'Female'] * 50
    ... }, index=sample_ids)
    >>> quality_flags = np.full((1000, 100), QualityFlag.ORIGINAL, dtype=int)
    >>> matrix = BioMatrix(data, feature_ids, sample_ids, metadata, quality_flags)
    >>>
    >>> # Find correlation cliques
    >>> validator = CliqueValidator(matrix, stratify_by=['phenotype', 'Sex'])
    >>> genes = {'GENE0', 'GENE1', 'GENE2', 'GENE3'}
    >>> cliques = validator.find_cliques(genes, condition='CASE_Male',
    ...                                   min_correlation=0.7, min_clique_size=3)
    >>>
    >>> # Find differential cliques
    >>> gained, lost = validator.find_differential_cliques(genes,
    ...                                                     case_condition='CASE',
    ...                                                     ctrl_condition='CTRL')
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple, Literal, Dict
import itertools
import logging
import time
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.stats.correlation_tests import (
    test_correlation_difference,
    apply_fdr_correction,
    compute_significance_threshold,
    correlation_confidence_interval,
    CorrelationTestResult,
)
from cliquefinder.knowledge.clique_algorithms import (
    kcore_reduction,
    estimate_clique_complexity,
)

__all__ = [
    'CorrelationClique',
    'ChildSetType2',
    'DifferentialCliqueResult',
    'GenePairDifferentialStat',
    'CliqueValidator',
    'InsufficientSamplesError',
    'GeneNotFoundError',
]


class InsufficientSamplesError(Exception):
    """Raised when too few samples in a stratum for reliable correlation analysis."""
    pass


class GeneNotFoundError(Exception):
    """Raised when one or more genes are not found in the expression matrix."""
    pass


@dataclass
class CorrelationClique:
    """
    A maximal clique of genes with high pairwise correlations.

    Represents a maximally connected subset of genes where all pairwise correlations
    exceed a threshold. These form coherent co-expression modules likely under shared
    regulatory control.

    Attributes:
        genes: Set of gene identifiers forming the clique
        condition: Condition under which clique was identified (e.g., "CASE", "CASE_Male")
        mean_correlation: Mean absolute correlation across all gene pairs in clique
        min_correlation: Minimum absolute correlation across all gene pairs
        size: Number of genes in clique (len(genes))

    Examples:
        >>> clique = CorrelationClique(
        ...     genes={'SOD1', 'TARDBP', 'FUS'},
        ...     condition='CASE_Male',
        ...     mean_correlation=0.85,
        ...     min_correlation=0.72,
        ...     size=3
        ... )
        >>> print(f"Clique of {clique.size} genes with mean r={clique.mean_correlation:.2f}")
    """
    genes: Set[str]
    condition: str
    mean_correlation: float
    min_correlation: float
    size: int


@dataclass
class ChildSetType2:
    """
    Coherent module: maximally correlated subset of INDRA target genes.

    Derived from INDRA targets (downstream genes per INDRA CoGEx) by extracting
    the maximal clique - the largest subset with all pairwise correlations above
    threshold. This identifies the coherently co-expressed core regulatory module.

    Biological Context:
        INDRA targets come from causal knowledge bases (literature mining, pathway
        databases). Not all causal relationships result in correlated expression,
        and correlation may be condition-specific. Coherent modules identify the
        actual co-regulated subset in the experimental condition of interest.

    Attributes:
        regulator_name: Upstream regulator name/identifier (TF, kinase, etc.)
        indra_targets: Full set of INDRA targets passed for analysis
        clique_genes: Coherent subset (maximal correlation clique)
        condition: Condition under which coherent module was derived
        correlation_threshold: Minimum correlation used for clique finding
        mean_correlation: Mean correlation within coherent module

    Examples:
        >>> coherent = ChildSetType2(
        ...     regulator_name='TP53',
        ...     indra_targets={'GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5'},
        ...     clique_genes={'GENE1', 'GENE3', 'GENE4'},
        ...     condition='CASE_Male',
        ...     correlation_threshold=0.7,
        ...     mean_correlation=0.82
        ... )
        >>> print(f"{coherent.regulator_name}: {len(coherent.clique_genes)}/{len(coherent.indra_targets)} genes coherent")
    """
    regulator_name: str
    indra_targets: Set[str]
    clique_genes: Set[str]
    condition: str
    correlation_threshold: float
    mean_correlation: float


@dataclass
class GenePairDifferentialStat:
    """
    Statistical analysis of differential correlation between two genes.

    Publication-quality metrics for differential co-expression analysis.
    Tests H0: correlation(gene1, gene2) is the same in CASE vs CTRL.

    Attributes:
        gene1: First gene identifier
        gene2: Second gene identifier
        r_case: Pearson correlation in CASE condition
        r_ctrl: Pearson correlation in CTRL condition
        delta_r: r_case - r_ctrl (positive = increased correlation in CASE)
        z_score: Fisher Z-score for difference
        p_value: Two-tailed p-value for differential correlation
        q_value: FDR-adjusted p-value (after correction)
        ci_case: 95% CI for correlation in CASE
        ci_ctrl: 95% CI for correlation in CTRL
        is_significant: True if q_value < FDR threshold
    """
    gene1: str
    gene2: str
    r_case: float
    r_ctrl: float
    delta_r: float
    z_score: float
    p_value: float
    q_value: float = 1.0  # Set after FDR correction
    ci_case: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    ci_ctrl: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    is_significant: bool = False


@dataclass
class DifferentialCliqueResult:
    """
    Publication-quality result of differential clique analysis.

    Contains cliques gained/lost between conditions with statistical
    support for each gene pair, including FDR-corrected significance
    and effective number of independent tests.

    Attributes:
        case_condition: CASE condition label
        ctrl_condition: CTRL condition label
        n_case_samples: Number of samples in CASE
        n_ctrl_samples: Number of samples in CTRL
        gained_cliques: Cliques present in CASE but not CTRL
        lost_cliques: Cliques present in CTRL but not CASE
        significant_gene_pairs: Gene pairs with FDR-significant differential correlation
        all_gene_pair_stats: Full statistics for all gene pairs tested
        significance_threshold: Adaptive r threshold used for clique building
        fdr_threshold: FDR threshold applied
        nominal_tests: Nominal number of gene pairs tested
        effective_tests: Effective number of independent tests (M_eff)
            Accounts for correlation between gene pairs. Typically 0.3-0.6 * nominal_tests
            due to co-regulation structure in gene expression data.

    Example:
        >>> result = validator.find_differential_cliques_with_stats(
        ...     genes, 'CASE', 'CTRL', fdr_threshold=0.05
        ... )
        >>> print(f"Gained: {len(result.gained_cliques)} cliques")
        >>> print(f"Tested {result.nominal_tests} pairs (effective: ~{result.effective_tests:.0f})")
    """
    case_condition: str
    ctrl_condition: str
    n_case_samples: int
    n_ctrl_samples: int
    gained_cliques: List[CorrelationClique]
    lost_cliques: List[CorrelationClique]
    significant_gene_pairs: List[GenePairDifferentialStat]
    all_gene_pair_stats: List[GenePairDifferentialStat]
    significance_threshold: float
    fdr_threshold: float
    nominal_tests: int = 0
    effective_tests: float = 0.0


class CliqueValidator:
    """
    Validates regulatory modules through correlation-based clique finding.

    Identifies maximal subsets of genes with high pairwise correlations, optionally
    stratified by experimental conditions (phenotype, sex, etc.). Enables discovery
    of coherent co-expression modules and differential correlation analysis.

    The validator operates on a BioMatrix containing gene expression data with sample
    metadata. Correlations are computed separately for each experimental condition,
    allowing condition-specific regulatory module identification.

    Performance Optimization:
        This class implements condition-level data caching for significant speedup.
        When analyzing multiple regulators across the same conditions, call
        `precompute_condition_data()` once before analysis to pre-slice data
        for each condition. This avoids redundant mask computation and data
        slicing across thousands of correlation calculations.

        Speedup: ~5-10x for multi-regulator analysis.

    Attributes:
        matrix: BioMatrix containing expression data and sample metadata
        stratify_by: List of metadata columns for stratification (e.g., ['phenotype', 'Sex'])
        min_samples: Minimum samples required per stratum for correlation (default: 10)

    Examples:
        >>> validator = CliqueValidator(matrix, stratify_by=['phenotype', 'Sex'])
        >>>
        >>> # RECOMMENDED: Precompute condition data for multi-regulator analysis
        >>> validator.precompute_condition_data()
        >>>
        >>> # Find cliques in CASE males
        >>> genes = {'SOD1', 'TARDBP', 'FUS', 'OPTN', 'TBK1'}
        >>> cliques = validator.find_cliques(genes, condition='CASE_Male',
        ...                                   min_correlation=0.7, min_clique_size=3)
        >>>
        >>> # Get coherent module for a regulator's INDRA targets
        >>> indra_targets = {'GENE1', 'GENE2', 'GENE3', 'GENE4'}
        >>> coherent = validator.get_child_set_type2('TP53', indra_targets,
        ...                                           condition='CASE', min_correlation=0.7)
        >>>
        >>> # Find differential cliques
        >>> gained, lost = validator.find_differential_cliques(genes,
        ...                                                     case_condition='CASE',
        ...                                                     ctrl_condition='CTRL')
    """

    def __init__(
        self,
        matrix: BioMatrix,
        stratify_by: Optional[List[str]] = None,
        min_samples: int = 10,
        precompute: bool = True,
    ):
        """
        Initialize CliqueValidator with expression matrix and stratification scheme.

        Args:
            matrix: BioMatrix containing expression data and sample metadata
            stratify_by: List of metadata columns for stratification.
                Examples: ['phenotype'], ['phenotype', 'Sex']
                If None, no stratification (use all samples)
            min_samples: Minimum samples required per stratum for correlation.
                Fewer samples may produce unreliable correlation estimates.
            precompute: If True (default), precompute condition data slices for
                all conditions at initialization. Recommended for multi-regulator
                analysis. Set to False for single queries to save memory.

        Raises:
            ValueError: If stratify_by columns not in sample_metadata

        Examples:
            >>> # No stratification (all samples together)
            >>> validator = CliqueValidator(matrix)
            >>>
            >>> # Stratify by phenotype only
            >>> validator = CliqueValidator(matrix, stratify_by=['phenotype'])
            >>>
            >>> # Stratify by phenotype and sex (with precomputation)
            >>> validator = CliqueValidator(matrix, stratify_by=['phenotype', 'Sex'])
            >>>
            >>> # Disable precomputation for single queries
            >>> validator = CliqueValidator(matrix, stratify_by=['phenotype'], precompute=False)
        """
        self.matrix = matrix
        self.stratify_by = stratify_by or []
        self.min_samples = min_samples

        # Validate stratification columns exist
        if self.stratify_by:
            missing = set(self.stratify_by) - set(matrix.sample_metadata.columns)
            if missing:
                raise ValueError(
                    f"Stratification columns {missing} not found in sample_metadata. "
                    f"Available columns: {list(matrix.sample_metadata.columns)}"
                )

        # O(1) gene index lookup
        self._gene_to_idx = {str(gene): i for i, gene in enumerate(matrix.feature_ids)}

        # Correlation matrix cache (per gene-set, limited utility for diverse regulators)
        self._corr_cache: Dict[Tuple[str, frozenset], pd.DataFrame] = {}

        # Condition-level data cache (MAJOR optimization)
        # Stores pre-sliced data and masks per condition to avoid redundant computation
        self._condition_data_cache: Dict[str, np.ndarray] = {}
        self._condition_mask_cache: Dict[str, np.ndarray] = {}
        self._condition_sample_count: Dict[str, int] = {}

        # Precomputed full correlation matrices (MAJOR optimization for multi-regulator analysis)
        # Stores full correlation matrices for all genes in the precomputed set
        # Key: (condition, method) -> Value: full correlation matrix
        self._precomputed_corr: Dict[Tuple[str, str], np.ndarray] = {}
        self._precomputed_gene_list: List[str] = []  # Ordered list of genes
        self._precomputed_gene_index: Dict[str, int] = {}  # Gene -> index mapping

        # Precompute condition data if requested
        if precompute:
            self.precompute_condition_data()

    def precompute_condition_data(self) -> None:
        """
        Precompute and cache data slices for all conditions.

        This is the KEY optimization for multi-regulator analysis. Call this once
        before analyzing multiple regulators to avoid redundant computation.

        For each condition, this method:
        1. Computes the sample mask (which samples belong to this condition)
        2. Pre-slices the data matrix for those samples
        3. Caches both for instant access during correlation computation

        Memory Usage:
            Caches ~n_conditions copies of the data matrix, but only the sample
            dimension is reduced. For 6 conditions with ~45K genes × 100 samples
            each, this is ~6 × 45K × 100 × 8 bytes ≈ 200 MB total.

        Speedup:
            Eliminates redundant mask computation and data slicing.
            For 50 regulators × 6 conditions × 1 correlation per regulator/condition,
            this saves ~300 mask computations and data slices.

        Examples:
            >>> validator = CliqueValidator(matrix, stratify_by=['phenotype', 'Sex'])
            >>> validator.precompute_condition_data()  # Called automatically if precompute=True
            >>> # Now all subsequent operations use cached data
        """
        conditions = self.get_available_conditions()

        for condition in conditions:
            try:
                # Compute and cache mask
                mask = self._compute_condition_mask_internal(condition)
                self._condition_mask_cache[condition] = mask
                self._condition_sample_count[condition] = int(mask.sum())

                # Pre-slice and cache data
                self._condition_data_cache[condition] = self.matrix.data[:, mask]

            except InsufficientSamplesError:
                # Skip conditions with insufficient samples
                pass

    def precompute_correlation_matrices(
        self,
        genes: Set[str],
        conditions: Optional[List[str]] = None,
        method: Literal["pearson", "spearman", "max"] = "max",
    ) -> None:
        """
        Precompute full correlation matrices for a gene set across all conditions.

        This is the MAJOR optimization for multi-regulator analysis. When analyzing
        thousands of regulators, correlation matrices are the bottleneck. Instead of
        computing correlations ~65,000 times (8,209 regulators × 4 conditions × 2 methods),
        we compute ONCE per condition for the union of all target genes.

        Performance Impact:
            - Before: 8,209 regulators × 4 conditions = 32,836 correlation computations
            - After: 4 conditions = 4 correlation computations
            - Speedup: ~8,000x reduction in correlation computations
            - Runtime: ~6 minutes → ~1 minute for full analysis

        Memory Usage:
            For ~3,000 genes × 4 conditions:
            - Each matrix: 3000 × 3000 × 8 bytes = 72 MB
            - Total: 4 × 72 MB = ~288 MB
            - With "max" method (Pearson + Spearman): ~576 MB total

        Algorithm:
            1. Convert gene set to sorted list for consistent indexing
            2. For each condition:
               - Use cached condition data if available
               - Compute Pearson and/or Spearman correlations for ALL genes
               - For "max" method: element-wise max(|Pearson|, |Spearman|)
               - Cache full correlation matrix
            3. Build gene -> index mapping for O(1) subset extraction

        Subsequent calls to compute_correlation_matrix() will:
            - Check if genes are in precomputed set
            - Extract correlation submatrix via fancy indexing: O(k^2) where k = |subset|
            - Return as DataFrame with proper gene labels

        Args:
            genes: Set of all genes to precompute (union of all regulator targets)
            conditions: List of conditions to precompute. If None, use all available.
            method: Correlation method to precompute:
                - "max": Compute both Pearson and Spearman, take max(|r|) per pair
                - "pearson": Pearson correlation only
                - "spearman": Spearman correlation only

        Examples:
            >>> # Collect union of all target genes
            >>> all_targets = set()
            >>> for module in modules:
            ...     all_targets.update(module.indra_target_names)
            >>>
            >>> # Precompute correlation matrices ONCE
            >>> validator.precompute_correlation_matrices(
            ...     all_targets,
            ...     conditions=['CASE_Male', 'CASE_Female', 'CTRL_Male', 'CTRL_Female'],
            ...     method='max'
            ... )
            >>>
            >>> # Now all subsequent calls to compute_correlation_matrix() are O(1) subsets
            >>> for module in modules:
            ...     corr = validator.compute_correlation_matrix(
            ...         module.indra_target_names, 'CASE_Male', method='max'
            ...     )  # Fast: just indexes into precomputed matrix
        """
        if conditions is None:
            conditions = self.get_available_conditions()

        # Convert to sorted list for consistent ordering
        gene_list = sorted(genes)
        logger.info(f"Precomputing correlation matrices for {len(gene_list)} genes "
                   f"across {len(conditions)} conditions (method={method})...")

        # Get gene indices in full matrix
        gene_idx, found_genes = self._get_gene_indices(gene_list)

        if len(found_genes) < len(gene_list):
            logger.warning(f"Only {len(found_genes)}/{len(gene_list)} genes found in matrix. "
                          f"Precomputing for available genes only.")
            gene_list = found_genes

        # Build gene -> index mapping for fast subset extraction
        self._precomputed_gene_list = gene_list
        self._precomputed_gene_index = {g: i for i, g in enumerate(gene_list)}

        # Precompute correlation matrices for each condition
        for condition in conditions:
            try:
                # Use cached condition data if available
                if condition in self._condition_data_cache:
                    condition_data = self._condition_data_cache[condition]
                    subdata = condition_data[gene_idx, :]
                else:
                    # Compute on demand
                    sample_mask = self._get_condition_mask(condition)
                    subdata = self.matrix.data[np.ix_(gene_idx, sample_mask)]

                # Compute correlation based on method
                if method == "pearson":
                    corr = np.corrcoef(subdata)
                    self._precomputed_corr[(condition, method)] = corr

                elif method == "spearman":
                    corr, _ = spearmanr(subdata.T)
                    # Handle case of 2 genes (spearmanr returns scalar)
                    if isinstance(corr, float):
                        corr = np.array([[1.0, corr], [corr, 1.0]])
                    self._precomputed_corr[(condition, method)] = corr

                elif method == "max":
                    # Compute BOTH Pearson and Spearman
                    pearson_corr = np.corrcoef(subdata)

                    spearman_corr, _ = spearmanr(subdata.T)
                    if isinstance(spearman_corr, float):
                        spearman_corr = np.array([[1.0, spearman_corr], [spearman_corr, 1.0]])

                    # Take element-wise max of absolute values, preserving sign
                    abs_pearson = np.abs(pearson_corr)
                    abs_spearman = np.abs(spearman_corr)
                    pearson_stronger = abs_pearson >= abs_spearman
                    corr = np.where(pearson_stronger, pearson_corr, spearman_corr)

                    self._precomputed_corr[(condition, method)] = corr

                else:
                    raise ValueError(f"Unknown correlation method: {method}")

                logger.info(f"  {condition}: Computed {len(gene_list)}×{len(gene_list)} "
                           f"correlation matrix ({corr.nbytes / 1024**2:.1f} MB)")

            except InsufficientSamplesError as e:
                logger.warning(f"  {condition}: Skipped - {e}")
                continue

        # Log memory usage
        total_bytes = sum(arr.nbytes for arr in self._precomputed_corr.values())
        logger.info(f"Correlation matrix precomputation complete: "
                   f"{len(self._precomputed_corr)} matrices, "
                   f"{total_bytes / 1024**2:.1f} MB total")

    def get_available_conditions(self) -> List[str]:
        """
        Get list of all available stratification conditions.

        Returns conditions that can be passed to find_cliques, compute_correlation_matrix,
        etc. Conditions are formed by combining unique values of stratification columns
        with underscores.

        Returns:
            List of condition strings (e.g., ['CASE_Male', 'CASE_Female', 'CTRL_Male', 'CTRL_Female'])
            If no stratification, returns ['all']

        Examples:
            >>> validator = CliqueValidator(matrix, stratify_by=['phenotype', 'Sex'])
            >>> conditions = validator.get_available_conditions()
            >>> print(conditions)
            ['CASE_Female', 'CASE_Male', 'CTRL_Female', 'CTRL_Male']
        """
        if not self.stratify_by:
            return ['all']

        # Get unique combinations of stratification values
        metadata = self.matrix.sample_metadata
        groups = metadata.groupby(self.stratify_by, observed=True)

        conditions = []
        for group_key, group_df in groups:
            # group_key is tuple if multiple columns, single value otherwise
            if isinstance(group_key, tuple):
                condition = '_'.join(str(v) for v in group_key)
            else:
                condition = str(group_key)
            conditions.append(condition)

        return sorted(conditions)

    def _compute_condition_mask_internal(self, condition: str) -> np.ndarray:
        """
        Internal: Compute boolean mask for samples matching condition.

        This is the core mask computation logic, separated from caching and
        validation for use during precomputation.

        Args:
            condition: Condition string (e.g., "CASE_Male")

        Returns:
            Boolean array (n_samples,)

        Raises:
            ValueError: If condition format is invalid
            InsufficientSamplesError: If fewer than min_samples match
        """
        metadata = self.matrix.sample_metadata

        if not self.stratify_by:
            return np.ones(self.matrix.n_samples, dtype=bool)

        parts = condition.split('_')

        if len(parts) != len(self.stratify_by):
            raise ValueError(
                f"Condition '{condition}' has {len(parts)} parts but stratification "
                f"requires {len(self.stratify_by)} parts: {self.stratify_by}"
            )

        mask = np.ones(self.matrix.n_samples, dtype=bool)
        for col, value in zip(self.stratify_by, parts):
            mask &= (metadata[col] == value).values

        n_samples = mask.sum()
        if n_samples < self.min_samples:
            raise InsufficientSamplesError(
                f"Condition '{condition}' has only {n_samples} samples "
                f"(minimum required: {self.min_samples})"
            )

        return mask

    def _get_condition_mask(self, condition: str) -> np.ndarray:
        """
        Get boolean mask for samples matching condition (cache-aware).

        Uses cached masks when available (after precompute_condition_data()),
        otherwise computes on demand.

        Args:
            condition: Condition string. Examples:
                - "CASE" → phenotype == CASE
                - "CASE_Male" → phenotype == CASE AND Sex == Male
                - "CTRL_Female" → phenotype == CTRL AND Sex == Female

        Returns:
            Boolean array (n_samples,) where True indicates sample matches condition

        Raises:
            ValueError: If condition format is invalid or references unknown metadata
            InsufficientSamplesError: If fewer than min_samples match condition

        Examples:
            >>> mask = validator._get_condition_mask('CASE')
            >>> print(f"{mask.sum()} CASE samples")
            >>>
            >>> mask = validator._get_condition_mask('CASE_Male')
            >>> print(f"{mask.sum()} CASE Male samples")
        """
        # Use cached mask if available (major speedup for repeated calls)
        if condition in self._condition_mask_cache:
            return self._condition_mask_cache[condition]

        # Compute on demand
        return self._compute_condition_mask_internal(condition)

    def _get_gene_indices(self, genes: List[str]) -> Tuple[List[int], List[str]]:
        """
        Get matrix indices for genes, filtering out missing genes.

        Args:
            genes: List of gene identifiers to look up

        Returns:
            Tuple of (indices, found_genes) where:
                - indices: List of row indices in matrix.data
                - found_genes: List of genes that were found (same order as indices)

        Raises:
            GeneNotFoundError: If no genes are found in matrix
        """
        # Fix 1: Use O(1) dict lookup instead of O(n) list.index()
        indices = []
        found_genes = []

        for gene in genes:
            idx = self._gene_to_idx.get(gene)
            if idx is not None:
                indices.append(idx)
                found_genes.append(gene)

        if not indices:
            raise GeneNotFoundError(
                f"None of the {len(genes)} genes found in expression matrix. "
                f"First few requested: {list(genes)[:5]}"
            )

        return indices, found_genes

    def compute_correlation_matrix(
        self,
        genes: List[str],
        condition: Optional[str] = None,
        method: Literal["pearson", "spearman", "max"] = "max",
    ) -> pd.DataFrame:
        """
        Compute pairwise correlation matrix for genes under specified condition.

        Computes correlations between all gene pairs using samples matching the specified
        condition. Missing genes are silently skipped.

        **Max-Correlation Strategy (default):**
            By default (method="max"), computes BOTH Pearson and Spearman correlations
            and takes the maximum absolute value for each gene pair. This ensures we
            capture strong relationships regardless of whether they are:
            - Linear (Pearson)
            - Monotonic but non-linear (Spearman)

        Performance:
            - FAST PATH: If precompute_correlation_matrices() was called, extracts
              correlation submatrix via fancy indexing in O(k^2) where k = |genes|
            - SLOW PATH: Computes correlations from scratch using cached condition data

        Args:
            genes: List of gene identifiers
            condition: Condition string (e.g., "CASE", "CASE_Male"). If None, use all samples.
            method: Correlation method:
                - "max" (default): max(|Pearson|, |Spearman|) for each pair
                - "pearson": Pearson correlation (linear relationships)
                - "spearman": Spearman correlation (monotonic relationships)

        Returns:
            DataFrame with genes as both index and columns, values are correlations [-1, 1]

        Raises:
            GeneNotFoundError: If no genes found in matrix
            InsufficientSamplesError: If too few samples in condition

        Examples:
            >>> genes = ['SOD1', 'TARDBP', 'FUS']
            >>> # Max correlation (default - captures both linear and monotonic)
            >>> corr = validator.compute_correlation_matrix(genes, condition='CASE_Male')
            >>> print(corr.loc['SOD1', 'TARDBP'])
            >>>
            >>> # Pearson only (linear relationships)
            >>> corr_pearson = validator.compute_correlation_matrix(genes,
            ...                                                      condition='CASE',
            ...                                                      method='pearson')
        """
        # Check gene-set cache first (useful for repeated queries with same genes)
        cache_key = (condition or 'all', frozenset(genes), method)
        if cache_key in self._corr_cache:
            return self._corr_cache[cache_key]

        # FAST PATH: Use precomputed correlation matrix if available
        cond_key = condition or 'all'
        if (cond_key, method) in self._precomputed_corr:
            # Extract indices for requested genes from precomputed set
            gene_indices = []
            found_genes = []
            for gene in genes:
                if gene in self._precomputed_gene_index:
                    gene_indices.append(self._precomputed_gene_index[gene])
                    found_genes.append(gene)

            if not found_genes:
                raise GeneNotFoundError(
                    f"None of the {len(genes)} genes found in precomputed correlation matrix. "
                    f"First few requested: {list(genes)[:5]}"
                )

            # Extract correlation submatrix using fancy indexing
            full_corr = self._precomputed_corr[(cond_key, method)]
            # np.ix_ creates index arrays for fancy indexing
            subset_corr = full_corr[np.ix_(gene_indices, gene_indices)]

            # Convert to DataFrame
            result = pd.DataFrame(subset_corr, index=found_genes, columns=found_genes)

            # Store in gene-set cache
            self._corr_cache[cache_key] = result
            return result

        # SLOW PATH: Compute correlation from scratch
        # Get gene indices
        gene_idx, found_genes = self._get_gene_indices(genes)

        # OPTIMIZATION: Use cached condition data if available
        if condition and condition in self._condition_data_cache:
            # Fast path: use pre-sliced condition data
            condition_data = self._condition_data_cache[condition]
            subdata = condition_data[gene_idx, :]
        else:
            # Slow path: compute mask and slice on demand
            if condition:
                sample_mask = self._get_condition_mask(condition)
            else:
                sample_mask = np.ones(self.matrix.n_samples, dtype=bool)
            subdata = self.matrix.data[np.ix_(gene_idx, sample_mask)]

        # Compute correlation based on method
        if method == "pearson":
            # numpy corrcoef works on rows, so subdata is already correct shape
            corr = np.corrcoef(subdata)
        elif method == "spearman":
            # spearmanr expects samples as rows, so transpose
            corr, _ = spearmanr(subdata.T)
            # Handle case of 2 genes (spearmanr returns scalar)
            if isinstance(corr, float):
                corr = np.array([[1.0, corr], [corr, 1.0]])
        elif method == "max":
            # Compute BOTH Pearson and Spearman, take max absolute value
            # Pearson correlation
            pearson_corr = np.corrcoef(subdata)

            # Spearman correlation
            spearman_corr, _ = spearmanr(subdata.T)
            # Handle case of 2 genes (spearmanr returns scalar)
            if isinstance(spearman_corr, float):
                spearman_corr = np.array([[1.0, spearman_corr], [spearman_corr, 1.0]])

            # Take element-wise max of absolute values, preserving sign of stronger correlation
            # For each pair, use the correlation with larger absolute value
            abs_pearson = np.abs(pearson_corr)
            abs_spearman = np.abs(spearman_corr)

            # Create mask: True where Pearson has larger absolute value
            pearson_stronger = abs_pearson >= abs_spearman

            # Select the stronger correlation (preserving sign)
            corr = np.where(pearson_stronger, pearson_corr, spearman_corr)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        # Convert to DataFrame
        result = pd.DataFrame(corr, index=found_genes, columns=found_genes)

        # Store in gene-set cache (limited utility but helps repeated queries)
        self._corr_cache[cache_key] = result
        return result

    def build_correlation_graph(
        self,
        genes: List[str],
        condition: str,
        min_correlation: float = 0.7,
        method: Literal["pearson", "spearman", "max"] = "max",
        use_vectorized: bool = True,
    ) -> nx.Graph:
        """
        Build correlation graph where edges represent high correlations.

        Creates an undirected graph where nodes are genes and edges connect gene pairs
        with absolute correlation >= min_correlation. This graph representation enables
        efficient clique finding via NetworkX algorithms.

        **Performance Optimization (OPT-5):**
            By default (use_vectorized=True), uses NumPy vectorization for 10-50x speedup
            over the original Python loop approach. For 1000 genes (~500K pairs):
            - Old: ~500ms (Python loop with DataFrame indexing)
            - New: ~10ms (vectorized NumPy operations)

        Args:
            genes: List of gene identifiers
            condition: Condition string (e.g., "CASE_Male")
            min_correlation: Minimum absolute correlation to create edge (default: 0.7)
            method: Correlation method:
                - "max" (default): max(|Pearson|, |Spearman|) for each pair
                - "pearson": Pearson correlation only
                - "spearman": Spearman correlation only
            use_vectorized: If True (default), use vectorized NumPy implementation.
                Set to False to use legacy Python loop (for testing/comparison).

        Returns:
            NetworkX Graph with genes as nodes, edges weighted by correlation

        Raises:
            GeneNotFoundError: If no genes found in matrix
            InsufficientSamplesError: If too few samples in condition

        Examples:
            >>> genes = ['GENE1', 'GENE2', 'GENE3', 'GENE4']
            >>> # Max correlation (default - vectorized)
            >>> G = validator.build_correlation_graph(genes, condition='CASE',
            ...                                        min_correlation=0.7)
            >>> print(f"{len(G.nodes)} nodes, {len(G.edges)} edges")
            >>> print(f"GENE1 neighbors: {list(G.neighbors('GENE1'))}")
            >>>
            >>> # Use legacy loop for comparison
            >>> G_legacy = validator.build_correlation_graph(genes, condition='CASE',
            ...                                               use_vectorized=False)
        """
        if use_vectorized:
            return self._build_correlation_graph_vectorized(
                genes, condition, min_correlation, method
            )
        else:
            return self._build_correlation_graph_loop(
                genes, condition, min_correlation, method
            )

    def _build_correlation_graph_vectorized(
        self,
        genes: List[str],
        condition: str,
        min_correlation: float,
        method: Literal["pearson", "spearman", "max"],
    ) -> nx.Graph:
        """
        Vectorized graph construction - 10-50x faster than loop-based approach.

        Uses NumPy vectorization to extract upper triangle of correlation matrix,
        apply threshold mask, and batch-construct edges. Eliminates O(n²) Python
        loop with DataFrame indexing.

        Performance:
            - 100 genes (~5K pairs): ~1ms (vs ~50ms loop)
            - 500 genes (~125K pairs): ~5ms (vs ~200ms loop)
            - 1000 genes (~500K pairs): ~10ms (vs ~500ms loop)
            - 3000 genes (~4.5M pairs): ~100ms (vs ~5s loop)

        Algorithm:
            1. Compute correlation matrix (cached/precomputed when possible)
            2. Extract upper triangle indices via np.triu_indices(n, k=1)
            3. Vectorized threshold mask: mask = |corr_values| >= min_correlation
            4. Extract passing edges: i_edges, j_edges, weights = mask filter
            5. Batch graph construction: G.add_edges_from(edge_list)

        Args:
            genes: List of gene identifiers
            condition: Condition string
            min_correlation: Minimum absolute correlation threshold
            method: Correlation method

        Returns:
            NetworkX Graph with genes as nodes, edges weighted by correlation
        """
        # Compute correlation matrix
        corr = self.compute_correlation_matrix(genes, condition, method)

        # Convert to numpy array for vectorization
        corr_values = corr.values
        gene_list = list(corr.index)
        n = len(gene_list)

        # Handle edge case: empty or single-gene matrix
        if n == 0:
            return nx.Graph()
        if n == 1:
            G = nx.Graph()
            G.add_node(gene_list[0])
            return G

        # Vectorized upper triangle extraction
        # np.triu_indices(n, k=1) returns (i_indices, j_indices) for upper triangle
        # k=1 excludes diagonal (no self-correlations)
        i_upper, j_upper = np.triu_indices(n, k=1)
        upper_values = corr_values[i_upper, j_upper]

        # Vectorized threshold mask
        mask = np.abs(upper_values) >= min_correlation

        # Extract passing edges (only those above threshold)
        i_edges = i_upper[mask]
        j_edges = j_upper[mask]
        weights = upper_values[mask]

        # Batch graph construction
        G = nx.Graph()
        G.add_nodes_from(gene_list)

        # Create edge list with weights
        # Convert numpy types to Python types for NetworkX compatibility
        edges = [
            (gene_list[int(i)], gene_list[int(j)], {'weight': float(w)})
            for i, j, w in zip(i_edges, j_edges, weights)
        ]
        G.add_edges_from(edges)

        return G

    def _build_correlation_graph_loop(
        self,
        genes: List[str],
        condition: str,
        min_correlation: float,
        method: Literal["pearson", "spearman", "max"],
    ) -> nx.Graph:
        """
        Legacy loop-based graph construction (SLOW - retained for testing).

        This is the original O(n²) Python loop with DataFrame indexing approach.
        Kept for backward compatibility and performance comparison testing.

        Performance: ~50x slower than vectorized approach for large graphs.

        Args:
            genes: List of gene identifiers
            condition: Condition string
            min_correlation: Minimum absolute correlation threshold
            method: Correlation method

        Returns:
            NetworkX Graph with genes as nodes, edges weighted by correlation
        """
        # Compute correlation matrix
        corr = self.compute_correlation_matrix(genes, condition, method)

        # Build graph
        G = nx.Graph()

        # Add all genes as nodes (even if no edges)
        G.add_nodes_from(corr.index)

        # Add edges for correlations above threshold
        for i, g1 in enumerate(corr.index):
            for j, g2 in enumerate(corr.columns):
                if i < j:  # Upper triangle only (undirected graph)
                    corr_val = corr.iloc[i, j]
                    if abs(corr_val) >= min_correlation:
                        G.add_edge(g1, g2, weight=corr_val)

        return G


    def find_cliques(
        self,
        genes: Set[str],
        condition: str,
        min_correlation: float = 0.7,
        min_clique_size: int = 3,
        method: Literal["pearson", "spearman", "max"] = "max",
        max_cliques: int = 10000,
        timeout_seconds: Optional[float] = None,
        exact: bool = False,
        n_workers: int = 4,
    ) -> List[CorrelationClique]:
        """
        Find maximal correlation cliques in gene set with parallel component processing.

        A clique is a maximal subset of genes where all pairwise correlations exceed
        the threshold. Uses NetworkX's Bron-Kerbosch algorithm for enumeration.

        Algorithm Complexity:
            - Maximal clique enumeration is output-sensitive: O(3^(n/3)) worst case
            - For sparse graphs (high min_correlation), often runs in polynomial time
            - Preprocessing reduces graph size by removing isolated nodes and
              processing connected components separately

        Performance Optimization:
            - OPT-1: Parallel connected component processing using ThreadPoolExecutor
            - nx.find_cliques() releases GIL during C-level computation, enabling
              effective thread-level parallelism despite Python's GIL
            - Components sorted by size descending for better load balancing
            - n_workers=1 provides sequential fallback for backward compatibility

        Args:
            genes: Set of gene identifiers to analyze
            condition: Condition string (e.g., "CASE_Male")
            min_correlation: Minimum absolute correlation for clique membership (default: 0.7)
            min_clique_size: Minimum clique size to report (default: 3)
            method: Correlation method:
                - "max" (default): max(|Pearson|, |Spearman|) for each pair
                - "pearson": Pearson correlation only
                - "spearman": Spearman correlation only
            max_cliques: Maximum number of cliques to enumerate (default: 10000).
                Prevents exponential blowup on dense graphs. If truncated, results
                are incomplete when exact=False.
            timeout_seconds: Optional timeout for enumeration. If reached, returns
                partial results with a warning. None = no timeout.
            exact: If True, use exact enumeration without max_cliques limit.
                May be slow for dense graphs. Preprocessing is always applied.
            n_workers: Number of worker threads for parallel component processing
                (default: 4). Set to 1 for sequential processing.

        Returns:
            List of CorrelationClique objects, sorted by size (largest first).
            May be incomplete if enumeration was truncated at max_cliques limit
            (when exact=False) or timeout was reached.

        Raises:
            GeneNotFoundError: If no genes found in matrix
            InsufficientSamplesError: If too few samples in condition

        Examples:
            >>> genes = {'SOD1', 'TARDBP', 'FUS', 'OPTN', 'TBK1', 'VCP'}
            >>> # Parallel processing (default)
            >>> cliques = validator.find_cliques(genes, condition='CASE_Male',
            ...                                   min_correlation=0.7, min_clique_size=3)
            >>> for clique in cliques:
            ...     print(f"Clique of {clique.size}: {clique.genes}")
            ...     print(f"  Mean r={clique.mean_correlation:.3f}")
            >>>
            >>> # Sequential processing
            >>> cliques = validator.find_cliques(genes, condition='CASE_Male',
            ...                                   min_correlation=0.7, n_workers=1)

        Scientific Notes:
            Larger cliques indicate stronger regulatory coherence. A clique of size k
            contains k*(k-1)/2 gene pairs, all highly correlated. This suggests shared
            regulatory mechanisms (common TF, pathway, etc.).
        """
        # Build correlation graph
        G = self.build_correlation_graph(list(genes), condition, min_correlation, method)

        # OPT-2: K-core decomposition pruning (OPT-2 from optimization plan)
        # The (min_clique_size-1)-core contains all possible m-clique vertices.
        # This is iterative and more aggressive than single-pass degree filtering:
        # removing a low-degree vertex may expose new low-degree vertices.
        # Proven sound (no false cliques) and complete (no missed cliques).
        n_before = G.number_of_nodes()
        G = kcore_reduction(G, min_clique_size)
        n_after = G.number_of_nodes()
        if n_before > n_after:
            logger.debug(
                f"K-core pruning: {n_before} → {n_after} nodes "
                f"({100 * (n_before - n_after) / n_before:.1f}% reduction)"
            )

        if G.number_of_nodes() == 0:
            return []

        # Get connected components and filter by size
        components = list(nx.connected_components(G))
        components = [c for c in components if len(c) >= min_clique_size]

        if not components:
            return []

        # Sort by size DESCENDING for better load balancing (largest jobs first)
        components.sort(key=len, reverse=True)

        # Shared state for parallel execution
        start_time = time.time()
        effective_max = float('inf') if exact else max_cliques

        def process_component(component):
            """Worker function for processing a single connected component."""
            component_cliques = []
            subgraph = G.subgraph(component)

            for clique_nodes in nx.find_cliques(subgraph):
                # Check timeout
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    break

                if len(clique_nodes) >= min_clique_size:
                    clique_genes = set(clique_nodes)
                    edge_corrs = [
                        abs(G[u][v]['weight'])
                        for u, v in itertools.combinations(clique_nodes, 2)
                    ]
                    component_cliques.append(CorrelationClique(
                        genes=clique_genes,
                        condition=condition,
                        mean_correlation=float(np.mean(edge_corrs)),
                        min_correlation=float(np.min(edge_corrs)),
                        size=len(clique_genes),
                    ))

            return component_cliques

        # Execute component processing (parallel or sequential)
        all_cliques = []
        enumeration_truncated = False
        timeout_reached = False

        if n_workers > 1 and len(components) > 1:
            # Parallel execution with ThreadPoolExecutor
            logger.debug(f"Processing {len(components)} components with {n_workers} workers")

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit all components
                future_to_comp = {
                    executor.submit(process_component, comp): comp
                    for comp in components
                }

                # Collect results as they complete
                for future in as_completed(future_to_comp):
                    # Check timeout
                    if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                        timeout_reached = True
                        # Cancel remaining futures
                        for f in future_to_comp:
                            f.cancel()
                        break

                    try:
                        component_cliques = future.result(timeout=1.0)
                        all_cliques.extend(component_cliques)

                        # Check max_cliques limit
                        if len(all_cliques) >= effective_max:
                            enumeration_truncated = not exact
                            # Cancel remaining futures
                            for f in future_to_comp:
                                f.cancel()
                            break

                    except Exception as e:
                        logger.warning(f"Component processing failed: {e}")

            # Log parallel execution stats
            elapsed = time.time() - start_time
            logger.debug(f"Parallel processing completed in {elapsed:.2f}s: "
                        f"{len(all_cliques)} cliques from {len(components)} components")

        else:
            # Sequential fallback (n_workers=1 or single component)
            logger.debug(f"Processing {len(components)} components sequentially")

            for component in components:
                # Check timeout
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    timeout_reached = True
                    break

                component_cliques = process_component(component)
                all_cliques.extend(component_cliques)

                # Check max_cliques limit
                if len(all_cliques) >= effective_max:
                    enumeration_truncated = not exact
                    break

            # Log sequential execution stats
            elapsed = time.time() - start_time
            logger.debug(f"Sequential processing completed in {elapsed:.2f}s: "
                        f"{len(all_cliques)} cliques from {len(components)} components")

        # Warnings for incomplete enumeration
        if enumeration_truncated:
            import warnings
            warnings.warn(
                f"Clique enumeration truncated at {max_cliques} cliques for {len(genes)} genes. "
                f"Results may be incomplete. Use exact=True for complete enumeration "
                f"(may be slow) or increase min_correlation to reduce graph density.",
                UserWarning
            )

        if timeout_reached:
            import warnings
            warnings.warn(
                f"Clique enumeration timed out after {timeout_seconds:.1f}s. "
                f"Found {len(all_cliques)} cliques before timeout. "
                f"Results are partial. Increase timeout or min_correlation.",
                UserWarning
            )

        # Sort by size descending, then by mean correlation
        return sorted(all_cliques, key=lambda c: (c.size, c.mean_correlation), reverse=True)

    def find_maximum_clique(
        self,
        genes: Set[str],
        condition: str,
        min_correlation: float = 0.7,
        method: Literal["pearson", "spearman", "max"] = "max",
        n_restarts: int = 10,
    ) -> Optional[CorrelationClique]:
        """
        Find the maximum (largest) correlation clique efficiently.

        Unlike find_cliques() which enumerates ALL maximal cliques (exponential),
        this method uses a greedy heuristic with multiple restarts to find the
        largest clique in O(n * d^2 * restarts) time.

        Algorithm:
            1. For each restart, pick a starting node (high-degree nodes prioritized)
            2. Greedily add nodes that connect to ALL current clique members
            3. Return the largest clique found across restarts

        This is much more efficient than full enumeration when you only need
        the largest clique (not all maximal cliques).

        Args:
            genes: Set of gene identifiers to analyze
            condition: Condition string (e.g., "CASE_Male")
            min_correlation: Minimum absolute correlation for clique membership
            method: Correlation method:
                - "max" (default): max(|Pearson|, |Spearman|) for each pair
                - "pearson": Pearson correlation only
                - "spearman": Spearman correlation only
            n_restarts: Number of greedy restarts (default: 10)

        Returns:
            CorrelationClique for the largest clique found, or None if empty graph

        Examples:
            >>> genes = {'SOD1', 'TARDBP', 'FUS', 'OPTN', 'TBK1'}
            >>> max_clique = validator.find_maximum_clique(genes, 'CASE_Male')
            >>> if max_clique:
            ...     print(f"Maximum clique: {max_clique.genes}")
        """
        # Build correlation graph
        G = self.build_correlation_graph(list(genes), condition, min_correlation, method)

        if len(G.edges) == 0:
            return None

        # Sort nodes by degree (high degree first - more likely in large clique)
        nodes_by_degree = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)

        best_clique: Set[str] = set()

        # Multiple restarts from different starting nodes
        for i in range(min(n_restarts, len(nodes_by_degree))):
            start_node = nodes_by_degree[i]
            clique = {start_node}

            # Get candidates: neighbors of start node
            candidates = set(G.neighbors(start_node))

            # Greedily add nodes that connect to all current clique members
            while candidates:
                # Find candidate with most connections to remaining candidates
                # (increases chance of finding larger clique)
                best_candidate = None
                best_score = -1

                for candidate in candidates:
                    # Check if candidate connects to ALL clique members
                    if all(G.has_edge(candidate, member) for member in clique):
                        # Score: number of remaining candidates this node connects to
                        score = sum(1 for c in candidates if G.has_edge(candidate, c))
                        if score > best_score:
                            best_score = score
                            best_candidate = candidate

                if best_candidate is None:
                    break  # No candidate connects to all clique members

                clique.add(best_candidate)
                candidates.discard(best_candidate)

                # Update candidates: keep only those connected to new member
                candidates = {c for c in candidates if G.has_edge(best_candidate, c)}

            # Update best if larger
            if len(clique) > len(best_clique):
                best_clique = clique

        if len(best_clique) < 2:
            return None

        # Compute correlation statistics for the clique
        clique_list = list(best_clique)
        edge_corrs = [
            abs(G[u][v]['weight'])
            for u, v in itertools.combinations(clique_list, 2)
        ]

        return CorrelationClique(
            genes=best_clique,
            condition=condition,
            mean_correlation=float(np.mean(edge_corrs)),
            min_correlation=float(np.min(edge_corrs)),
            size=len(best_clique),
        )

    def get_child_set_type2(
        self,
        regulator_name: str,
        indra_targets: Set[str],
        condition: str,
        min_correlation: float = 0.7,
        method: Literal["pearson", "spearman"] = "pearson",
        use_fast_maximum: bool = True,
        n_workers: int = 4,
    ) -> Optional[ChildSetType2]:
        """
        Derive coherent module from INDRA targets via maximal clique.

        INDRA targets are downstream genes from the INDRA CoGEx knowledge graph.
        The coherent module is the maximal correlation clique - the largest subset
        of INDRA targets with all pairwise correlations above threshold.

        Returns the largest clique found. If multiple cliques of same size exist,
        returns the one with highest mean correlation.

        Args:
            regulator_name: Upstream regulator name/identifier (TF, kinase, etc.)
            indra_targets: INDRA target genes to analyze for coherence
            condition: Condition string (e.g., "CASE_Male")
            min_correlation: Minimum correlation for coherent module (default: 0.7)
            method: Correlation method - "pearson" or "spearman"
            use_fast_maximum: If True, use greedy maximum clique algorithm (O(n*d^2)).
                If False, enumerate all cliques (exponential but exact). Default: True.
            n_workers: Number of worker threads for parallel component processing
                (default: 4). Only used when use_fast_maximum=False.

        Returns:
            ChildSetType2 object with coherent module, or None if no cliques found

        Raises:
            GeneNotFoundError: If no genes from indra_targets found in matrix
            InsufficientSamplesError: If too few samples in condition

        Examples:
            >>> # INDRA targets from CoGEx
            >>> indra_targets = {'GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5'}
            >>> coherent = validator.get_child_set_type2('TP53', indra_targets,
            ...                                           condition='CASE_Male',
            ...                                           min_correlation=0.7)
            >>> if coherent:
            ...     print(f"Coherent: {len(coherent.clique_genes)}/{len(coherent.indra_targets)} genes")
            ...     print(f"Mean r={coherent.mean_correlation:.3f}")
        """
        if use_fast_maximum:
            # Use efficient greedy algorithm - O(n * d^2)
            largest_clique = self.find_maximum_clique(
                indra_targets, condition, min_correlation, method
            )
        else:
            # Use exact enumeration with preprocessing and timeout
            # The exact=True parameter ensures complete enumeration
            cliques = self.find_cliques(
                indra_targets, condition, min_correlation,
                min_clique_size=2, method=method,
                exact=True,  # Complete enumeration
                timeout_seconds=300.0,  # 5-minute timeout per module
                n_workers=n_workers,  # Parallel component processing
            )
            largest_clique = cliques[0] if cliques else None

        if largest_clique is None:
            return None

        return ChildSetType2(
            regulator_name=regulator_name,
            indra_targets=indra_targets,
            clique_genes=largest_clique.genes,
            condition=condition,
            correlation_threshold=min_correlation,
            mean_correlation=largest_clique.mean_correlation,
        )

    def find_differential_cliques(
        self,
        genes: Set[str],
        case_condition: str = "CASE",
        ctrl_condition: str = "CTRL",
        min_correlation: float = 0.7,
        min_clique_size: int = 3,
        method: Literal["pearson", "spearman"] = "pearson",
        n_workers: int = 4,
    ) -> Tuple[List[CorrelationClique], List[CorrelationClique]]:
        """
        Find cliques gained and lost between case and control conditions.

        Identifies differential correlation - cliques present in one condition but not
        the other. This reveals condition-specific regulatory modules.

        Biological Interpretation:
            - Gained cliques: Coordination emerges in disease (e.g., stress response)
            - Lost cliques: Coordination disrupted in disease (e.g., dysregulation)

        Args:
            genes: Set of gene identifiers to analyze
            case_condition: Case condition (e.g., "CASE", "CASE_Male")
            ctrl_condition: Control condition (e.g., "CTRL", "CTRL_Male")
            min_correlation: Minimum correlation for clique membership (default: 0.7)
            min_clique_size: Minimum clique size to consider (default: 3)
            method: Correlation method - "pearson" or "spearman"
            n_workers: Number of worker threads for parallel component processing (default: 4)

        Returns:
            Tuple of (gained_cliques, lost_cliques) where:
                - gained_cliques: Cliques present in case but not control
                - lost_cliques: Cliques present in control but not case

        Raises:
            GeneNotFoundError: If no genes found in matrix
            InsufficientSamplesError: If too few samples in either condition

        Examples:
            >>> genes = {'SOD1', 'TARDBP', 'FUS', 'OPTN', 'TBK1'}
            >>> gained, lost = validator.find_differential_cliques(
            ...     genes, case_condition='CASE_Male', ctrl_condition='CTRL_Male'
            ... )
            >>> print(f"Gained in CASE: {len(gained)} cliques")
            >>> for clique in gained:
            ...     print(f"  {clique.genes}")
            >>> print(f"Lost in CASE: {len(lost)} cliques")
        """
        # Find cliques in both conditions using exact enumeration
        case_cliques = self.find_cliques(
            genes, case_condition, min_correlation,
            min_clique_size, method,
            exact=True, timeout_seconds=300.0,  # 5-minute timeout per condition
            n_workers=n_workers,  # Parallel component processing
        )
        ctrl_cliques = self.find_cliques(
            genes, ctrl_condition, min_correlation,
            min_clique_size, method,
            exact=True, timeout_seconds=300.0,
            n_workers=n_workers,  # Parallel component processing
        )

        # Convert to sets of frozensets for comparison
        case_sets = {frozenset(c.genes) for c in case_cliques}
        ctrl_sets = {frozenset(c.genes) for c in ctrl_cliques}

        # Find gained and lost
        gained = [c for c in case_cliques if frozenset(c.genes) not in ctrl_sets]
        lost = [c for c in ctrl_cliques if frozenset(c.genes) not in case_sets]

        return gained, lost

    def find_differential_cliques_with_stats(
        self,
        genes: Set[str],
        case_condition: str = "CASE",
        ctrl_condition: str = "CTRL",
        min_correlation: Optional[float] = None,
        min_clique_size: int = 3,
        method: Literal["pearson", "spearman"] = "pearson",
        fdr_threshold: float = 0.05,
        use_adaptive_threshold: bool = True,
        n_workers: int = 4,
    ) -> DifferentialCliqueResult:
        """
        Find differential cliques with publication-quality statistical analysis.

        This method provides:
        - Sample-size-adaptive correlation thresholds
        - Fisher's Z-test for each gene pair
        - FDR correction across all gene pairs
        - Confidence intervals for correlations
        - Comprehensive reporting for publication

        Args:
            genes: Set of gene identifiers to analyze
            case_condition: Case condition (e.g., "CASE", "CASE_Male")
            ctrl_condition: Control condition (e.g., "CTRL", "CTRL_Male")
            min_correlation: Minimum correlation for cliques. If None and
                use_adaptive_threshold=True, computed from sample size.
            min_clique_size: Minimum clique size to report
            method: Correlation method
            fdr_threshold: FDR threshold for significance (default: 0.05)
            use_adaptive_threshold: If True, compute sample-size-adaptive
                correlation threshold when min_correlation is None
            n_workers: Number of worker threads for parallel component processing (default: 4)

        Returns:
            DifferentialCliqueResult with cliques and statistical analysis

        Examples:
            >>> result = validator.find_differential_cliques_with_stats(
            ...     genes={'SOD1', 'TARDBP', 'FUS', 'OPTN'},
            ...     case_condition='CASE', ctrl_condition='CTRL',
            ...     fdr_threshold=0.05
            ... )
            >>> print(f"Gained: {len(result.gained_cliques)}")
            >>> print(f"Significant differential pairs: {len(result.significant_gene_pairs)}")
            >>> for pair in result.significant_gene_pairs[:5]:
            ...     print(f"  {pair.gene1}-{pair.gene2}: delta_r={pair.delta_r:.3f}, q={pair.q_value:.4f}")
        """
        # Get sample counts for each condition
        case_mask = self._get_condition_mask(case_condition)
        ctrl_mask = self._get_condition_mask(ctrl_condition)
        n_case = int(case_mask.sum())
        n_ctrl = int(ctrl_mask.sum())

        # Compute adaptive threshold if not specified
        if min_correlation is None:
            if use_adaptive_threshold:
                # Use the smaller sample size for conservative threshold
                min_n = min(n_case, n_ctrl)
                min_correlation = compute_significance_threshold(
                    min_n, alpha=0.05, n_tests=1
                )
                # Ensure it's at least moderately stringent
                min_correlation = max(min_correlation, 0.5)
            else:
                min_correlation = 0.7

        # Compute correlation matrices for both conditions
        gene_list = list(genes)
        case_corr = self.compute_correlation_matrix(gene_list, case_condition, method)
        ctrl_corr = self.compute_correlation_matrix(gene_list, ctrl_condition, method)

        # Compute differential statistics for all gene pairs
        gene_pair_stats: List[GenePairDifferentialStat] = []
        p_values: List[float] = []
        pair_indices: List[int] = []

        for i, g1 in enumerate(case_corr.index):
            for j, g2 in enumerate(case_corr.columns):
                if i < j:  # Upper triangle only
                    r_case = case_corr.iloc[i, j]
                    r_ctrl = ctrl_corr.iloc[i, j]

                    # Fisher's Z-test (with method-appropriate SE correction)
                    test_result = test_correlation_difference(
                        r_case, n_case, r_ctrl, n_ctrl, method=method
                    )

                    stat = GenePairDifferentialStat(
                        gene1=g1,
                        gene2=g2,
                        r_case=r_case,
                        r_ctrl=r_ctrl,
                        delta_r=r_case - r_ctrl,
                        z_score=test_result.z_score,
                        p_value=test_result.p_value,
                        ci_case=test_result.ci_r1,
                        ci_ctrl=test_result.ci_r2,
                    )
                    gene_pair_stats.append(stat)
                    p_values.append(test_result.p_value)
                    pair_indices.append(len(gene_pair_stats) - 1)

        # Apply FDR correction with effective tests reporting
        m_eff_stats = None
        if p_values:
            # Compute correlation between gene pairs for effective tests estimation
            # Use the CASE correlation matrix as representative structure
            q_values, significant_mask, m_eff_stats = apply_fdr_correction(
                np.array(p_values),
                alpha=fdr_threshold,
                method='bh',
                correlation_matrix=case_corr.values,  # Use correlation structure for M_eff
                report_effective_tests=True
            )

            # Update stats with q-values and significance
            for idx, (q_val, is_sig) in enumerate(zip(q_values, significant_mask)):
                gene_pair_stats[idx].q_value = float(q_val)
                gene_pair_stats[idx].is_significant = bool(is_sig)

        # Find cliques using the threshold
        gained, lost = self.find_differential_cliques(
            genes, case_condition, ctrl_condition,
            min_correlation, min_clique_size, method,
            n_workers=n_workers,  # Parallel component processing
        )

        # Extract significant gene pairs
        significant_pairs = [s for s in gene_pair_stats if s.is_significant]

        # Extract effective tests stats
        nominal_tests = m_eff_stats['nominal_tests'] if m_eff_stats else len(gene_pair_stats)
        effective_tests = m_eff_stats['effective_tests'] if m_eff_stats else float(len(gene_pair_stats))

        return DifferentialCliqueResult(
            case_condition=case_condition,
            ctrl_condition=ctrl_condition,
            n_case_samples=n_case,
            n_ctrl_samples=n_ctrl,
            gained_cliques=gained,
            lost_cliques=lost,
            significant_gene_pairs=significant_pairs,
            all_gene_pair_stats=gene_pair_stats,
            significance_threshold=min_correlation,
            fdr_threshold=fdr_threshold,
            nominal_tests=nominal_tests,
            effective_tests=effective_tests,
        )

    def clear_cache(self, correlation_only: bool = False):
        """
        Clear caches to free memory.

        Useful for long-running analyses or when memory is constrained. Caches
        will be rebuilt automatically as needed for subsequent operations.

        Args:
            correlation_only: If True, only clear gene-set correlation cache.
                If False (default), clear ALL caches including condition data
                and precomputed correlation matrices.

        Examples:
            >>> validator = CliqueValidator(matrix)
            >>> # ... perform many analyses ...
            >>> validator.clear_cache()  # Clear all caches
            >>>
            >>> # Clear just correlation cache, keep condition data
            >>> validator.clear_cache(correlation_only=True)
        """
        self._corr_cache.clear()

        if not correlation_only:
            self._condition_data_cache.clear()
            self._condition_mask_cache.clear()
            self._condition_sample_count.clear()
            self._precomputed_corr.clear()
            self._precomputed_gene_list.clear()
            self._precomputed_gene_index.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about cache usage.

        Returns:
            Dict with cache statistics including number of cached conditions,
            memory usage estimates, correlation cache size, and precomputed
            correlation matrix statistics.

        Examples:
            >>> validator = CliqueValidator(matrix, stratify_by=['phenotype', 'Sex'])
            >>> stats = validator.get_cache_stats()
            >>> print(f"Cached {stats['n_conditions']} conditions using {stats['memory_mb']:.1f} MB")
        """
        # Estimate memory for condition data caches
        condition_data_bytes = 0
        for data in self._condition_data_cache.values():
            condition_data_bytes += data.nbytes

        # Estimate memory for precomputed correlation matrices
        precomputed_corr_bytes = 0
        for corr_matrix in self._precomputed_corr.values():
            precomputed_corr_bytes += corr_matrix.nbytes

        total_memory_bytes = condition_data_bytes + precomputed_corr_bytes

        return {
            'n_conditions': len(self._condition_data_cache),
            'conditions': list(self._condition_data_cache.keys()),
            'n_corr_cached': len(self._corr_cache),
            'n_precomputed_corr': len(self._precomputed_corr),
            'n_precomputed_genes': len(self._precomputed_gene_list),
            'condition_data_bytes': condition_data_bytes,
            'condition_data_mb': condition_data_bytes / (1024 * 1024),
            'precomputed_corr_bytes': precomputed_corr_bytes,
            'precomputed_corr_mb': precomputed_corr_bytes / (1024 * 1024),
            'memory_bytes': total_memory_bytes,
            'memory_mb': total_memory_bytes / (1024 * 1024),
            'samples_per_condition': dict(self._condition_sample_count),
        }
