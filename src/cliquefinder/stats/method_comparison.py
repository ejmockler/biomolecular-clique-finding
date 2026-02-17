"""
Method Comparison Framework for Multi-Method Clique Differential Testing.

This module provides unified abstractions for running multiple statistical methods
(OLS, LMM, ROAST, Permutation) on the same cliques and quantifying their concordance.

Key Principle: Methods answer different questions - disagreement is informative, not a bug.

| Method        | Question Answered                          | Test Type             |
|---------------|--------------------------------------------|-----------------------|
| OLS/LMM       | "Is aggregated clique abundance different?"| Parametric contrast   |
| ROAST         | "Are genes in this clique enriched for DE?"| Self-contained rotation|
| Permutation   | "Is this clique more DE than random sets?" | Competitive enrichment|

Design Principles:
    1. Frozen (immutable) dataclasses for reproducibility
    2. Protocol-based method interface for extensibility
    3. Unified result format across all methods
    4. Concordance metrics for method comparison

Statistical Note:
    This framework is for DESCRIPTIVE comparison, not inference.
    - We do NOT select the "best" p-value per clique (would inflate FDR)
    - We do NOT combine p-values across methods (requires strong assumptions)
    - We DO report concordance metrics to characterize agreement
    - We DO flag disagreement cases for biological investigation

References:
    - Wu D, Smyth GK (2012). "Camera: a competitive gene set test accounting for
      inter-gene correlation." Nucleic Acids Research 40(17):e133.
    - Subramanian A, et al. (2005). "Gene set enrichment analysis: A knowledge-based
      approach." PNAS 102(43):15545-50.
    - Cohen J (1960). "A coefficient of agreement for nominal scales."
      Educational and Psychological Measurement 20(1):37-46.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


# =============================================================================
# Enums
# =============================================================================


class MethodName(Enum):
    """
    Registered differential testing methods.

    Naming convention: METHOD_VARIANT for methods with multiple modes.

    Attributes:
        OLS: Fixed effects ordinary least squares
        LMM: Linear mixed model with random subject effects
        ROAST_MSQ: ROAST with MSQ statistic (bidirectional, direction-agnostic)
        ROAST_MEAN: ROAST with mean statistic (directional)
        ROAST_FLOORMEAN: ROAST with floormean statistic
        PERMUTATION_COMPETITIVE: Competitive permutation test (random gene set null)
        PERMUTATION_ROTATION: Rotation-based permutation (future extension)
    """

    OLS = "ols"
    LMM = "lmm"
    ROAST_MSQ = "roast_msq"
    ROAST_MEAN = "roast_mean"
    ROAST_FLOORMEAN = "roast_floormean"
    PERMUTATION_COMPETITIVE = "permutation_competitive"
    PERMUTATION_ROTATION = "permutation_rotation"


# =============================================================================
# Core Dataclasses
# =============================================================================


@dataclass(frozen=True)
class UnifiedCliqueResult:
    """
    Standardized result format across all differential testing methods.

    This frozen dataclass provides a common structure for results from OLS, LMM,
    ROAST, and permutation-based methods, enabling direct comparison and
    concordance analysis.

    Design Principles:
        1. Frozen (immutable) for reproducibility and safe sharing
        2. Core fields are semantically equivalent across methods
        3. method_metadata captures method-specific details for deep analysis

    Semantic Equivalence Notes:
        - effect_size: log2FC for OLS/LMM, mean z-score for ROAST, observed t for permutation
        - p_value: parametric for OLS/LMM, exact for ROAST, empirical for permutation
        - statistic_value: t-statistic, rotation statistic, or empirical z-score

    Attributes:
        clique_id: Unique identifier for the clique (typically regulator name)
        method: Which statistical method produced this result
        effect_size: Magnitude of differential expression (interpretation varies by method)
        effect_size_se: Standard error of effect size (None if not applicable)
        p_value: Method-appropriate p-value
        statistic_value: Raw test statistic
        statistic_type: Type of statistic ("t", "z", "msq", "empirical_z", etc.)
        degrees_of_freedom: Degrees of freedom for parametric tests (None for non-parametric)
        n_proteins: Number of proteins in the clique definition
        n_proteins_found: Number of proteins found in the expression data
        method_metadata: Method-specific additional information

    Example:
        >>> result = UnifiedCliqueResult(
        ...     clique_id="TP53",
        ...     method=MethodName.OLS,
        ...     effect_size=1.5,
        ...     effect_size_se=0.3,
        ...     p_value=0.001,
        ...     statistic_value=5.0,
        ...     statistic_type="t",
        ...     degrees_of_freedom=45.0,
        ...     n_proteins=25,
        ...     n_proteins_found=20,
        ... )
        >>> result.is_valid
        True
        >>> result.to_dict()['method']
        'ols'
    """

    # Identity
    clique_id: str
    method: MethodName

    # Core statistics (semantically equivalent across methods)
    effect_size: float
    effect_size_se: float | None
    p_value: float

    # Test statistic details
    statistic_value: float
    statistic_type: str
    degrees_of_freedom: float | None

    # Clique metadata
    n_proteins: int
    n_proteins_found: int

    # Method-specific (for deep analysis)
    method_metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """
        Flatten result for DataFrame construction.

        Returns a dictionary suitable for creating pandas DataFrames.
        Method metadata is flattened with 'meta_' prefix.

        Returns:
            Dictionary with all fields, metadata prefixed with 'meta_'
        """
        base: dict[str, object] = {
            "clique_id": self.clique_id,
            "method": self.method.value,
            "effect_size": self.effect_size,
            "effect_size_se": self.effect_size_se,
            "p_value": self.p_value,
            "statistic_value": self.statistic_value,
            "statistic_type": self.statistic_type,
            "df": self.degrees_of_freedom,
            "n_proteins": self.n_proteins,
            "n_proteins_found": self.n_proteins_found,
        }
        # Flatten metadata with prefix
        for k, v in self.method_metadata.items():
            base[f"meta_{k}"] = v
        return base

    @property
    def is_valid(self) -> bool:
        """
        Check if result has valid statistics.

        A result is valid if:
            - p_value is finite and in [0, 1]
            - effect_size is finite

        Returns:
            True if the result has valid, usable statistics
        """
        return (
            np.isfinite(self.p_value)
            and 0 <= self.p_value <= 1
            and np.isfinite(self.effect_size)
        )

    @property
    def is_significant(self) -> bool:
        """
        Check if result is nominally significant at alpha=0.05.

        Note: This uses unadjusted p-values. For multiple testing correction,
        use FDR-adjusted p-values from the analysis pipeline.

        Returns:
            True if p_value < 0.05 and result is valid
        """
        return self.is_valid and self.p_value < 0.05

    @property
    def coverage_fraction(self) -> float:
        """
        Fraction of clique proteins found in the data.

        Returns:
            n_proteins_found / n_proteins, or 0 if n_proteins is 0
        """
        if self.n_proteins == 0:
            return 0.0
        return self.n_proteins_found / self.n_proteins


@dataclass(frozen=True)
class ConcordanceMetrics:
    """
    Pairwise concordance metrics between two differential testing methods.

    This frozen dataclass captures multiple dimensions of agreement between
    two methods tested on the same set of cliques:
        - Rank agreement (Spearman rho of p-values)
        - Classification agreement (Cohen's kappa for sig/non-sig)
        - Effect size agreement (Pearson r and RMSE)
        - Direction agreement (fraction with same effect sign)

    All metrics are computed on the intersection of cliques that both
    methods successfully tested.

    Interpretation Guide:
        - Spearman rho > 0.8: excellent rank agreement
        - Spearman rho < 0.5: poor rank agreement
        - Cohen's kappa > 0.6: substantial classification agreement
        - Cohen's kappa < 0.2: slight agreement
        - Jaccard > 0.5: majority overlap in significant calls

    Attributes:
        method_a: First method in the comparison
        method_b: Second method in the comparison
        n_cliques_compared: Number of cliques both methods tested
        spearman_rho: Spearman correlation of p-value ranks
        spearman_pvalue: Statistical significance of the rank correlation
        threshold: P-value threshold used for classification (default 0.05)
        n_both_significant: Cliques significant in both methods
        n_both_nonsignificant: Cliques non-significant in both methods
        n_a_only: Cliques significant only in method A
        n_b_only: Cliques significant only in method B
        cohen_kappa: Cohen's kappa for classification agreement
        effect_pearson_r: Pearson correlation of effect sizes
        effect_rmse: Root mean squared error of effect sizes
        direction_agreement_frac: Fraction of cliques with same effect sign

    Example:
        >>> metrics = ConcordanceMetrics(
        ...     method_a=MethodName.OLS,
        ...     method_b=MethodName.ROAST_MSQ,
        ...     n_cliques_compared=100,
        ...     spearman_rho=0.85,
        ...     spearman_pvalue=1e-10,
        ...     threshold=0.05,
        ...     n_both_significant=15,
        ...     n_both_nonsignificant=70,
        ...     n_a_only=5,
        ...     n_b_only=10,
        ...     cohen_kappa=0.65,
        ...     effect_pearson_r=0.72,
        ...     effect_rmse=0.45,
        ...     direction_agreement_frac=0.92,
        ... )
        >>> metrics.jaccard_index
        0.5
        >>> metrics.agreement_rate
        0.85
    """

    method_a: MethodName
    method_b: MethodName
    n_cliques_compared: int

    # Rank-based agreement
    spearman_rho: float
    spearman_pvalue: float

    # Classification agreement at threshold
    threshold: float
    n_both_significant: int
    n_both_nonsignificant: int
    n_a_only: int
    n_b_only: int
    cohen_kappa: float

    # Effect size agreement
    effect_pearson_r: float
    effect_rmse: float

    # Direction agreement (for signed effects)
    direction_agreement_frac: float

    @property
    def jaccard_index(self) -> float:
        """
        Jaccard index: overlap of significant calls.

        Computed as |A intersect B| / |A union B| where A and B are the sets
        of cliques called significant by each method.

        Returns:
            Jaccard similarity coefficient in [0, 1], or 0 if no significant calls
        """
        union = self.n_both_significant + self.n_a_only + self.n_b_only
        if union == 0:
            return 0.0
        return self.n_both_significant / union

    @property
    def agreement_rate(self) -> float:
        """
        Fraction of cliques where methods agree on significance call.

        This is the simple agreement rate (both significant + both non-significant)
        divided by total compared. Note that Cohen's kappa is a better metric
        as it accounts for chance agreement.

        Returns:
            Agreement fraction in [0, 1], or 0 if no cliques compared
        """
        if self.n_cliques_compared == 0:
            return 0.0
        agreed = self.n_both_significant + self.n_both_nonsignificant
        return agreed / self.n_cliques_compared

    @property
    def sensitivity_a(self) -> float:
        """
        Fraction of method A's significant calls that B also calls significant.

        Returns:
            Sensitivity from A's perspective, or NaN if A has no significant calls
        """
        total_a_sig = self.n_both_significant + self.n_a_only
        if total_a_sig == 0:
            return float("nan")
        return self.n_both_significant / total_a_sig

    @property
    def sensitivity_b(self) -> float:
        """
        Fraction of method B's significant calls that A also calls significant.

        Returns:
            Sensitivity from B's perspective, or NaN if B has no significant calls
        """
        total_b_sig = self.n_both_significant + self.n_b_only
        if total_b_sig == 0:
            return float("nan")
        return self.n_both_significant / total_b_sig

    def to_dict(self) -> dict[str, object]:
        """
        Flatten metrics for DataFrame construction.

        Returns:
            Dictionary with all metrics, including computed properties
        """
        return {
            "method_a": self.method_a.value,
            "method_b": self.method_b.value,
            "n_compared": self.n_cliques_compared,
            "spearman_rho": self.spearman_rho,
            "spearman_pvalue": self.spearman_pvalue,
            "threshold": self.threshold,
            "n_both_sig": self.n_both_significant,
            "n_both_nonsig": self.n_both_nonsignificant,
            "n_a_only": self.n_a_only,
            "n_b_only": self.n_b_only,
            "cohen_kappa": self.cohen_kappa,
            "jaccard": self.jaccard_index,
            "agreement_rate": self.agreement_rate,
            "effect_r": self.effect_pearson_r,
            "effect_rmse": self.effect_rmse,
            "direction_agreement": self.direction_agreement_frac,
        }

    def summary(self) -> str:
        """
        Generate human-readable summary of concordance.

        Returns:
            Multi-line string summarizing the concordance metrics
        """
        lines = [
            f"Concordance: {self.method_a.value} vs {self.method_b.value}",
            f"  Cliques compared: {self.n_cliques_compared}",
            f"  Spearman rho: {self.spearman_rho:.3f} (p={self.spearman_pvalue:.2e})",
            f"  Cohen's kappa: {self.cohen_kappa:.3f}",
            f"  Jaccard index: {self.jaccard_index:.3f}",
            f"  Agreement rate: {self.agreement_rate:.1%}",
            f"  Direction agreement: {self.direction_agreement_frac:.1%}",
            f"  Classification (threshold={self.threshold}):",
            f"    Both significant: {self.n_both_significant}",
            f"    Both non-significant: {self.n_both_nonsignificant}",
            f"    {self.method_a.value} only: {self.n_a_only}",
            f"    {self.method_b.value} only: {self.n_b_only}",
        ]
        return "\n".join(lines)


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class CliqueTestMethod(Protocol):
    """
    Protocol for clique differential testing methods.

    All methods implementing this protocol must:
        1. Have a `name` property returning their MethodName
        2. Have a `test` method accepting PreparedCliqueExperiment
        3. Return list[UnifiedCliqueResult]
        4. Be stateless (no side effects on experiment)

    Methods MAY accept additional kwargs for method-specific configuration
    in their test() method.

    This protocol enables:
        - Uniform interface for all testing methods
        - Easy addition of new methods
        - Type-safe method comparison framework

    Example Implementation:
        >>> class MyCustomMethod:
        ...     @property
        ...     def name(self) -> MethodName:
        ...         return MethodName.OLS  # or custom enum value
        ...
        ...     def test(
        ...         self,
        ...         experiment: PreparedCliqueExperiment,
        ...         **kwargs,
        ...     ) -> list[UnifiedCliqueResult]:
        ...         results = []
        ...         for clique in experiment.cliques:
        ...             # ... run analysis ...
        ...             results.append(UnifiedCliqueResult(...))
        ...         return results
    """

    @property
    def name(self) -> MethodName:
        """
        Unique identifier for this method.

        Returns:
            MethodName enum value identifying this method
        """
        ...

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        **kwargs: object,
    ) -> list[UnifiedCliqueResult]:
        """
        Run differential test on all cliques in the experiment.

        This method should:
            - Not modify the experiment (it's frozen)
            - Return one UnifiedCliqueResult per clique with sufficient data
            - Skip cliques without enough proteins in the data
            - Handle errors gracefully (skip problematic cliques)

        Args:
            experiment: Prepared data (immutable, shared across methods)
            **kwargs: Method-specific options (e.g., use_gpu, seed, verbose)

        Returns:
            List of UnifiedCliqueResult, one per clique with data.
            Order should match experiment.cliques where possible.
        """
        ...


# =============================================================================
# PreparedCliqueExperiment
# =============================================================================


@dataclass(frozen=True)
class PreparedCliqueExperiment:
    """
    Immutable snapshot of preprocessed data ready for testing.

    ALL methods receive the same PreparedCliqueExperiment, ensuring:
        1. No preprocessing-induced differences between methods
        2. Fair comparison (same input data)
        3. Reproducibility (frozen state)

    This class should be created via prepare_experiment() factory function.

    Attributes:
        data: Expression matrix (n_features, n_samples) as float64.
            This array should be treated as immutable after creation.
        feature_ids: Tuple of feature identifiers (immutable).
        feature_to_idx: Mapping from feature ID to row index.
            Note: dict is mutable but frozen class prevents reassignment.
        sample_metadata: DataFrame with sample annotations.
        condition_column: Name of the condition column in metadata.
        subject_column: Name of the subject column (None for no random effects).
        conditions: Ordered tuple of unique conditions.
        n_samples: Number of samples.
        cliques: Tuple of CliqueDefinition objects (immutable).
        clique_to_feature_indices: Mapping from clique_id to feature row indices.
        contrast: Tuple of (test_condition, reference_condition).
        contrast_name: Human-readable contrast name (e.g., "ALS_vs_Control").
        preprocessing_params: Dict capturing preprocessing provenance.
        creation_timestamp: ISO format timestamp of creation.

    Example:
        >>> experiment = prepare_experiment(
        ...     data=expression_matrix,
        ...     feature_ids=protein_ids,
        ...     sample_metadata=metadata,
        ...     cliques=clique_definitions,
        ...     condition_column='phenotype',
        ...     contrast=('ALS', 'Control'),
        ... )
        >>> print(f"Features: {experiment.n_features}, Cliques: {experiment.n_cliques}")
        >>> clique_data, clique_ids = experiment.get_clique_data("TP53")
    """

    # Expression data (preprocessed)
    data: NDArray[np.float64]
    feature_ids: tuple[str, ...]
    feature_to_idx: dict[str, int]

    # Sample information
    sample_metadata: object  # pd.DataFrame - using object for frozen compatibility
    condition_column: str
    subject_column: str | None
    conditions: tuple[str, ...]
    n_samples: int

    # Clique definitions
    cliques: tuple[object, ...]  # tuple[CliqueDefinition, ...]
    clique_to_feature_indices: dict[str, tuple[int, ...]]

    # ID mapping (symbol -> feature_id for UniProt/Ensembl translation)
    symbol_to_feature: dict[str, str]

    # Contrast
    contrast: tuple[str, str]
    contrast_name: str

    # Provenance
    preprocessing_params: dict[str, object]
    creation_timestamp: str

    @property
    def n_features(self) -> int:
        """Number of features (proteins/genes) in the data."""
        return self.data.shape[0]

    @property
    def n_cliques(self) -> int:
        """Number of clique definitions."""
        return len(self.cliques)

    def get_clique_data(
        self, clique_id: str
    ) -> tuple[NDArray[np.float64], list[str]]:
        """
        Extract expression data for a specific clique.

        Args:
            clique_id: Identifier of the clique to extract

        Returns:
            Tuple of (data_subset, feature_ids_subset) where:
                - data_subset has shape (n_proteins_found, n_samples)
                - feature_ids_subset lists the feature IDs included
                - Returns empty array and list if clique has no proteins in data
        """
        indices = self.clique_to_feature_indices.get(clique_id, ())
        if not indices:
            return np.array([]).reshape(0, self.n_samples), []

        data_subset = self.data[list(indices), :]
        ids_subset = [self.feature_ids[i] for i in indices]
        return data_subset, ids_subset

    def get_condition_mask(self, condition: str) -> NDArray[np.bool_]:
        """
        Get boolean mask for samples in given condition.

        Args:
            condition: Condition label to match

        Returns:
            Boolean array of shape (n_samples,) where True indicates
            the sample belongs to the specified condition.

        Raises:
            TypeError: If sample_metadata is not a pandas DataFrame
        """
        import pandas as pd

        if isinstance(self.sample_metadata, pd.DataFrame):
            return (self.sample_metadata[self.condition_column] == condition).values
        raise TypeError("sample_metadata must be a pandas DataFrame")

    def get_design_matrix(self) -> NDArray[np.float64]:
        """
        Build design matrix for OLS/LMM.

        Constructs a design matrix with an intercept column and dummy-coded
        condition variables. The first condition (alphabetically) is used as
        the reference level and is dropped.

        Returns:
            Design matrix with shape (n_samples, n_conditions).
            First column is the intercept (constant 1).

        Raises:
            TypeError: If sample_metadata is not a pandas DataFrame
            ImportError: If statsmodels is not installed
        """
        import pandas as pd

        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError(
                "statsmodels is required for design matrix construction. "
                "Install with: pip install statsmodels"
            )

        if not isinstance(self.sample_metadata, pd.DataFrame):
            raise TypeError("sample_metadata must be a pandas DataFrame")

        condition_cat = pd.Categorical(
            self.sample_metadata[self.condition_column],
            categories=list(self.conditions)
        )
        X = pd.get_dummies(
            pd.DataFrame({'condition': condition_cat}),
            drop_first=True,
            dtype=float
        )
        X = sm.add_constant(X)
        return X.values


# =============================================================================
# Factory Function: prepare_experiment()
# =============================================================================


def prepare_experiment(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_metadata: object,  # pd.DataFrame
    cliques: list[object],    # list[CliqueDefinition]
    condition_column: str,
    contrast: tuple[str, str],
    subject_column: str | None = None,
    normalization_method: str = "median",
    imputation_method: str = "min_feature",
    map_ids: bool = True,
    verbose: bool = True,
    precomputed_symbol_map: dict[str, str] | None = None,
) -> PreparedCliqueExperiment:
    """
    Prepare data for multi-method comparison.

    This is the SINGLE preprocessing entry point. All methods receive
    the same prepared data, ensuring fair comparison.

    Preprocessing pipeline:
        1. Copy data (never modify input)
        2. Normalize (median centering, quantile, etc.)
        3. Impute missing values
        4. Map feature IDs (UniProt -> Symbol if needed)
        5. Build clique -> feature index mapping
        6. Validate contrast

    Args:
        data: Expression matrix (n_features, n_samples) of log2 intensities.
        feature_ids: Feature identifiers (UniProt, Ensembl, or Symbol).
        sample_metadata: DataFrame with sample annotations. Must contain
            the condition_column and optionally subject_column.
        cliques: List of CliqueDefinition objects to analyze.
        condition_column: Metadata column name for condition labels.
        contrast: Tuple of (test_condition, reference_condition).
        subject_column: Optional metadata column for biological replicates
            (required for LMM).
        normalization_method: Normalization to apply. Options: "none",
            "median", "quantile", "global_standards", "vsn".
        imputation_method: Imputation for missing values. Options: "none",
            "min_feature", "min_global", "min_sample", "aft", "knn", "qrilc".
        map_ids: Whether to map feature IDs to gene symbols for clique
            matching. Set to False if IDs are already gene symbols.
        verbose: Print progress messages.

    Returns:
        PreparedCliqueExperiment (frozen, immutable snapshot).

    Raises:
        ValueError: If contrast conditions not found in data.
        ValueError: If no cliques have proteins in data.

    Example:
        >>> from cliquefinder.stats.method_comparison import prepare_experiment
        >>> experiment = prepare_experiment(
        ...     data=expression_matrix,
        ...     feature_ids=protein_ids,
        ...     sample_metadata=metadata,
        ...     cliques=clique_definitions,
        ...     condition_column='phenotype',
        ...     contrast=('ALS', 'Control'),
        ...     subject_column='subject_id',
        ... )
        >>> print(f"Prepared: {experiment.n_features} features, {experiment.n_cliques} cliques")
    """
    import pandas as pd
    from datetime import datetime

    # Import preprocessing modules
    from .normalization import NormalizationMethod, normalize
    from .missing import ImputationMethod, impute_missing_values
    from .clique_analysis import map_feature_ids_to_symbols

    # Convert string to enum if needed
    if isinstance(normalization_method, str):
        norm_method = NormalizationMethod(normalization_method)
    else:
        norm_method = normalization_method

    if isinstance(imputation_method, str):
        imp_method = ImputationMethod(imputation_method)
    else:
        imp_method = imputation_method

    if verbose:
        print("Preparing experiment for method comparison")
        print(f"  Data: {data.shape[0]} features x {data.shape[1]} samples")
        print(f"  Cliques: {len(cliques)}")
        print(f"  Contrast: {contrast[0]} vs {contrast[1]}")

    # 1. Copy and convert to float64 (never modify input)
    work_data = data.astype(np.float64).copy()
    work_ids = list(feature_ids)

    # 2. Normalize
    if norm_method != NormalizationMethod.NONE:
        if verbose:
            print(f"  Normalizing: {norm_method.value}")
        norm_result = normalize(work_data, method=norm_method)
        work_data = norm_result.data

    # 3. Impute missing values
    if imp_method != ImputationMethod.NONE:
        n_missing_before = int(np.sum(np.isnan(work_data)))
        if verbose and n_missing_before > 0:
            missing_rate = n_missing_before / work_data.size
            print(f"  Missing values: {n_missing_before} ({missing_rate:.1%})")
            print(f"  Imputing: {imp_method.value}")
        imp_result = impute_missing_values(work_data, method=imp_method)
        work_data = imp_result.data

    # 4. ID mapping (UniProt/Ensembl -> Symbol)
    # If precomputed_symbol_map is provided, use it directly (critical for bootstrap efficiency)
    symbol_to_feature: dict[str, str] = {}
    if precomputed_symbol_map is not None:
        symbol_to_feature = precomputed_symbol_map
        if verbose:
            print(f"  Using precomputed symbol map ({len(symbol_to_feature)} mappings)")
    elif map_ids and len(cliques) > 0:
        # Check if mapping is needed by sampling clique proteins
        sample_proteins: list[str] = []
        for clique in cliques[:10]:
            # Access protein_ids attribute safely
            if hasattr(clique, 'protein_ids'):
                sample_proteins.extend(clique.protein_ids[:3])

        matches = sum(1 for p in sample_proteins if p in work_ids)
        if sample_proteins and matches < len(sample_proteins) * 0.5:
            if verbose:
                print("  Mapping feature IDs to symbols...")
            # Uses module-level cache for efficiency across bootstrap iterations
            symbol_to_feature = map_feature_ids_to_symbols(work_ids, verbose=verbose)

    # 5. Build feature index map
    feature_to_idx: dict[str, int] = {fid: i for i, fid in enumerate(work_ids)}

    # Also add symbol -> idx mapping if we did ID conversion
    if symbol_to_feature:
        for symbol, feature_id in symbol_to_feature.items():
            if feature_id in feature_to_idx:
                feature_to_idx[symbol] = feature_to_idx[feature_id]

    # 6. Build clique -> feature indices mapping
    clique_to_indices: dict[str, tuple[int, ...]] = {}
    n_cliques_with_data = 0

    for clique in cliques:
        indices: list[int] = []
        # Access protein_ids attribute safely
        protein_ids = getattr(clique, 'protein_ids', [])
        clique_id = getattr(clique, 'clique_id', str(clique))

        for pid in protein_ids:
            # Try direct match first
            if pid in feature_to_idx:
                indices.append(feature_to_idx[pid])
            # Try symbol mapping (already added to feature_to_idx if available)
            elif symbol_to_feature and pid in symbol_to_feature:
                mapped = symbol_to_feature[pid]
                if mapped in feature_to_idx:
                    indices.append(feature_to_idx[mapped])

        if indices:
            clique_to_indices[clique_id] = tuple(indices)
            n_cliques_with_data += 1

    if n_cliques_with_data == 0:
        # Get sample info for error message
        sample_clique_proteins: list[str] = []
        if cliques and hasattr(cliques[0], 'protein_ids'):
            sample_clique_proteins = cliques[0].protein_ids[:5]
        raise ValueError(
            "No cliques have any proteins in the data. "
            "Check that protein IDs match between clique definitions and feature_ids. "
            f"Sample clique proteins: {sample_clique_proteins}, "
            f"Sample feature IDs: {work_ids[:5]}"
        )

    if verbose:
        print(f"  Cliques with data: {n_cliques_with_data}/{len(cliques)}")

    # 7. Validate conditions and contrast
    if not isinstance(sample_metadata, pd.DataFrame):
        raise TypeError("sample_metadata must be a pandas DataFrame")

    available_conditions = sorted(
        sample_metadata[condition_column].dropna().unique().tolist()
    )
    test_condition, ref_condition = contrast

    if test_condition not in available_conditions:
        raise ValueError(
            f"Test condition '{test_condition}' not found in data. "
            f"Available conditions: {available_conditions}"
        )
    if ref_condition not in available_conditions:
        raise ValueError(
            f"Reference condition '{ref_condition}' not found in data. "
            f"Available conditions: {available_conditions}"
        )

    # 8. Build preprocessing params for provenance
    preprocessing_params: dict[str, object] = {
        'normalization': norm_method.value,
        'imputation': imp_method.value,
        'map_ids': map_ids,
        'n_features_original': len(feature_ids),
        'n_samples': work_data.shape[1],
        'n_cliques_total': len(cliques),
        'n_cliques_with_data': n_cliques_with_data,
        'n_symbols_mapped': len(symbol_to_feature),
    }

    # 9. Build and return the frozen experiment
    creation_ts = datetime.now().isoformat()
    contrast_name = f"{test_condition}_vs_{ref_condition}"

    if verbose:
        print(f"  Creation timestamp: {creation_ts}")
        print()

    return PreparedCliqueExperiment(
        data=work_data,
        feature_ids=tuple(work_ids),
        feature_to_idx=feature_to_idx,
        sample_metadata=sample_metadata.copy(),
        condition_column=condition_column,
        subject_column=subject_column,
        conditions=tuple(available_conditions),
        n_samples=work_data.shape[1],
        cliques=tuple(cliques),
        clique_to_feature_indices=clique_to_indices,
        symbol_to_feature=symbol_to_feature,
        contrast=contrast,
        contrast_name=contrast_name,
        preprocessing_params=preprocessing_params,
        creation_timestamp=creation_ts,
    )


# =============================================================================
# Method Implementations
# =============================================================================


class OLSMethod:
    """
    OLS-based clique differential testing with optional Empirical Bayes moderation.

    This method wraps the existing differential analysis infrastructure to test
    clique-level differential abundance using fixed effects ordinary least squares.

    Statistical Approach:
        1. Summarize proteins within each clique via Tukey's Median Polish (or other method)
        2. Fit OLS: summarized_abundance ~ condition
        3. Test contrast with t-statistic
        4. Return UnifiedCliqueResult for each clique

    The summarization step aggregates protein-level data into a single clique-level
    abundance per sample, which is then tested for differential abundance between
    conditions.

    Attributes:
        summarization: Method for aggregating proteins within clique.
            Options: "tmp" (Tukey's Median Polish), "median", "mean", "logsum", "pca"
        eb_moderation: Whether to apply Empirical Bayes variance shrinkage (future).
            Currently not implemented but reserved for limma-style moderation.

    Example:
        >>> from cliquefinder.stats.method_comparison import OLSMethod, prepare_experiment
        >>> method = OLSMethod(summarization="tmp", eb_moderation=True)
        >>> experiment = prepare_experiment(data, feature_ids, metadata, cliques, ...)
        >>> results = method.test(experiment)
        >>> significant = [r for r in results if r.p_value < 0.05]

    See Also:
        - LMMMethod: Linear mixed model variant with random subject effects
        - differential_analysis_single: Underlying statistical function
        - summarize_clique: Summarization function
    """

    def __init__(
        self,
        summarization: str = "tmp",
        eb_moderation: bool = True,
    ) -> None:
        """
        Initialize OLS method adapter.

        Args:
            summarization: Summarization method for aggregating proteins.
                Options: "tmp" (Tukey's Median Polish), "median", "mean", "logsum", "pca".
                Default is "tmp" which is robust to outliers.
            eb_moderation: Whether to apply Empirical Bayes variance shrinkage.
                Currently reserved for future implementation.
        """
        self.summarization = summarization
        self.eb_moderation = eb_moderation

    @property
    def name(self) -> MethodName:
        """Return the method identifier."""
        return MethodName.OLS

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        verbose: bool = False,
        **kwargs: object,
    ) -> list[UnifiedCliqueResult]:
        """
        Run OLS differential test on all cliques in the experiment.

        For each clique:
            1. Extract protein expression data using experiment.get_clique_data()
            2. Summarize to clique-level abundance using the specified method
            3. Fit OLS model: abundance ~ condition
            4. Test the specified contrast
            5. Package results into UnifiedCliqueResult

        Cliques with fewer than 2 proteins found in the data are skipped.

        Args:
            experiment: Prepared and preprocessed experiment data (immutable).
            verbose: If True, print progress messages.
            **kwargs: Additional keyword arguments (ignored for OLS).

        Returns:
            List of UnifiedCliqueResult objects, one per clique with sufficient data.
            Results are ordered by clique definition order in the experiment.
        """
        import pandas as pd

        from .summarization import SummarizationMethod, summarize_clique
        from .differential import differential_analysis_single, build_contrast_matrix

        results: list[UnifiedCliqueResult] = []

        # Build contrast matrix for the experiment's contrast
        conditions = list(experiment.conditions)
        contrast_dict = {experiment.contrast_name: experiment.contrast}
        contrast_matrix, contrast_names = build_contrast_matrix(conditions, contrast_dict)

        # Get sample condition labels
        if not isinstance(experiment.sample_metadata, pd.DataFrame):
            raise TypeError("experiment.sample_metadata must be a pandas DataFrame")

        sample_condition = experiment.sample_metadata[experiment.condition_column].values

        # Resolve summarization method
        try:
            sum_method = SummarizationMethod(self.summarization)
        except ValueError:
            sum_method = SummarizationMethod.TUKEY_MEDIAN_POLISH

        n_processed = 0
        n_skipped = 0

        for clique in experiment.cliques:
            # Access clique attributes
            clique_id = getattr(clique, 'clique_id', str(clique))
            protein_ids_definition = getattr(clique, 'protein_ids', [])

            # Get clique data from experiment
            clique_data, clique_features = experiment.get_clique_data(clique_id)

            if len(clique_features) < 2:
                # Not enough proteins found in data
                n_skipped += 1
                continue

            # Summarize proteins to clique level
            summary = summarize_clique(
                clique_data,
                clique_features,
                clique_id,
                method=sum_method,
                compute_coherence=True,
            )

            # Run differential analysis on summarized data
            try:
                protein_result = differential_analysis_single(
                    intensities=summary.sample_abundances,
                    condition=sample_condition,
                    subject=None,  # OLS = no random effects
                    feature_id=clique_id,
                    contrast_matrix=contrast_matrix,
                    contrast_names=contrast_names,
                    conditions=conditions,
                    use_mixed=False,
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: OLS failed for {clique_id}: {e}")
                n_skipped += 1
                continue

            # Extract contrast result and convert to UnifiedCliqueResult
            if protein_result.contrasts:
                contrast = protein_result.contrasts[0]

                results.append(UnifiedCliqueResult(
                    clique_id=clique_id,
                    method=self.name,
                    effect_size=contrast.log2_fc,
                    effect_size_se=contrast.se,
                    p_value=contrast.p_value,
                    statistic_value=contrast.t_value,
                    statistic_type="t",
                    degrees_of_freedom=contrast.df,
                    n_proteins=len(protein_ids_definition),
                    n_proteins_found=len(clique_features),
                    method_metadata={
                        'summarization': self.summarization,
                        'eb_moderation': self.eb_moderation,
                        'ci_lower': contrast.ci_lower,
                        'ci_upper': contrast.ci_upper,
                        'coherence': summary.coherence,
                        'model_type': protein_result.model_type.value,
                    },
                ))
                n_processed += 1
            else:
                n_skipped += 1

        if verbose:
            print(f"OLS: processed {n_processed} cliques, skipped {n_skipped}")

        return results


class LMMMethod:
    """
    Linear Mixed Model clique differential testing with random subject effects.

    This method extends OLS by incorporating random effects for biological subjects,
    properly accounting for repeated measurements from the same individuals.

    Statistical Approach:
        1. Summarize proteins within each clique via Tukey's Median Polish (or other method)
        2. Fit LMM: summarized_abundance ~ condition + (1 | subject)
        3. Compute Satterthwaite degrees of freedom for the contrast
        4. Test contrast with t-statistic
        5. Return UnifiedCliqueResult for each clique

    The random intercept for subject captures between-subject variability,
    preventing pseudoreplication when subjects have multiple samples.

    Attributes:
        summarization: Method for aggregating proteins within clique.
            Options: "tmp" (Tukey's Median Polish), "median", "mean", "logsum", "pca"
        use_satterthwaite: Whether to use Satterthwaite degrees of freedom approximation.
            If True (default), computes more accurate df for mixed models.

    Requires:
        experiment.subject_column must be set to the column name containing
        subject identifiers.

    Example:
        >>> from cliquefinder.stats.method_comparison import LMMMethod, prepare_experiment
        >>> method = LMMMethod(summarization="tmp", use_satterthwaite=True)
        >>> experiment = prepare_experiment(
        ...     data, feature_ids, metadata, cliques,
        ...     condition_column='phenotype',
        ...     contrast=('ALS', 'Control'),
        ...     subject_column='subject_id',  # Required for LMM
        ... )
        >>> results = method.test(experiment)

    See Also:
        - OLSMethod: Fixed effects variant without random subject effects
        - differential_analysis_single: Underlying statistical function
        - satterthwaite_df: Degrees of freedom approximation
    """

    def __init__(
        self,
        summarization: str = "tmp",
        use_satterthwaite: bool = True,
    ) -> None:
        """
        Initialize LMM method adapter.

        Args:
            summarization: Summarization method for aggregating proteins.
                Options: "tmp" (Tukey's Median Polish), "median", "mean", "logsum", "pca".
                Default is "tmp" which is robust to outliers.
            use_satterthwaite: Whether to use Satterthwaite approximation for df.
                Default True. Set to False for faster but less accurate df.
        """
        self.summarization = summarization
        self.use_satterthwaite = use_satterthwaite

    @property
    def name(self) -> MethodName:
        """Return the method identifier."""
        return MethodName.LMM

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        verbose: bool = False,
        **kwargs: object,
    ) -> list[UnifiedCliqueResult]:
        """
        Run LMM differential test on all cliques in the experiment.

        For each clique:
            1. Extract protein expression data using experiment.get_clique_data()
            2. Summarize to clique-level abundance using the specified method
            3. Fit LMM: abundance ~ condition + (1 | subject)
            4. Test the specified contrast with Satterthwaite df
            5. Package results into UnifiedCliqueResult

        Cliques with fewer than 2 proteins found in the data are skipped.

        Args:
            experiment: Prepared and preprocessed experiment data (immutable).
                Must have subject_column set for random effects.
            verbose: If True, print progress messages.
            **kwargs: Additional keyword arguments (ignored for LMM).

        Returns:
            List of UnifiedCliqueResult objects, one per clique with sufficient data.

        Raises:
            ValueError: If experiment.subject_column is None.
        """
        import pandas as pd

        from .summarization import SummarizationMethod, summarize_clique
        from .differential import differential_analysis_single, build_contrast_matrix

        # Validate that subject column is available for mixed model
        if experiment.subject_column is None:
            raise ValueError(
                "LMMMethod requires subject_column in experiment. "
                "Set subject_column when calling prepare_experiment()."
            )

        results: list[UnifiedCliqueResult] = []

        # Build contrast matrix for the experiment's contrast
        conditions = list(experiment.conditions)
        contrast_dict = {experiment.contrast_name: experiment.contrast}
        contrast_matrix, contrast_names = build_contrast_matrix(conditions, contrast_dict)

        # Get sample condition and subject labels
        if not isinstance(experiment.sample_metadata, pd.DataFrame):
            raise TypeError("experiment.sample_metadata must be a pandas DataFrame")

        sample_condition = experiment.sample_metadata[experiment.condition_column].values
        sample_subject = experiment.sample_metadata[experiment.subject_column].values

        # Resolve summarization method
        try:
            sum_method = SummarizationMethod(self.summarization)
        except ValueError:
            sum_method = SummarizationMethod.TUKEY_MEDIAN_POLISH

        n_processed = 0
        n_skipped = 0
        n_fallback_to_ols = 0

        for clique in experiment.cliques:
            # Access clique attributes
            clique_id = getattr(clique, 'clique_id', str(clique))
            protein_ids_definition = getattr(clique, 'protein_ids', [])

            # Get clique data from experiment
            clique_data, clique_features = experiment.get_clique_data(clique_id)

            if len(clique_features) < 2:
                # Not enough proteins found in data
                n_skipped += 1
                continue

            # Summarize proteins to clique level
            summary = summarize_clique(
                clique_data,
                clique_features,
                clique_id,
                method=sum_method,
                compute_coherence=True,
            )

            # Run differential analysis with mixed model
            try:
                protein_result = differential_analysis_single(
                    intensities=summary.sample_abundances,
                    condition=sample_condition,
                    subject=sample_subject,  # LMM uses subject for random effects
                    feature_id=clique_id,
                    contrast_matrix=contrast_matrix,
                    contrast_names=contrast_names,
                    conditions=conditions,
                    use_mixed=True,  # Key difference from OLS
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: LMM failed for {clique_id}: {e}")
                n_skipped += 1
                continue

            # Track if model fell back to fixed effects
            from .differential import ModelType
            if protein_result.model_type == ModelType.FIXED:
                n_fallback_to_ols += 1

            # Extract contrast result and convert to UnifiedCliqueResult
            if protein_result.contrasts:
                contrast = protein_result.contrasts[0]

                results.append(UnifiedCliqueResult(
                    clique_id=clique_id,
                    method=self.name,
                    effect_size=contrast.log2_fc,
                    effect_size_se=contrast.se,
                    p_value=contrast.p_value,
                    statistic_value=contrast.t_value,
                    statistic_type="t",
                    degrees_of_freedom=contrast.df,
                    n_proteins=len(protein_ids_definition),
                    n_proteins_found=len(clique_features),
                    method_metadata={
                        'summarization': self.summarization,
                        'use_satterthwaite': self.use_satterthwaite,
                        'ci_lower': contrast.ci_lower,
                        'ci_upper': contrast.ci_upper,
                        'coherence': summary.coherence,
                        'model_type': protein_result.model_type.value,
                        'subject_variance': protein_result.subject_variance,
                        'residual_variance': protein_result.residual_variance,
                        'convergence': protein_result.convergence,
                        'issue': protein_result.issue,
                    },
                ))
                n_processed += 1
            else:
                n_skipped += 1

        if verbose:
            print(f"LMM: processed {n_processed} cliques, skipped {n_skipped}")
            if n_fallback_to_ols > 0:
                print(f"     (note: {n_fallback_to_ols} fell back to OLS due to convergence)")

        return results


# =============================================================================
# ROAST Method Implementation
# =============================================================================


class ROASTMethod:
    """
    Rotation-based gene set testing (ROAST) method adapter.

    Tests whether genes within a clique show coordinated differential
    expression while preserving inter-gene correlation structure.
    This is a self-contained test - it asks "Are genes in this clique
    enriched for differential expression?"

    Key Advantages:
        - MSQ statistic detects bidirectional regulation (TFs that both
          activate AND repress different targets)
        - Preserves inter-gene correlation structure via rotation
        - Exact p-values from rotation distribution

    Statistical Approach:
        1. Project gene expression onto residual space (QR decomposition)
        2. Generate random rotations that preserve correlation structure
        3. Compute gene set statistics for observed and rotated data
        4. Compute exact p-value from rotation null distribution

    Attributes:
        statistic: Set statistic to use ("msq", "mean", or "floormean")
        alternative: Alternative hypothesis ("up", "down", or "mixed")
        n_rotations: Number of random rotations for p-value estimation

    Example:
        >>> method = ROASTMethod(statistic="msq", n_rotations=9999)
        >>> results = method.test(experiment)
        >>> # MSQ statistic is direction-agnostic, detects bidirectional regulation
    """

    def __init__(
        self,
        statistic: str = "msq",
        alternative: str = "mixed",
        n_rotations: int = 9999,
    ) -> None:
        """
        Initialize ROAST method adapter.

        Args:
            statistic: Set statistic for aggregating gene-level effects.
                - "msq": Mean of squared z-scores (direction-agnostic, detects
                  bidirectional regulation - RECOMMENDED for TF cliques)
                - "mean": Mean z-score (directional, detects coordinated up/down)
                - "floormean": Mean with floored small effects (robust)
            alternative: Alternative hypothesis specification.
                - "mixed": Genes DE in either direction (two-sided, RECOMMENDED)
                - "up": Genes up-regulated (one-sided)
                - "down": Genes down-regulated (one-sided)
            n_rotations: Number of random rotations. Higher = more precise p-values.
                9999 is standard (gives minimum p-value of 1/10000 = 0.0001).
        """
        self.statistic = statistic
        self.alternative = alternative
        self.n_rotations = n_rotations

    @property
    def name(self) -> MethodName:
        """
        Get the method name based on the configured statistic.

        Returns:
            MethodName.ROAST_MSQ, ROAST_MEAN, or ROAST_FLOORMEAN
        """
        if self.statistic == "msq":
            return MethodName.ROAST_MSQ
        elif self.statistic == "mean":
            return MethodName.ROAST_MEAN
        else:
            return MethodName.ROAST_FLOORMEAN

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        use_gpu: bool = True,
        seed: int | None = None,
        verbose: bool = False,
        **kwargs: object,
    ) -> list[UnifiedCliqueResult]:
        """
        Run ROAST rotation test on all cliques in the experiment.

        Args:
            experiment: Prepared clique experiment (immutable)
            use_gpu: Whether to use GPU acceleration (requires MLX)
            seed: Random seed for reproducibility
            verbose: Print progress messages
            **kwargs: Additional arguments (ignored)

        Returns:
            List of UnifiedCliqueResult, one per clique with sufficient data
        """
        import logging
        import pandas as pd

        logger = logging.getLogger(__name__)

        # Import rotation module components
        try:
            from .rotation import (
                RotationTestEngine,
                RotationTestConfig,
            )
        except ImportError as e:
            logger.warning(f"ROAST method unavailable: {e}")
            return []

        results: list[UnifiedCliqueResult] = []

        # Initialize rotation engine with full expression data
        if not isinstance(experiment.sample_metadata, pd.DataFrame):
            logger.warning("ROAST requires sample_metadata as DataFrame")
            return []

        engine = RotationTestEngine(
            data=experiment.data,
            gene_ids=list(experiment.feature_ids),
            metadata=experiment.sample_metadata,
        )

        # Fit model (QR decomposition, EB priors)
        try:
            engine.fit(
                conditions=list(experiment.conditions),
                contrast=experiment.contrast,
                condition_column=experiment.condition_column,
            )
        except Exception as e:
            logger.warning(f"ROAST fit failed: {e}")
            return []

        # Build gene sets dict from cliques: clique_id -> feature_ids
        # IMPORTANT: Convert symbols to feature IDs (UniProt) for rotation engine
        gene_sets: dict[str, list[str]] = {}
        clique_metadata: dict[str, tuple[int, int]] = {}  # clique_id -> (n_proteins, n_found)

        # Build reverse map: get feature_ids from experiment
        feature_id_set = set(experiment.feature_ids)

        for clique in experiment.cliques:
            clique_id = getattr(clique, "clique_id", str(clique))
            protein_ids = getattr(clique, "protein_ids", [])

            # Convert symbols to feature IDs and filter to those in data
            feature_ids_for_set: list[str] = []
            for pid in protein_ids:
                # If pid is already a feature ID, use it directly
                if pid in feature_id_set:
                    feature_ids_for_set.append(pid)
                # Otherwise, try symbol -> feature mapping
                elif pid in experiment.symbol_to_feature:
                    fid = experiment.symbol_to_feature[pid]
                    if fid in feature_id_set:
                        feature_ids_for_set.append(fid)

            if len(feature_ids_for_set) >= 2:
                gene_sets[clique_id] = feature_ids_for_set
                clique_metadata[clique_id] = (len(protein_ids), len(feature_ids_for_set))

        if not gene_sets:
            if verbose:
                print("  No cliques with sufficient proteins for ROAST")
            return []

        # Configure test
        config = RotationTestConfig(
            n_rotations=self.n_rotations,
            use_gpu=use_gpu,
            seed=seed,
        )

        # Run tests on all gene sets
        try:
            rotation_results = engine.test_gene_sets(
                gene_sets, config=config, verbose=verbose
            )
        except Exception as e:
            logger.warning(f"ROAST test_gene_sets failed: {e}")
            return []

        # Convert RotationResult to UnifiedCliqueResult
        stat_key = self.statistic
        alt_key = self.alternative

        for rot_result in rotation_results:
            try:
                # Get p-value for the configured statistic and alternative
                p_value = rot_result.get_pvalue(stat_key, alt_key)

                # Get observed statistic (observed_stats is nested: {stat: {alt: value}})
                observed = rot_result.observed_stats.get(stat_key, {}).get(alt_key, np.nan)

                # For effect_size, use mean z-score with 'up' direction (interpretable)
                mean_z = rot_result.observed_stats.get("mean", {}).get("up", 0.0)

                # Get clique size metadata
                n_proteins, n_found = clique_metadata.get(
                    rot_result.feature_set_id, (rot_result.n_genes, rot_result.n_genes_found)
                )

                # Get active proportion if available
                active_prop = rot_result.active_proportion.get(alt_key, np.nan)

                # Helper to safely extract nested stat values
                def get_stat_val(stat: str, alt: str = "mixed") -> float | None:
                    val = rot_result.observed_stats.get(stat, {}).get(alt, np.nan)
                    return float(val) if np.isfinite(val) else None

                results.append(
                    UnifiedCliqueResult(
                        clique_id=rot_result.feature_set_id,
                        method=self.name,
                        effect_size=float(mean_z),  # Mean z-score as effect size proxy
                        effect_size_se=None,  # ROAST doesn't provide SE
                        p_value=float(p_value),
                        statistic_value=float(observed) if np.isfinite(observed) else np.nan,
                        statistic_type=stat_key,
                        degrees_of_freedom=None,  # Rotation test = exact, no df
                        n_proteins=n_proteins,
                        n_proteins_found=n_found,
                        method_metadata={
                            "alternative": alt_key,
                            "n_rotations": rot_result.n_rotations,
                            "active_proportion": float(active_prop) if np.isfinite(active_prop) else None,
                            "observed_msq": get_stat_val("msq", alt_key),
                            "observed_mean": get_stat_val("mean", "up"),
                            "observed_floormean": get_stat_val("floormean", alt_key),
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"ROAST result conversion failed for {rot_result.feature_set_id}: {e}")
                continue

        return results


# =============================================================================
# Permutation Method Implementation
# =============================================================================


class PermutationMethod:
    """
    Competitive permutation-based testing method adapter.

    Tests whether a clique has stronger differential expression than
    random gene sets of the same size from the measured proteome.
    This is a competitive test - it asks "Is this clique more DE than
    random sets of the same size?"

    Null Hypothesis:
        Random gene sets show equal or greater effect than the observed clique.

    Key Characteristics:
        - Non-parametric (no distributional assumptions)
        - Accounts for gene-gene correlation via empirical null
        - Competitive enrichment (relative to background)

    Statistical Approach:
        1. Summarize proteins within clique via Tukey's Median Polish
        2. Compute t-statistic for observed clique
        3. Sample random gene sets of same size from regulated pool
        4. Compute t-statistics for random sets
        5. Empirical p-value = fraction of null >= observed

    Attributes:
        n_permutations: Number of permutations for null distribution
        summarization: Method for aggregating proteins ("tmp" = median polish)

    Example:
        >>> method = PermutationMethod(n_permutations=10000)
        >>> results = method.test(experiment)
        >>> # Low p-value means clique is more DE than expected by chance
    """

    def __init__(
        self,
        n_permutations: int = 10000,
        summarization: str = "tmp",
    ) -> None:
        """
        Initialize permutation method adapter.

        Args:
            n_permutations: Number of random gene set permutations.
                10000 is standard (gives minimum p-value of 1/10001 ~ 0.0001).
                Higher = more precise p-values, more compute time.
            summarization: How to aggregate proteins within clique.
                "tmp": Tukey's Median Polish (robust, recommended)
        """
        self.n_permutations = n_permutations
        self.summarization = summarization

    @property
    def name(self) -> MethodName:
        """
        Get the method name.

        Returns:
            MethodName.PERMUTATION_COMPETITIVE
        """
        return MethodName.PERMUTATION_COMPETITIVE

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        use_gpu: bool = True,
        seed: int | None = None,
        verbose: bool = False,
        **kwargs: object,
    ) -> list[UnifiedCliqueResult]:
        """
        Run competitive permutation test on all cliques in the experiment.

        Args:
            experiment: Prepared clique experiment (immutable)
            use_gpu: Whether to use GPU acceleration (requires MLX)
            seed: Random seed for reproducibility
            verbose: Print progress messages
            **kwargs: Additional arguments (ignored)

        Returns:
            List of UnifiedCliqueResult, one per clique with sufficient data
        """
        import logging
        import pandas as pd
        from scipy import stats as scipy_stats

        logger = logging.getLogger(__name__)

        # Import permutation module
        try:
            from .permutation_gpu import run_permutation_test_gpu
        except ImportError as e:
            logger.warning(f"Permutation method unavailable (GPU): {e}")
            # Could fall back to CPU version here if available
            return []

        results: list[UnifiedCliqueResult] = []

        if not isinstance(experiment.sample_metadata, pd.DataFrame):
            logger.warning("Permutation method requires sample_metadata as DataFrame")
            return []

        # Run GPU-accelerated permutation test
        try:
            perm_results, null_df = run_permutation_test_gpu(
                data=experiment.data,
                feature_ids=list(experiment.feature_ids),
                sample_metadata=experiment.sample_metadata,
                clique_definitions=list(experiment.cliques),
                condition_col=experiment.condition_column,
                contrast=experiment.contrast,
                subject_col=experiment.subject_column,
                n_permutations=self.n_permutations,
                random_state=seed,
                verbose=verbose,
            )
        except Exception as e:
            logger.warning(f"Permutation test failed: {e}")
            return []

        # Convert PermutationTestResult to UnifiedCliqueResult
        # perm_results is a list of PermutationTestResult dataclass objects
        for perm_result in perm_results:
            try:
                # Extract fields from PermutationTestResult
                clique_id = perm_result.clique_id
                observed_t = perm_result.observed_tvalue
                empirical_p = perm_result.empirical_pvalue

                # Compute empirical z-score from p-value using inverse normal CDF
                # z = -Phi^-1(p) for upper tail
                # For two-sided p-value, we use the sign of the observed t
                if empirical_p <= 0 or empirical_p >= 1:
                    # Edge case: clamp p-value
                    empirical_p = max(1e-15, min(1 - 1e-15, empirical_p))

                # Convert p-value to z-score (preserving sign from observed t)
                if observed_t >= 0:
                    z_score = -scipy_stats.norm.ppf(empirical_p / 2)  # Two-sided
                else:
                    z_score = scipy_stats.norm.ppf(empirical_p / 2)

                # Get clique size from the original clique definition
                clique_def = None
                for c in experiment.cliques:
                    if getattr(c, "clique_id", str(c)) == clique_id:
                        clique_def = c
                        break

                if clique_def is not None:
                    n_proteins = len(getattr(clique_def, "protein_ids", []))
                else:
                    n_proteins = 0

                # n_proteins_found from looking at how many are in feature_to_idx
                if clique_def is not None:
                    protein_ids = getattr(clique_def, "protein_ids", [])
                    n_found = sum(1 for p in protein_ids if p in experiment.feature_to_idx)
                else:
                    n_found = 0

                # Build method metadata from null distribution info
                null_row = null_df[null_df["clique_id"] == clique_id] if not null_df.empty else None
                null_mean = float(null_row["null_tvalue_mean"].iloc[0]) if null_row is not None and len(null_row) > 0 else np.nan
                null_std = float(null_row["null_tvalue_std"].iloc[0]) if null_row is not None and len(null_row) > 0 else np.nan

                results.append(
                    UnifiedCliqueResult(
                        clique_id=clique_id,
                        method=self.name,
                        effect_size=float(observed_t),  # Use observed t as effect size
                        effect_size_se=None,  # Permutation doesn't provide SE
                        p_value=float(empirical_p),
                        statistic_value=float(z_score),
                        statistic_type="empirical_z",
                        degrees_of_freedom=None,  # Non-parametric
                        n_proteins=n_proteins,
                        n_proteins_found=n_found,
                        method_metadata={
                            "n_permutations": self.n_permutations,
                            "percentile": float(perm_result.percentile_rank),
                            "null_mean": null_mean,
                            "null_std": null_std,
                            "observed_log2fc": float(perm_result.observed_log2fc),
                            "is_significant_empirical": perm_result.is_significant,
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"Permutation result conversion failed for clique: {e}")
                continue

        return results


# =============================================================================
# Concordance Computation Functions
# =============================================================================


def compute_pairwise_concordance(
    results_a: list[UnifiedCliqueResult],
    results_b: list[UnifiedCliqueResult],
    threshold: float = 0.05,
) -> ConcordanceMetrics:
    """
    Compute concordance metrics between two differential testing methods.

    This function quantifies agreement between two methods across multiple
    dimensions: rank correlation of p-values, classification agreement
    (significant vs non-significant), effect size correlation, and
    direction agreement.

    Statistical Background:
        - Spearman rho measures monotonic relationship between p-value ranks
        - Cohen's kappa adjusts for chance agreement in classification
        - Pearson r measures linear relationship between effect sizes
        - RMSE quantifies average magnitude of effect size differences

    Args:
        results_a: List of UnifiedCliqueResult from method A.
            Must contain at least 3 valid results.
        results_b: List of UnifiedCliqueResult from method B.
            Must contain at least 3 valid results.
        threshold: P-value threshold for significant/non-significant
            classification. Default is 0.05 (nominal significance level).

    Returns:
        ConcordanceMetrics dataclass with all agreement measures computed
        on the intersection of valid cliques tested by both methods.

    Raises:
        ValueError: If fewer than 3 cliques are common to both result sets.
            At least 3 observations are needed for meaningful correlation.

    Example:
        >>> from cliquefinder.stats.method_comparison import compute_pairwise_concordance
        >>> conc = compute_pairwise_concordance(ols_results, roast_results, threshold=0.05)
        >>> print(f"Spearman rho: {conc.spearman_rho:.3f}")
        >>> print(f"Cohen's kappa: {conc.cohen_kappa:.3f}")
        >>> print(f"Agreement rate: {conc.agreement_rate:.1%}")

    Notes:
        - Only valid results (finite p-value in [0,1] and finite effect size)
          are included in the comparison
        - Cliques must have matching clique_id to be compared
        - For effect size comparisons, both results must have finite effect sizes
    """
    from scipy import stats as scipy_stats

    # Build lookup by clique_id for valid results only
    a_by_id: dict[str, UnifiedCliqueResult] = {
        r.clique_id: r for r in results_a if r.is_valid
    }
    b_by_id: dict[str, UnifiedCliqueResult] = {
        r.clique_id: r for r in results_b if r.is_valid
    }

    # Find common cliques (intersection)
    common_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    n = len(common_ids)

    if n < 3:
        raise ValueError(
            f"Need at least 3 common cliques for concordance analysis, got {n}. "
            f"Method A has {len(a_by_id)} valid results, "
            f"Method B has {len(b_by_id)} valid results."
        )

    # Extract aligned vectors for common cliques
    p_a = np.array([a_by_id[cid].p_value for cid in common_ids])
    p_b = np.array([b_by_id[cid].p_value for cid in common_ids])
    eff_a = np.array([a_by_id[cid].effect_size for cid in common_ids])
    eff_b = np.array([b_by_id[cid].effect_size for cid in common_ids])

    # 1. Rank correlation of p-values (Spearman)
    # Spearman rho is robust to outliers and non-linear relationships
    rho, rho_pval = scipy_stats.spearmanr(p_a, p_b)

    # 2. Classification agreement at threshold
    sig_a = p_a < threshold
    sig_b = p_b < threshold

    n_both_sig = int(np.sum(sig_a & sig_b))
    n_both_nonsig = int(np.sum(~sig_a & ~sig_b))
    n_a_only = int(np.sum(sig_a & ~sig_b))
    n_b_only = int(np.sum(~sig_a & sig_b))

    # 3. Cohen's kappa
    # kappa = (p_o - p_e) / (1 - p_e)
    # where p_o = observed agreement, p_e = expected agreement by chance
    p_o = (n_both_sig + n_both_nonsig) / n  # Observed agreement rate

    # Expected agreement: probability that both agree by chance
    # P(both sig) = P(A sig) * P(B sig)
    # P(both nonsig) = P(A nonsig) * P(B nonsig)
    p_a_sig_rate = np.sum(sig_a) / n
    p_b_sig_rate = np.sum(sig_b) / n
    p_e = p_a_sig_rate * p_b_sig_rate + (1 - p_a_sig_rate) * (1 - p_b_sig_rate)

    if p_e < 1:
        kappa = (p_o - p_e) / (1 - p_e)
    else:
        # Perfect expected agreement - kappa is 1 if observed is also perfect
        kappa = 1.0 if p_o == 1.0 else 0.0

    # 4. Effect size agreement (Pearson correlation and RMSE)
    # Only use cliques with finite effect sizes in both methods
    valid_eff_mask = np.isfinite(eff_a) & np.isfinite(eff_b)
    n_valid_eff = int(np.sum(valid_eff_mask))

    if n_valid_eff >= 3:
        eff_r, _ = scipy_stats.pearsonr(eff_a[valid_eff_mask], eff_b[valid_eff_mask])
        eff_rmse = np.sqrt(np.mean((eff_a[valid_eff_mask] - eff_b[valid_eff_mask]) ** 2))
    else:
        eff_r = np.nan
        eff_rmse = np.nan

    # 5. Direction agreement (same sign of effect size)
    # Important for detecting consistent biological interpretation
    if n_valid_eff >= 3:
        # Compare signs: positive, negative, or zero
        # Use np.sign which returns -1, 0, or 1
        sign_a = np.sign(eff_a[valid_eff_mask])
        sign_b = np.sign(eff_b[valid_eff_mask])
        same_sign = sign_a == sign_b
        dir_agree = float(np.mean(same_sign))
    else:
        dir_agree = np.nan

    # Get method names from the first valid result in each list
    # (all results in a list should have the same method)
    method_a = results_a[0].method
    method_b = results_b[0].method

    return ConcordanceMetrics(
        method_a=method_a,
        method_b=method_b,
        n_cliques_compared=n,
        spearman_rho=float(rho) if np.isfinite(rho) else np.nan,
        spearman_pvalue=float(rho_pval) if np.isfinite(rho_pval) else np.nan,
        threshold=threshold,
        n_both_significant=n_both_sig,
        n_both_nonsignificant=n_both_nonsig,
        n_a_only=n_a_only,
        n_b_only=n_b_only,
        cohen_kappa=float(kappa) if np.isfinite(kappa) else np.nan,
        effect_pearson_r=float(eff_r) if np.isfinite(eff_r) else np.nan,
        effect_rmse=float(eff_rmse) if np.isfinite(eff_rmse) else np.nan,
        direction_agreement_frac=float(dir_agree) if np.isfinite(dir_agree) else np.nan,
    )


def identify_disagreements(
    results_by_method: dict[MethodName, list[UnifiedCliqueResult]],
    threshold: float = 0.05,
) -> "pd.DataFrame":
    """
    Identify cliques where statistical methods disagree on significance.

    A clique is flagged as a "disagreement case" if:
        - At least one method calls it significant (p < threshold)
        - At least one method calls it non-significant (p >= threshold)

    These disagreement cases are particularly interesting for biological
    investigation because they may reveal:
        - Method-specific sensitivities (e.g., ROAST detecting bidirectional
          regulation that OLS misses due to signal cancellation)
        - Edge cases near the significance boundary
        - Cliques where effect size interpretation differs across methods

    Args:
        results_by_method: Dictionary mapping MethodName to list of
            UnifiedCliqueResult for that method. All methods should have
            been run on the same set of cliques.
        threshold: P-value threshold for significance classification.
            Default is 0.05.

    Returns:
        pandas DataFrame with disagreement cases, containing columns:
            - clique_id: Clique identifier
            - {method}_pvalue: P-value from each method (e.g., ols_pvalue)
            - {method}_effect: Effect size from each method (e.g., ols_effect)
            - n_methods_significant: Count of methods calling clique significant
            - n_methods_nonsignificant: Count of methods calling clique non-significant
            - is_disagreement: True for all rows (DataFrame is filtered to disagreements)

        DataFrame is sorted by n_methods_significant descending (cliques where
        most methods agree on significance appear first).

    Example:
        >>> from cliquefinder.stats.method_comparison import identify_disagreements
        >>> disagreements = identify_disagreements(results_by_method, threshold=0.05)
        >>> print(f"Found {len(disagreements)} disagreement cases")
        >>> # Cliques significant in some but not all methods
        >>> for _, row in disagreements.head(5).iterrows():
        ...     print(f"{row['clique_id']}: {row['n_methods_significant']} sig, "
        ...           f"{row['n_methods_nonsignificant']} non-sig")

    Notes:
        - Only valid results (is_valid=True) are included
        - Cliques with NaN p-values in some methods are still included;
          those methods are counted as neither significant nor non-significant
        - Empty DataFrame returned if no disagreements found
    """
    import pandas as pd

    # Collect all unique clique IDs from all methods (valid results only)
    all_ids: set[str] = set()
    for results in results_by_method.values():
        all_ids.update(r.clique_id for r in results if r.is_valid)

    if not all_ids:
        # No valid results from any method
        return pd.DataFrame()

    # Build wide-format table: one row per clique
    rows: list[dict[str, object]] = []

    for cid in sorted(all_ids):
        row: dict[str, object] = {"clique_id": cid}
        sig_count = 0
        nonsig_count = 0

        for method, results in results_by_method.items():
            # Find result for this clique from this method
            result = next(
                (r for r in results if r.clique_id == cid and r.is_valid),
                None
            )

            method_key = method.value  # e.g., "ols", "roast_msq"

            if result is not None:
                row[f"{method_key}_pvalue"] = result.p_value
                row[f"{method_key}_effect"] = result.effect_size

                if result.p_value < threshold:
                    sig_count += 1
                else:
                    nonsig_count += 1
            else:
                # Method didn't produce a valid result for this clique
                row[f"{method_key}_pvalue"] = np.nan
                row[f"{method_key}_effect"] = np.nan

        row["n_methods_significant"] = sig_count
        row["n_methods_nonsignificant"] = nonsig_count

        # A disagreement occurs when some methods call it significant
        # and some call it non-significant
        row["is_disagreement"] = (sig_count > 0) and (nonsig_count > 0)

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Filter to only disagreement cases
    disagreements = df[df["is_disagreement"]].copy()

    # Sort by number of methods calling significant (descending)
    # Cliques that are significant in more methods are of higher interest
    disagreements = disagreements.sort_values(
        "n_methods_significant", ascending=False
    )

    # Reset index for clean output
    disagreements = disagreements.reset_index(drop=True)

    return disagreements


# =============================================================================
# MethodComparisonResult Dataclass
# =============================================================================


@dataclass
class MethodComparisonResult:
    """
    Complete comparison results across all differential testing methods.

    This is the main output of run_method_comparison(). It aggregates results
    from multiple statistical methods (OLS, LMM, ROAST, Permutation) and provides
    multiple views into the comparison for different analyses.

    This dataclass is NOT frozen because it contains mutable dict fields.
    However, it should be treated as read-only after creation.

    Design Principles:
        1. Raw results preserved for custom analysis
        2. Pre-computed concordance metrics for efficiency
        3. Helper methods for common queries (robust hits, method-specific hits)
        4. Full provenance for reproducibility

    Statistical Note:
        This framework is for DESCRIPTIVE comparison, not inference.
        - Do NOT select the "best" p-value per clique (would inflate FDR)
        - Do NOT combine p-values across methods (requires strong assumptions)
        - DO use concordance to validate findings (high agreement = robust)
        - DO investigate disagreements for biological insights

    Attributes:
        results_by_method: Raw results from each method, keyed by MethodName.
            Access individual method results with results_by_method[MethodName.OLS].
        pairwise_concordance: List of ConcordanceMetrics for all method pairs.
            Use concordance_matrix() for a convenient matrix view.
        mean_spearman_rho: Average Spearman correlation of p-value ranks across all pairs.
            Interpretation: >0.8 excellent, 0.6-0.8 good, <0.5 poor agreement.
        mean_cohen_kappa: Average Cohen's kappa (classification agreement) across all pairs.
            Interpretation: >0.6 substantial, 0.4-0.6 moderate, <0.2 slight agreement.
        disagreement_cases: DataFrame of cliques where methods disagree on significance.
            Useful for investigating method-specific sensitivities.
        preprocessing_params: Dict capturing normalization, imputation, etc.
            for full reproducibility.
        methods_run: List of MethodName values that were successfully executed.
        n_cliques_tested: Total number of cliques tested by at least one method.

    Example:
        >>> comparison = run_method_comparison(...)
        >>> print(comparison.summary())
        >>>
        >>> # Get robust discoveries (significant in all methods)
        >>> robust = comparison.robust_hits(threshold=0.01)
        >>> print(f"Robust hits: {len(robust)}")
        >>>
        >>> # Get ROAST-specific hits (bidirectional regulation?)
        >>> roast_only = comparison.method_specific_hits(MethodName.ROAST_MSQ)
        >>>
        >>> # Export to CSV for downstream analysis
        >>> wide_df = comparison.wide_format()
        >>> wide_df.to_csv("method_comparison_results.csv")

    See Also:
        - run_method_comparison: Main entry point that creates this object
        - ConcordanceMetrics: Detailed pairwise agreement metrics
        - UnifiedCliqueResult: Individual method results
    """

    # Raw results from each method
    results_by_method: dict[MethodName, list[UnifiedCliqueResult]]

    # Pairwise concordance metrics
    pairwise_concordance: list[ConcordanceMetrics]

    # Aggregate statistics
    mean_spearman_rho: float
    mean_cohen_kappa: float

    # Disagreement analysis
    disagreement_cases: object  # pd.DataFrame - using object for typing flexibility

    # Provenance
    preprocessing_params: dict[str, object]
    methods_run: list[MethodName]
    n_cliques_tested: int

    def wide_format(self) -> "pd.DataFrame":
        """
        Pivot results to wide format: one row per clique.

        Creates a DataFrame with one row per clique and columns for each method's
        statistics, enabling easy comparison, filtering, and export.

        Columns returned:
            - clique_id: Unique clique identifier
            - n_proteins: Number of proteins in clique definition
            - n_proteins_found: Number of proteins found in expression data
            - {method}_pvalue: P-value from method (e.g., ols_pvalue, roast_msq_pvalue)
            - {method}_effect_size: Effect size from method
            - {method}_statistic: Test statistic from method

        Cliques not tested by some methods will have NaN for those method columns.

        Returns:
            DataFrame with one row per unique clique across all methods.
            Rows are sorted alphabetically by clique_id.

        Example:
            >>> wide = comparison.wide_format()
            >>> # Filter to significant in at least one method
            >>> pval_cols = [c for c in wide.columns if c.endswith('_pvalue')]
            >>> sig_any = (wide[pval_cols] < 0.05).any(axis=1)
            >>> significant = wide[sig_any]
            >>> significant.to_csv("significant_cliques.csv", index=False)
        """
        import pandas as pd

        # Collect all unique clique IDs across all methods
        all_ids: set[str] = set()
        for results in self.results_by_method.values():
            all_ids.update(r.clique_id for r in results)

        if not all_ids:
            # No results from any method
            return pd.DataFrame(columns=["clique_id", "n_proteins", "n_proteins_found"])

        rows: list[dict[str, object]] = []

        for cid in sorted(all_ids):
            row: dict[str, object] = {"clique_id": cid}
            n_proteins: int | None = None
            n_proteins_found: int | None = None

            for method, results in self.results_by_method.items():
                # Find result for this clique from this method
                result = next((r for r in results if r.clique_id == cid), None)
                prefix = method.value

                if result is not None:
                    row[f"{prefix}_pvalue"] = result.p_value
                    row[f"{prefix}_effect_size"] = result.effect_size
                    row[f"{prefix}_statistic"] = result.statistic_value
                    # Capture clique size from first method that has it
                    if n_proteins is None:
                        n_proteins = result.n_proteins
                        n_proteins_found = result.n_proteins_found
                else:
                    row[f"{prefix}_pvalue"] = np.nan
                    row[f"{prefix}_effect_size"] = np.nan
                    row[f"{prefix}_statistic"] = np.nan

            # Add clique size columns
            row["n_proteins"] = n_proteins
            row["n_proteins_found"] = n_proteins_found

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Reorder columns for clarity: clique_id, n_proteins, n_proteins_found first
        base_cols = ["clique_id", "n_proteins", "n_proteins_found"]
        other_cols = sorted([c for c in df.columns if c not in base_cols])
        df = df[base_cols + other_cols]

        return df

    def concordance_matrix(self) -> "pd.DataFrame":
        """
        Create square matrix of pairwise Spearman correlations.

        Returns a symmetric matrix suitable for heatmap visualization where
        each cell (i,j) contains the Spearman rho between methods i and j.
        Diagonal values are 1.0 (self-correlation).

        Returns:
            DataFrame with method names (e.g., "ols", "roast_msq") as both
            index and columns. Values are Spearman rho in range [-1, 1].

        Example:
            >>> import seaborn as sns
            >>> import matplotlib.pyplot as plt
            >>> matrix = comparison.concordance_matrix()
            >>> fig, ax = plt.subplots(figsize=(8, 6))
            >>> sns.heatmap(
            ...     matrix, annot=True, fmt='.2f',
            ...     cmap='RdYlGn', vmin=-1, vmax=1, ax=ax
            ... )
            >>> ax.set_title("Method Concordance (Spearman rho)")
            >>> plt.tight_layout()
        """
        import pandas as pd

        methods = [m.value for m in self.methods_run]
        n = len(methods)

        if n == 0:
            return pd.DataFrame()

        # Initialize with identity matrix (diagonal = 1.0, perfect self-correlation)
        matrix = np.eye(n)

        # Fill in off-diagonal elements from pairwise concordance
        for conc in self.pairwise_concordance:
            try:
                i = methods.index(conc.method_a.value)
                j = methods.index(conc.method_b.value)
                matrix[i, j] = conc.spearman_rho
                matrix[j, i] = conc.spearman_rho  # Symmetric matrix
            except ValueError:
                # Method not in methods_run (shouldn't happen, but be defensive)
                continue

        return pd.DataFrame(matrix, index=methods, columns=methods)

    def robust_hits(self, threshold: float = 0.05) -> list[str]:
        """
        Get cliques significant in ALL methods.

        These are high-confidence discoveries that replicate across different
        statistical approaches. A clique is a "robust hit" only if its p-value
        is below the threshold in EVERY method that tested it.

        This is a conservative criterion that minimizes false positives at the
        cost of potentially missing some true positives.

        Args:
            threshold: P-value threshold for significance (default 0.05).
                Use stricter thresholds (0.01, 0.001) for higher confidence.

        Returns:
            List of clique_id strings for cliques significant in all methods.
            Returns empty list if no cliques meet the criterion or if no
            methods were run.

        Example:
            >>> # Check robust hits at multiple thresholds
            >>> for thresh in [0.05, 0.01, 0.001]:
            ...     robust = comparison.robust_hits(threshold=thresh)
            ...     print(f"p < {thresh}: {len(robust)} robust hits")
            p < 0.05: 42 robust hits
            p < 0.01: 18 robust hits
            p < 0.001: 5 robust hits
        """
        if not self.methods_run:
            return []

        wide = self.wide_format()
        pval_cols = [c for c in wide.columns if c.endswith("_pvalue")]

        if not pval_cols:
            return []

        # A clique is a robust hit if ALL p-values are below threshold
        # Note: This requires the clique to be tested by ALL methods
        # NaN values mean method didn't test it, so we use dropna behavior
        mask = (wide[pval_cols] < threshold).all(axis=1)

        return wide.loc[mask, "clique_id"].tolist()

    def method_specific_hits(
        self, method: MethodName, threshold: float = 0.05
    ) -> list[str]:
        """
        Get cliques significant ONLY in the specified method.

        These hits are detected by one method but not others, which may indicate:
            - Method-specific sensitivity (e.g., ROAST detecting bidirectional
              regulation that OLS misses due to signal cancellation)
            - Potential false positives in that method
            - Unique biological signal that other methods are insensitive to

        Investigating these cases helps understand method behavior and can
        reveal which method is most appropriate for specific biological questions.

        Args:
            method: MethodName to check for unique significance.
            threshold: P-value threshold for significance (default 0.05).

        Returns:
            List of clique_id strings that are:
                - Significant (p < threshold) in the specified method
                - NOT significant (p >= threshold OR NaN) in all other methods
            Returns empty list if the specified method wasn't run or has no results.

        Example:
            >>> # Find cliques only ROAST detects (bidirectional regulation?)
            >>> roast_only = comparison.method_specific_hits(MethodName.ROAST_MSQ)
            >>> print(f"ROAST-specific hits: {len(roast_only)}")
            >>>
            >>> # Compare method-specific counts
            >>> for m in comparison.methods_run:
            ...     specific = comparison.method_specific_hits(m)
            ...     print(f"{m.value}: {len(specific)} unique hits")
        """
        if method not in self.results_by_method:
            return []

        wide = self.wide_format()
        method_col = f"{method.value}_pvalue"

        if method_col not in wide.columns:
            return []

        other_cols = [
            c for c in wide.columns if c.endswith("_pvalue") and c != method_col
        ]

        # Significant in this method (p < threshold)
        sig_in_method = wide[method_col] < threshold

        # Not significant in ANY other method
        # NaN values are treated as "not significant" (method didn't test it)
        not_sig_elsewhere = True
        for col in other_cols:
            # p >= threshold OR p is NaN counts as "not significant"
            col_not_sig = (wide[col] >= threshold) | wide[col].isna()
            not_sig_elsewhere = not_sig_elsewhere & col_not_sig

        mask = sig_in_method & not_sig_elsewhere
        return wide.loc[mask, "clique_id"].tolist()

    def get_concordance(
        self, method_a: MethodName, method_b: MethodName
    ) -> ConcordanceMetrics | None:
        """
        Retrieve concordance metrics for a specific pair of methods.

        Looks up the pre-computed ConcordanceMetrics for the given method pair.
        Order of arguments doesn't matter (will find either direction).

        Args:
            method_a: First method in the pair.
            method_b: Second method in the pair.

        Returns:
            ConcordanceMetrics for the specified pair, containing Spearman rho,
            Cohen's kappa, effect correlation, etc.
            Returns None if the pair wasn't computed (e.g., method not run or
            insufficient common cliques).

        Example:
            >>> conc = comparison.get_concordance(MethodName.OLS, MethodName.ROAST_MSQ)
            >>> if conc:
            ...     print(f"OLS vs ROAST:")
            ...     print(f"  Spearman rho: {conc.spearman_rho:.3f}")
            ...     print(f"  Cohen's kappa: {conc.cohen_kappa:.3f}")
            ...     print(f"  Jaccard index: {conc.jaccard_index:.3f}")
        """
        for conc in self.pairwise_concordance:
            # Check both orderings since pairs are stored in one direction only
            if (conc.method_a == method_a and conc.method_b == method_b) or (
                conc.method_a == method_b and conc.method_b == method_a
            ):
                return conc
        return None

    def summary(self) -> str:
        """
        Generate human-readable summary of comparison results.

        Provides a multi-line text summary including:
            - Number of cliques tested
            - Methods run
            - Mean concordance metrics
            - Robust hit counts at multiple thresholds
            - Number of disagreement cases
            - Pairwise concordance breakdown

        Returns:
            Multi-line string suitable for printing, logging, or saving to file.

        Example:
            >>> print(comparison.summary())
            Method Comparison Results
            ========================================
            Cliques tested: 250
            Methods: ols, roast_msq, permutation_competitive

            Concordance Summary:
              Mean Spearman rho: 0.782
              Mean Cohen's kappa: 0.654

            Robust hits (significant in all methods):
              p < 0.05: 42
              p < 0.01: 18
              p < 0.001: 5

            Disagreement cases: 35

            Pairwise Concordance:
              ols vs roast_msq: rho=0.823, kappa=0.712
              ols vs permutation_competitive: rho=0.756, kappa=0.623
              roast_msq vs permutation_competitive: rho=0.768, kappa=0.627
        """
        lines = [
            "Method Comparison Results",
            "=" * 40,
            f"Cliques tested: {self.n_cliques_tested}",
            f"Methods: {', '.join(m.value for m in self.methods_run)}",
            "",
            "Concordance Summary:",
            f"  Mean Spearman rho: {self.mean_spearman_rho:.3f}",
            f"  Mean Cohen's kappa: {self.mean_cohen_kappa:.3f}",
            "",
            "Robust hits (significant in all methods):",
        ]

        # Show robust hits at standard thresholds
        for thresh in [0.05, 0.01, 0.001]:
            n_robust = len(self.robust_hits(thresh))
            lines.append(f"  p < {thresh}: {n_robust}")

        # Add disagreement count
        import pandas as pd

        if isinstance(self.disagreement_cases, pd.DataFrame):
            n_disagreements = len(self.disagreement_cases)
        else:
            n_disagreements = 0

        lines.extend([
            "",
            f"Disagreement cases: {n_disagreements}",
        ])

        # Add pairwise concordance details if available
        if self.pairwise_concordance:
            lines.extend([
                "",
                "Pairwise Concordance:",
            ])
            for conc in self.pairwise_concordance:
                lines.append(
                    f"  {conc.method_a.value} vs {conc.method_b.value}: "
                    f"rho={conc.spearman_rho:.3f}, kappa={conc.cohen_kappa:.3f}"
                )

        return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================


def run_method_comparison(
    data: np.ndarray,
    feature_ids: list[str],
    sample_metadata: "pd.DataFrame",
    cliques: list[object],  # list[CliqueDefinition]
    condition_column: str,
    contrast: tuple[str, str],
    subject_column: str | None = None,
    methods: list[CliqueTestMethod] | None = None,
    concordance_threshold: float = 0.05,
    normalization_method: str = "median",
    imputation_method: str = "min_feature",
    n_rotations: int = 9999,
    n_permutations: int = 10000,
    use_gpu: bool = True,
    seed: int | None = None,
    verbose: bool = True,
    precomputed_symbol_map: dict[str, str] | None = None,
) -> MethodComparisonResult:
    """
    Run multiple differential testing methods and compute concordance.

    This is the main entry point for method comparison. Pipeline:
        1. Prepare data ONCE (shared preprocessing)
        2. Run each method on the prepared data
        3. Compute pairwise concordance metrics
        4. Identify disagreement cases
        5. Return structured comparison results

    Statistical Note:
        Results are DESCRIPTIVE. Do NOT:
        - Select the "best" p-value per clique (inflates FDR)
        - Combine p-values (requires careful assumptions)
        - Declare a "winner" method

        Instead, use concordance to:
        - Validate findings (high agreement = robust)
        - Understand method behavior
        - Flag interesting disagreements

    Args:
        data: Expression matrix (n_features, n_samples) of log2 intensities.
        feature_ids: Gene/protein identifiers matching data rows.
        sample_metadata: DataFrame with sample annotations.
        cliques: List of CliqueDefinition objects to test.
        condition_column: Metadata column name for condition labels.
        contrast: Tuple of (test_condition, reference_condition).
        subject_column: Optional metadata column for biological replicates.
            If provided, LMMMethod will be included in the default methods.
        methods: List of CliqueTestMethod instances to run.
            Default is [OLSMethod(), LMMMethod() if subject_column, ROASTMethod(msq),
            ROASTMethod(mean), PermutationMethod()].
        concordance_threshold: P-value threshold for classification agreement
            analysis (default 0.05).
        normalization_method: Preprocessing normalization (default "median").
            Options: "none", "median", "quantile", "global_standards", "vsn".
        imputation_method: Preprocessing imputation (default "min_feature").
            Options: "none", "min_feature", "min_global", "min_sample", "knn".
        n_rotations: Number of rotations for ROAST method (default 9999).
        n_permutations: Number of permutations for permutation method (default 10000).
        use_gpu: Whether to use GPU acceleration (default True). Falls back to
            CPU if GPU is not available.
        seed: Random seed for reproducibility. Applies to ROAST and permutation
            methods.
        verbose: Print progress messages (default True).

    Returns:
        MethodComparisonResult with:
            - Per-method results (results_by_method)
            - Pairwise concordance metrics (pairwise_concordance)
            - Mean concordance statistics (mean_spearman_rho, mean_cohen_kappa)
            - Disagreement analysis (disagreement_cases DataFrame)
            - Wide-format combined table (via wide_format() method)

    Raises:
        ValueError: If contrast conditions not found in data.
        ValueError: If no cliques have proteins in data.

    Example:
        >>> from cliquefinder.stats import run_method_comparison, MethodName
        >>> comparison = run_method_comparison(
        ...     data=expression_matrix,
        ...     feature_ids=protein_ids,
        ...     sample_metadata=metadata,
        ...     cliques=clique_definitions,
        ...     condition_column='phenotype',
        ...     contrast=('ALS', 'Control'),
        ...     subject_column='subject_id',
        ... )
        >>>
        >>> # Check overall agreement
        >>> print(f"Mean Spearman rho: {comparison.mean_spearman_rho:.3f}")
        >>>
        >>> # Get robust hits (significant in all methods)
        >>> robust = comparison.robust_hits(threshold=0.05)
        >>>
        >>> # Get ROAST-specific hits (bidirectional regulation?)
        >>> roast_only = comparison.method_specific_hits(MethodName.ROAST_MSQ)
        >>>
        >>> # Export wide format for downstream analysis
        >>> wide_df = comparison.wide_format()
        >>> wide_df.to_csv("method_comparison.csv")

    See Also:
        - prepare_experiment: Data preprocessing function
        - MethodComparisonResult: Output structure with helper methods
        - ConcordanceMetrics: Pairwise agreement metrics
        - UnifiedCliqueResult: Individual method results
    """
    import pandas as pd

    # 1. Print header (if verbose)
    if verbose:
        print("=" * 60)
        print("METHOD COMPARISON FRAMEWORK")
        print("=" * 60)
        print()

    # 2. Prepare experiment using prepare_experiment()
    # This ensures all methods get the same preprocessed data
    experiment = prepare_experiment(
        data=data,
        feature_ids=feature_ids,
        sample_metadata=sample_metadata,
        cliques=cliques,
        condition_column=condition_column,
        contrast=contrast,
        subject_column=subject_column,
        normalization_method=normalization_method,
        imputation_method=imputation_method,
        verbose=verbose,
        precomputed_symbol_map=precomputed_symbol_map,
    )

    # 3. Default methods if not specified
    if methods is None:
        methods = [
            OLSMethod(),
        ]
        # Add LMMMethod if subject_column is set
        if subject_column is not None:
            methods.append(LMMMethod())
        # Add ROAST methods with different statistics
        methods.extend([
            ROASTMethod(statistic="msq", n_rotations=n_rotations),
            ROASTMethod(statistic="mean", n_rotations=n_rotations),
        ])
        # Add permutation method
        methods.append(PermutationMethod(n_permutations=n_permutations))

    if verbose:
        print()
        print(f"Methods to run: {[m.name.value for m in methods]}")
        print()

    # 4. Run each method
    results_by_method: dict[MethodName, list[UnifiedCliqueResult]] = {}

    for method in methods:
        if verbose:
            print(f"Running {method.name.value}...", end=" ", flush=True)

        try:
            results = method.test(
                experiment,
                use_gpu=use_gpu,
                seed=seed,
                verbose=False,  # Method-level verbosity off
            )
            results_by_method[method.name] = results

            if verbose:
                n_sig = sum(1 for r in results if r.p_value < concordance_threshold)
                print(f"done ({len(results)} cliques, {n_sig} significant)")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results_by_method[method.name] = []

    # 5. Compute pairwise concordance
    if verbose:
        print()
        print("Computing concordance metrics...")

    # Get method names with non-empty results
    method_names = [
        m for m in results_by_method.keys()
        if len(results_by_method[m]) > 0
    ]
    pairwise: list[ConcordanceMetrics] = []

    for i, name_a in enumerate(method_names):
        for name_b in method_names[i + 1:]:
            try:
                conc = compute_pairwise_concordance(
                    results_by_method[name_a],
                    results_by_method[name_b],
                    threshold=concordance_threshold,
                )
                pairwise.append(conc)
            except ValueError as e:
                if verbose:
                    print(f"  Skipping {name_a.value} vs {name_b.value}: {e}")

    # 6. Aggregate metrics
    if pairwise:
        mean_rho = float(np.mean([c.spearman_rho for c in pairwise]))
        mean_kappa = float(np.mean([c.cohen_kappa for c in pairwise]))
    else:
        mean_rho = np.nan
        mean_kappa = np.nan

    # 7. Identify disagreements
    disagreements = identify_disagreements(
        results_by_method,
        threshold=concordance_threshold,
    )

    # 8. Count cliques tested (union of all clique_ids across all methods)
    all_tested: set[str] = set()
    for results in results_by_method.values():
        all_tested.update(r.clique_id for r in results if r.is_valid)

    # 9. Print summary (if verbose)
    if verbose:
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Methods compared: {len(method_names)}")
        print(f"Cliques tested: {len(all_tested)}")
        print(f"Mean Spearman rho: {mean_rho:.3f}")
        print(f"Mean Cohen's kappa: {mean_kappa:.3f}")

        if isinstance(disagreements, pd.DataFrame):
            print(f"Disagreement cases: {len(disagreements)}")
        else:
            print("Disagreement cases: 0")

        print()

        # Quick concordance table
        if pairwise:
            print("Pairwise concordance (Spearman rho):")
            for conc in pairwise:
                print(
                    f"  {conc.method_a.value} vs {conc.method_b.value}: "
                    f"rho={conc.spearman_rho:.3f}, kappa={conc.cohen_kappa:.3f}"
                )

    # 10. Return MethodComparisonResult with all fields populated
    return MethodComparisonResult(
        results_by_method=results_by_method,
        pairwise_concordance=pairwise,
        mean_spearman_rho=mean_rho,
        mean_cohen_kappa=mean_kappa,
        disagreement_cases=disagreements,
        preprocessing_params=experiment.preprocessing_params,
        methods_run=list(method_names),
        n_cliques_tested=len(all_tested),
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "MethodName",
    # Core dataclasses
    "UnifiedCliqueResult",
    "PreparedCliqueExperiment",
    "ConcordanceMetrics",
    "MethodComparisonResult",
    # Protocol
    "CliqueTestMethod",
    # Factory function
    "prepare_experiment",
    # Method adapters
    "OLSMethod",
    "LMMMethod",
    "ROASTMethod",
    "PermutationMethod",
    # Concordance functions
    "compute_pairwise_concordance",
    "identify_disagreements",
    # Main entry point
    "run_method_comparison",
]
