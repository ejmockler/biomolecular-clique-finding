"""
Core types for the Method Comparison Framework.

This module defines the foundational types used across the method comparison
framework: enums for method identification, dataclasses for results and
concordance metrics, and the protocol interface for pluggable methods.

These types are re-exported from ``method_comparison`` for backward
compatibility -- prefer importing from there in application code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    pass


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
        ...         experiment: "PreparedCliqueExperiment",
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
        experiment: object,  # PreparedCliqueExperiment
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


__all__ = [
    "MethodName",
    "UnifiedCliqueResult",
    "ConcordanceMetrics",
    "CliqueTestMethod",
]
