"""
Generalized Permutation Testing Framework for Biomolecular Data.

Engineering Design Principles:
1. Protocol-based abstractions that work across any -omics modality
2. Metadata-aware experimental design specification
3. Pluggable statistical tests and summarization methods
4. Composable null distribution strategies

This framework enables:
- Proteomics, transcriptomics, metabolomics, or any feature × sample data
- Any categorical contrast (C9 vs Sporadic, Treatment vs Control, etc.)
- Any gene set definition (TF targets, pathways, custom modules)
- Any summarization method (TMP, median, PCA, etc.)
- Any statistical test (t-test, mixed model, rank-based, etc.)

References:
    - GSEA: Subramanian et al., PNAS 2005
    - Camera: Wu & Smyth, NAR 2012
    - Competitive vs self-contained tests: Goeman & Bühlmann, Briefings 2007
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Generic,
    Iterator,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# =============================================================================
# Core Protocols - The Engineering Foundation
# =============================================================================

@runtime_checkable
class FeatureSet(Protocol):
    """Protocol for any collection of molecular features (genes, proteins, metabolites).

    This abstraction works for:
    - Protein cliques (our current use case)
    - TF target gene sets
    - Pathway gene sets (KEGG, Reactome)
    - Custom modules
    - Single features (for matched comparison)
    """

    @property
    def id(self) -> str:
        """Unique identifier for this feature set."""
        ...

    @property
    def feature_ids(self) -> list[str]:
        """List of feature identifiers (gene symbols, protein IDs, etc.)."""
        ...

    @property
    def size(self) -> int:
        """Number of features in this set."""
        ...

    def metadata(self) -> dict:
        """Optional metadata about this feature set."""
        ...


@runtime_checkable
class ExperimentalDesign(Protocol):
    """Protocol specifying an experimental comparison.

    This abstraction works for:
    - CASE vs CTRL (current ALS analysis)
    - C9orf72 vs Sporadic ALS (collaborator's request)
    - Treatment vs Vehicle
    - Timepoint comparisons
    - Multi-factor designs
    """

    @property
    def condition_column(self) -> str:
        """Metadata column containing condition labels."""
        ...

    @property
    def test_condition(self) -> str:
        """Label for the test/treatment condition."""
        ...

    @property
    def reference_condition(self) -> str:
        """Label for the reference/control condition."""
        ...

    @property
    def blocking_column(self) -> str | None:
        """Optional column for blocking/subject effects (for mixed models)."""
        ...

    def sample_mask(self, metadata: pd.DataFrame) -> NDArray[np.bool_]:
        """Return boolean mask for samples to include in this comparison."""
        ...


@runtime_checkable
class Summarizer(Protocol):
    """Protocol for aggregating features to a single value per sample.

    This abstraction works for:
    - Tukey's Median Polish (our default)
    - Simple median/mean
    - PCA first component
    - Weighted average by quality scores
    """

    def summarize(
        self,
        data: NDArray[np.float64],
        feature_ids: list[str],
    ) -> NDArray[np.float64]:
        """Aggregate feature × sample matrix to 1D sample vector."""
        ...


@runtime_checkable
class StatisticalTest(Protocol):
    """Protocol for differential abundance testing.

    This abstraction works for:
    - MSstats-style linear mixed models
    - Simple t-test
    - Wilcoxon rank-sum
    - Limma moderated t-test
    - DESeq2-style negative binomial
    """

    def test(
        self,
        values: NDArray[np.float64],
        design: ExperimentalDesign,
        metadata: pd.DataFrame,
    ) -> TestResult:
        """Run differential test, return test result."""
        ...


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class TestResult:
    """Result from a statistical test.

    Generic enough for any test type while capturing key statistics
    needed for permutation comparison.
    """

    feature_set_id: str
    effect_size: float  # log2FC, mean difference, etc.
    test_statistic: float  # t, z, W, etc.
    p_value: float
    n_test: int  # samples in test condition
    n_reference: int  # samples in reference condition
    additional: dict = field(default_factory=dict)  # method-specific extras


@dataclass
class PermutationResult:
    """Result of permutation-based significance testing."""

    feature_set_id: str
    observed: TestResult
    null_distribution: NDArray[np.float64]  # test statistics from permutations
    empirical_pvalue: float
    empirical_pvalue_onesided: float
    percentile_rank: float  # 0-100, where observed falls in null
    n_permutations: int
    is_significant: bool = False

    @property
    def zscore(self) -> float:
        """Standardized score: (observed - null_mean) / null_std."""
        if len(self.null_distribution) < 2:
            return np.nan
        null_std = np.std(self.null_distribution)
        if null_std < 1e-10:
            return np.nan
        return (self.observed.test_statistic - np.mean(self.null_distribution)) / null_std

    def to_dict(self) -> dict:
        return {
            'feature_set_id': self.feature_set_id,
            'observed_effect': self.observed.effect_size,
            'observed_statistic': self.observed.test_statistic,
            'observed_pvalue': self.observed.p_value,
            'null_mean': float(np.mean(self.null_distribution)),
            'null_std': float(np.std(self.null_distribution)),
            'zscore': self.zscore,
            'empirical_pvalue': self.empirical_pvalue,
            'empirical_pvalue_onesided': self.empirical_pvalue_onesided,
            'percentile_rank': self.percentile_rank,
            'n_permutations': self.n_permutations,
            'is_significant': self.is_significant,
        }


# =============================================================================
# Concrete Implementations
# =============================================================================

@dataclass
class SimpleFeatureSet:
    """Basic feature set implementation."""

    _id: str
    _feature_ids: list[str]
    _metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self._id

    @property
    def feature_ids(self) -> list[str]:
        return self._feature_ids

    @property
    def size(self) -> int:
        return len(self._feature_ids)

    def metadata(self) -> dict:
        return self._metadata


@dataclass
class TwoGroupDesign:
    """Simple two-group experimental design.

    Examples:
        # CASE vs CTRL (current analysis)
        design = TwoGroupDesign(
            condition_column="phenotype",
            test_condition="CASE",
            reference_condition="CTRL",
            blocking_column="subject_id",
        )

        # C9 vs Sporadic (collaborator's request)
        design = TwoGroupDesign(
            condition_column="genetic_subtype",  # derived column
            test_condition="C9orf72",
            reference_condition="Sporadic",
            blocking_column="subject_id",
            sample_filter=lambda df: df['phenotype'] == 'CASE',  # ALS only
        )
    """

    condition_column: str
    test_condition: str
    reference_condition: str
    blocking_column: str | None = None
    sample_filter: Callable[[pd.DataFrame], NDArray[np.bool_]] | None = None

    def sample_mask(self, metadata: pd.DataFrame) -> NDArray[np.bool_]:
        """Return mask for samples in either condition."""
        in_comparison = metadata[self.condition_column].isin([
            self.test_condition, self.reference_condition
        ]).values

        if self.sample_filter is not None:
            in_comparison = in_comparison & self.sample_filter(metadata)

        return in_comparison


@dataclass
class MetadataDerivedDesign:
    """Experimental design derived from multiple metadata columns.

    Enables complex comparisons like C9 vs Sporadic ALS by computing
    condition labels from existing metadata.

    Example:
        # C9orf72 vs Sporadic ALS comparison
        def derive_genetic_subtype(row):
            if 'C9orf72' in str(row.get('ClinReport_Mutations_Details', '')):
                return 'C9orf72'
            elif row.get('phenotype') == 'CASE' and pd.isna(row.get('ClinReport_Mutations_Details')):
                return 'Sporadic'
            else:
                return None  # Exclude

        design = MetadataDerivedDesign(
            derivation_fn=derive_genetic_subtype,
            test_condition="C9orf72",
            reference_condition="Sporadic",
            blocking_column="subject_id",
        )
    """

    derivation_fn: Callable[[pd.Series], str | None]
    test_condition: str
    reference_condition: str
    blocking_column: str | None = None
    _derived_column_name: str = "_derived_condition"

    @property
    def condition_column(self) -> str:
        return self._derived_column_name

    def derive_conditions(self, metadata: pd.DataFrame) -> pd.Series:
        """Apply derivation function to create condition labels."""
        return metadata.apply(self.derivation_fn, axis=1)

    def sample_mask(self, metadata: pd.DataFrame) -> NDArray[np.bool_]:
        """Return mask for samples with valid derived conditions."""
        derived = self.derive_conditions(metadata)
        return derived.isin([self.test_condition, self.reference_condition]).values


# =============================================================================
# Null Distribution Strategies
# =============================================================================

class NullStrategy(Enum):
    """Strategies for generating null distributions."""

    RANDOM_GENE_SETS = "random_gene_sets"  # Competitive: random sets from pool
    SAMPLE_PERMUTATION = "sample_permutation"  # Self-contained: permute labels
    ROTATION = "rotation"  # Camera-style rotation for correlated features


@dataclass
class CompetitiveNullGenerator:
    """Generate null by sampling random feature sets from a pool.

    This is the collaborator's suggested approach:
    - Pool: all regulated genes (union of all TF targets)
    - Null set: random sample of same size as test set
    - Preserves correlation structure
    - Tests: "Is this set special vs random sets?"
    """

    feature_pool: list[str]
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def generate_null_set(self, size: int) -> list[str]:
        """Sample random features from pool."""
        if size >= len(self.feature_pool):
            return self.feature_pool.copy()
        return list(self.rng.choice(self.feature_pool, size=size, replace=False))


@dataclass
class SamplePermutationGenerator:
    """Generate null by permuting sample condition labels.

    Traditional self-contained test:
    - Permute which samples are "test" vs "reference"
    - Tests: "Is there any effect vs label shuffling?"
    """

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def permute_labels(self, condition_labels: NDArray) -> NDArray:
        """Return permuted condition labels."""
        return self.rng.permutation(condition_labels)


# =============================================================================
# The Permutation Test Engine
# =============================================================================

class PermutationTestEngine:
    """
    Generalized permutation testing engine.

    Combines:
    - Feature sets (what to test)
    - Experimental design (what comparison)
    - Summarization method (how to aggregate)
    - Statistical test (what test)
    - Null strategy (how to build null distribution)

    This design allows any combination of the above, enabling:
    - Clique vs random gene sets (current implementation)
    - Pathway enrichment with rotation
    - Single-gene permutation tests
    - Custom aggregation strategies
    """

    def __init__(
        self,
        data: NDArray[np.float64],
        feature_ids: list[str],
        metadata: pd.DataFrame,
        summarizer: Summarizer,
        test: StatisticalTest,
    ):
        self.data = data
        self.feature_ids = feature_ids
        self.feature_to_idx = {f: i for i, f in enumerate(feature_ids)}
        self.metadata = metadata
        self.summarizer = summarizer
        self.test = test

    def run_competitive_test(
        self,
        feature_sets: list[FeatureSet],
        design: ExperimentalDesign,
        feature_pool: list[str],
        n_permutations: int = 1000,
        significance_threshold: float = 0.05,
        seed: int | None = None,
        verbose: bool = True,
    ) -> list[PermutationResult]:
        """
        Run competitive permutation test.

        For each feature set:
        1. Compute observed test statistic
        2. Generate null by testing random sets of same size
        3. Compute empirical p-value

        Args:
            feature_sets: Feature sets to test
            design: Experimental design specification
            feature_pool: Pool of features for null generation
            n_permutations: Number of permutations
            significance_threshold: Threshold for significance
            seed: Random seed
            verbose: Print progress

        Returns:
            List of PermutationResult for each feature set
        """
        rng = np.random.default_rng(seed)
        null_gen = CompetitiveNullGenerator(feature_pool, rng)

        # Prepare metadata with derived conditions if needed
        if isinstance(design, MetadataDerivedDesign):
            working_metadata = self.metadata.copy()
            working_metadata[design.condition_column] = design.derive_conditions(self.metadata)
        else:
            working_metadata = self.metadata

        results = []

        for i, fset in enumerate(feature_sets):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Testing feature set {i + 1}/{len(feature_sets)}")

            # Get observed result
            observed = self._test_feature_set(fset, design, working_metadata)
            if observed is None:
                continue

            # Generate null distribution
            null_stats = []
            for _ in range(n_permutations):
                null_features = null_gen.generate_null_set(fset.size)
                null_set = SimpleFeatureSet(
                    _id=f"null_{fset.id}",
                    _feature_ids=null_features,
                )
                null_result = self._test_feature_set(null_set, design, working_metadata)
                if null_result is not None:
                    null_stats.append(null_result.test_statistic)

            if len(null_stats) < 10:
                continue

            null_array = np.array(null_stats)

            # Compute empirical p-values
            obs_t = observed.test_statistic
            n_extreme_two = np.sum(np.abs(null_array) >= np.abs(obs_t))
            empirical_p_two = (n_extreme_two + 1) / (len(null_stats) + 1)

            if obs_t > 0:
                n_extreme_one = np.sum(null_array >= obs_t)
            else:
                n_extreme_one = np.sum(null_array <= obs_t)
            empirical_p_one = (n_extreme_one + 1) / (len(null_stats) + 1)

            percentile = 100 * np.mean(np.abs(null_array) < np.abs(obs_t))

            results.append(PermutationResult(
                feature_set_id=fset.id,
                observed=observed,
                null_distribution=null_array,
                empirical_pvalue=empirical_p_two,
                empirical_pvalue_onesided=empirical_p_one,
                percentile_rank=percentile,
                n_permutations=len(null_stats),
                is_significant=empirical_p_two < significance_threshold,
            ))

        return results

    def _test_feature_set(
        self,
        fset: FeatureSet,
        design: ExperimentalDesign,
        metadata: pd.DataFrame,
    ) -> TestResult | None:
        """Test a single feature set."""
        # Get feature indices
        indices = []
        valid_features = []
        for f in fset.feature_ids:
            if f in self.feature_to_idx:
                indices.append(self.feature_to_idx[f])
                valid_features.append(f)

        if len(indices) < 2:
            return None

        # Extract and summarize
        feature_data = self.data[indices, :]
        summarized = self.summarizer.summarize(feature_data, valid_features)

        # Run test
        try:
            return self.test.test(summarized, design, metadata)
        except Exception:
            return None


# =============================================================================
# Pre-built Summarizers
# =============================================================================

@dataclass
class MedianPolishSummarizer:
    """Tukey's Median Polish summarization."""

    max_iter: int = 10
    eps: float = 0.01

    def summarize(
        self,
        data: NDArray[np.float64],
        feature_ids: list[str],
    ) -> NDArray[np.float64]:
        from .summarization import tukey_median_polish
        result = tukey_median_polish(data, max_iter=self.max_iter, eps=self.eps)
        return result.sample_abundances


@dataclass
class SimpleSummarizer:
    """Simple aggregation (median, mean, etc.)."""

    method: Literal["median", "mean"] = "median"

    def summarize(
        self,
        data: NDArray[np.float64],
        feature_ids: list[str],
    ) -> NDArray[np.float64]:
        if self.method == "median":
            return np.nanmedian(data, axis=0)
        else:
            return np.nanmean(data, axis=0)


# =============================================================================
# Pre-built Statistical Tests
# =============================================================================

@dataclass
class TTestStatistic:
    """Simple two-sample t-test."""

    def test(
        self,
        values: NDArray[np.float64],
        design: ExperimentalDesign,
        metadata: pd.DataFrame,
    ) -> TestResult:
        from scipy.stats import ttest_ind

        # Get condition labels
        if isinstance(design, MetadataDerivedDesign):
            conditions = design.derive_conditions(metadata).values
        else:
            conditions = metadata[design.condition_column].values

        test_mask = conditions == design.test_condition
        ref_mask = conditions == design.reference_condition

        test_vals = values[test_mask]
        ref_vals = values[ref_mask]

        # Remove NaNs
        test_vals = test_vals[~np.isnan(test_vals)]
        ref_vals = ref_vals[~np.isnan(ref_vals)]

        if len(test_vals) < 3 or len(ref_vals) < 3:
            raise ValueError("Insufficient samples")

        t_stat, p_val = ttest_ind(test_vals, ref_vals)
        effect = np.mean(test_vals) - np.mean(ref_vals)

        return TestResult(
            feature_set_id="",  # Filled by caller
            effect_size=effect,
            test_statistic=t_stat,
            p_value=p_val,
            n_test=len(test_vals),
            n_reference=len(ref_vals),
        )


@dataclass
class MixedModelStatistic:
    """Mixed model with subject random effect."""

    def test(
        self,
        values: NDArray[np.float64],
        design: ExperimentalDesign,
        metadata: pd.DataFrame,
    ) -> TestResult:
        from .differential import fit_linear_model

        # Get condition and subject
        if isinstance(design, MetadataDerivedDesign):
            conditions = design.derive_conditions(metadata).values
        else:
            conditions = metadata[design.condition_column].values

        subjects = None
        if design.blocking_column and design.blocking_column in metadata.columns:
            subjects = metadata[design.blocking_column].values

        # Filter to comparison samples
        mask = np.isin(conditions, [design.test_condition, design.reference_condition])
        y = values[mask]
        cond = conditions[mask]
        subj = subjects[mask] if subjects is not None else None

        # Fit model
        coef_df, model_type, resid_var, subj_var, converged, issue, cov_beta, n_obs, n_groups, df_residual = fit_linear_model(
            y, cond, subj,
            use_mixed=subj is not None,
            conditions=sorted([design.test_condition, design.reference_condition]),
        )

        # Extract contrast (test vs reference)
        # Reference is first alphabetically, so coefficient is test - reference
        test_coef_name = f"condition[T.{design.test_condition}]"
        if test_coef_name in coef_df.index:
            effect = coef_df.loc[test_coef_name, 'Coef.']
            se = coef_df.loc[test_coef_name, 'Std.Err.']
            t_stat = effect / se if se > 0 else 0.0
            p_val = coef_df.loc[test_coef_name, 'P>|t|']
        else:
            # Fallback to first non-intercept
            non_intercept = coef_df[~coef_df.index.str.contains('Intercept')]
            if len(non_intercept) > 0:
                effect = non_intercept['Coef.'].iloc[0]
                se = non_intercept['Std.Err.'].iloc[0]
                t_stat = effect / se if se > 0 else 0.0
                p_val = non_intercept['P>|t|'].iloc[0]
            else:
                raise ValueError("No condition coefficient found")

        n_test = np.sum(cond == design.test_condition)
        n_ref = np.sum(cond == design.reference_condition)

        return TestResult(
            feature_set_id="",
            effect_size=effect,
            test_statistic=t_stat,
            p_value=p_val,
            n_test=int(n_test),
            n_reference=int(n_ref),
            additional={
                'model_type': model_type.value if hasattr(model_type, 'value') else str(model_type),
                'converged': converged,
            },
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_c9_vs_sporadic_design(blocking_column: str = "subject_id") -> MetadataDerivedDesign:
    """
    Create experimental design for C9orf72 vs Sporadic ALS comparison.

    Based on AnswerALS metadata structure:
    - C9orf72 carriers: ClinReport_Mutations_Details contains "C9orf72"
    - Sporadic ALS: CASE with no known mutations

    Returns:
        MetadataDerivedDesign for C9 vs Sporadic comparison
    """
    def derive_genetic_subtype(row: pd.Series) -> str | None:
        # C9orf72 carrier
        mutations = str(row.get('ClinReport_Mutations_Details', ''))
        if 'C9orf72' in mutations:
            return 'C9orf72'

        # Sporadic: ALS case with no known mutations
        phenotype = row.get('phenotype', '')
        if phenotype == 'CASE' and (pd.isna(row.get('ClinReport_Mutations_Details')) or mutations == ''):
            return 'Sporadic'

        # Exclude other mutation carriers and controls
        return None

    return MetadataDerivedDesign(
        derivation_fn=derive_genetic_subtype,
        test_condition="C9orf72",
        reference_condition="Sporadic",
        blocking_column=blocking_column,
    )


def create_indra_neighbor_feature_sets(
    regulator: str,
    relationship_types: list[str] | None = None,
) -> list[SimpleFeatureSet]:
    """
    Create feature sets from INDRA 1-hop neighbors.

    For C9orf72 analysis per collaborator's request:
    - Get all 1-hop neighbors of C9orf72
    - Filter by relationship type (inc/dec/act/inh)
    - Create feature sets for differential testing

    Args:
        regulator: Regulator name (e.g., "C9orf72")
        relationship_types: Filter to these types (e.g., ["IncreaseAmount", "Activation"])

    Returns:
        List of SimpleFeatureSet for each relationship type
    """
    # This would integrate with INDRA/CoGEx
    # Placeholder for actual implementation
    raise NotImplementedError(
        "INDRA integration required. Use CoGEx to query:\n"
        f"  SELECT target, relationship FROM indra WHERE source = '{regulator}'"
    )
