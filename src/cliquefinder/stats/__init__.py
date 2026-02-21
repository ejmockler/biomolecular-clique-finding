"""
Statistical analysis module for proteomics and clique-level analysis.

This module provides MSstats-inspired statistical methods for:
- Differential abundance testing (protein and clique level)
- Data normalization (median, quantile)
- Missing value handling (AFT model for censored data)
- Protein/clique summarization (Tukey's Median Polish)
- Multiple testing correction (FDR)

The clique-level analysis extends MSstats methodology to protein groups,
treating co-regulated proteins as a single statistical unit.

Core Components:
    - correlation_tests: Differential correlation testing
    - differential: Linear mixed models for differential abundance
    - summarization: Tukey's Median Polish and aggregation methods
    - normalization: Sample normalization methods
    - missing: Missing value analysis and imputation
    - clique_analysis: Clique-level differential abundance

Example:
    >>> from cliquefinder.stats import run_clique_differential_analysis
    >>> result = run_clique_differential_analysis(
    ...     data=matrix.data,
    ...     feature_ids=protein_ids,
    ...     sample_metadata=metadata,
    ...     clique_definitions=cliques,
    ...     condition_col="treatment_group",
    ... )
    >>> df = result.to_dataframe()

References:
    - MSstats: Choi et al. (2014) Bioinformatics 30(17):2524-2526
    - MSstats v4: Kohler et al. (2023) J Proteome Res 22(5):1466-1482
"""

# Correlation testing (existing)
from .correlation_tests import (
    CorrelationTestResult,
    fisher_z_transform,
    inverse_fisher_z,
    fisher_z_standard_error,
    correlation_confidence_interval,
    test_correlation_difference,
    compute_significance_threshold,
    estimate_effective_tests,
    permutation_fdr,
    apply_fdr_correction,
)

# Summarization methods
from .summarization import (
    SummarizationMethod,
    MedianPolishResult,
    CliqueSummary,
    tukey_median_polish,
    summarize_to_protein,
    summarize_clique,
    parallel_clique_summarization,
)

# Normalization methods
from .normalization import (
    NormalizationMethod,
    NormalizationResult,
    normalize,
    median_normalization,
    quantile_normalization,
    global_standards_normalization,
    assess_normalization_quality,
)

# Missing value handling
from .missing import (
    MissingMechanism,
    ImputationMethod,
    MissingValueAnalysis,
    ImputationResult,
    analyze_missing_values,
    estimate_censoring_threshold,
    impute_missing_values,
    impute_aft_model,
    impute_qrilc,
    impute_knn,
)

# Differential analysis
from .differential import (
    NetworkEnrichmentResult,
    ModelType,
    ContrastResult,
    ProteinResult,
    DifferentialResult,
    fdr_correction,
    build_contrast_matrix,
    run_differential_analysis,
    run_protein_differential,
    run_network_enrichment_test,
)

# Clique-level analysis
from .clique_analysis import (
    CliqueDefinition,
    CliqueDifferentialResult,
    CliqueAnalysisResult,
    load_clique_definitions,
    run_clique_differential_analysis,
    run_clique_roast_analysis,  # ROAST rotation-based gene set test
    compare_protein_vs_clique_results,
    # Permutation-based significance testing (original)
    PermutationTestResult,
    run_permutation_clique_test,
    run_matched_single_gene_comparison,
)

# Generalized permutation framework (protocol-based)
# Note: create_c9_vs_sporadic_design moved to examples/ (experiment-specific)
from .permutation_framework import (
    FeatureSet,
    ExperimentalDesign,
    Summarizer,
    StatisticalTest,
    TwoGroupDesign,
    MetadataDerivedDesign,
    PermutationTestEngine,
    PermutationResult,
)

# ROAST: Rotation-based gene set testing
# Self-contained tests that preserve inter-gene correlation
from .rotation import (
    SetStatistic,
    Alternative,
    RotationPrecomputed,
    GeneEffects,
    RotationResult,
    RotationTestConfig,
    RotationTestEngine,
    run_rotation_test,
    compute_rotation_matrices,
    extract_gene_effects,
    generate_rotation_vectors,
    apply_rotations_batched,
    compute_set_statistics,
)

# Method comparison framework (cross-method concordance analysis)
from .method_comparison import (
    MethodName,
    UnifiedCliqueResult,
    ConcordanceMetrics,
    MethodComparisonResult,
    PreparedCliqueExperiment,
    run_method_comparison,
    prepare_experiment,
)

# Bootstrap subsampling for imbalanced designs
from .bootstrap_comparison import (
    BootstrapConfig,
    BootstrapCliqueResult,
    run_bootstrap_comparison,
)

__all__ = [
    # Correlation tests
    "CorrelationTestResult",
    "fisher_z_transform",
    "inverse_fisher_z",
    "fisher_z_standard_error",
    "correlation_confidence_interval",
    "test_correlation_difference",
    "compute_significance_threshold",
    "estimate_effective_tests",
    "permutation_fdr",
    "apply_fdr_correction",
    # Summarization
    "SummarizationMethod",
    "MedianPolishResult",
    "CliqueSummary",
    "tukey_median_polish",
    "summarize_to_protein",
    "summarize_clique",
    "parallel_clique_summarization",
    # Normalization
    "NormalizationMethod",
    "NormalizationResult",
    "normalize",
    "median_normalization",
    "quantile_normalization",
    "global_standards_normalization",
    "assess_normalization_quality",
    # Missing values
    "MissingMechanism",
    "ImputationMethod",
    "MissingValueAnalysis",
    "ImputationResult",
    "analyze_missing_values",
    "estimate_censoring_threshold",
    "impute_missing_values",
    "impute_aft_model",
    "impute_qrilc",
    "impute_knn",
    # Differential analysis
    "NetworkEnrichmentResult",
    "ModelType",
    "ContrastResult",
    "ProteinResult",
    "DifferentialResult",
    "fdr_correction",
    "build_contrast_matrix",
    "run_differential_analysis",
    "run_protein_differential",
    "run_network_enrichment_test",
    # Clique analysis
    "CliqueDefinition",
    "CliqueDifferentialResult",
    "CliqueAnalysisResult",
    "load_clique_definitions",
    "run_clique_differential_analysis",
    "compare_protein_vs_clique_results",
    # Permutation-based significance testing (original)
    "PermutationTestResult",
    "run_permutation_clique_test",
    "run_matched_single_gene_comparison",
    # Generalized permutation framework (protocol-based)
    "FeatureSet",
    "ExperimentalDesign",
    "Summarizer",
    "StatisticalTest",
    "TwoGroupDesign",
    "MetadataDerivedDesign",
    "PermutationTestEngine",
    "PermutationResult",
    # ROAST: Rotation-based gene set testing
    "SetStatistic",
    "Alternative",
    "RotationPrecomputed",
    "GeneEffects",
    "RotationResult",
    "RotationTestConfig",
    "RotationTestEngine",
    "run_rotation_test",
    "compute_rotation_matrices",
    "extract_gene_effects",
    "generate_rotation_vectors",
    "apply_rotations_batched",
    "compute_set_statistics",
    # Method comparison framework (cross-method concordance analysis)
    "MethodName",
    "UnifiedCliqueResult",
    "ConcordanceMetrics",
    "MethodComparisonResult",
    "PreparedCliqueExperiment",
    "run_method_comparison",
    "prepare_experiment",
    # Bootstrap subsampling for imbalanced designs
    "BootstrapConfig",
    "BootstrapCliqueResult",
    "run_bootstrap_comparison",
]
